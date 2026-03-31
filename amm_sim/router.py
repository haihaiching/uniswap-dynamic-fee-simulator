"""
amm_sim/router.py
=================
Optimal order routing via KKT bisection on the Lagrange multiplier ν.

Theory:
    Problem:  max Σᵢ fᵢ(Δᵢ)  s.t.  Σᵢ Δᵢ = Δ,  Δᵢ ≥ 0
    KKT:      fᵢ'(Δᵢ*) = ν  for active pools (Δᵢ* > 0)
              fᵢ'(0)   ≤ ν  for inactive pools (Δᵢ* = 0)

    Solution: bisect on ν until g(ν) = Σᵢ fᵢ'⁻¹(ν) = Δ

Interface:
    marginal_inverse_fn(nu) → delta_i
    State is closed over in engine._route_fn — router does not need to know about state.

Uses lax.fori_loop (not lax.scan) to avoid ConcretizationTypeError when called
inside engine.py's retail lax.scan — fori_loop handles traced carry correctly.
"""

import jax
import jax.numpy as jnp
from typing import Callable


# ══════════════════════════════════════════════════════════════════
# ANALYTIC ROUTER
# ══════════════════════════════════════════════════════════════════

def route_bisection(
    marginal_inverse_fns: list[Callable],   # list of (nu) → delta_i  (state closed over)
    total_delta: jnp.ndarray,
    num_iters: int = 32,
    nu_hi: float = 1e6,
) -> tuple[list[jnp.ndarray], jnp.ndarray]:
    """
    Find optimal allocation across N pools via bisection on shadow price ν.

    g(ν) = Σᵢ fᵢ'⁻¹(ν) is strictly decreasing.
    Bisect to find ν* such that g(ν*) = total_delta.

    Uses lax.fori_loop instead of lax.scan to avoid ConcretizationTypeError
    when called inside engine.py's retail order lax.scan.

    Parameters
    ----------
    marginal_inverse_fns : list of fn(nu) → delta_i
        State already closed over — each fn only takes nu.
    total_delta  : total order size Δ to split (can be traced)
    num_iters    : bisection iterations (static Python int)
    nu_hi        : upper bound on ν (static Python float)
    """
    num_pools = len(marginal_inverse_fns)

    def g(nu):
        # Total allocation at shadow price nu
        total = jnp.float32(0.0)
        for i in range(num_pools):   # unrolled at trace time (num_pools is static)
            total = total + marginal_inverse_fns[i](nu)
        return total

    def bisect_body(i, carry):
        lo, hi = carry
        mid   = (lo + hi) * jnp.float32(0.5)
        g_mid = g(mid)
        lo = jnp.where(g_mid > total_delta, mid, lo)
        hi = jnp.where(g_mid > total_delta, hi,  mid)
        return lo, hi

    lo_f, hi_f = jax.lax.fori_loop(
        jnp.int32(0), jnp.int32(num_iters), bisect_body,
        (jnp.float32(0.0), jnp.float32(float(nu_hi)))
    )
    nu_star = (lo_f + hi_f) * jnp.float32(0.5)
    splits  = [marginal_inverse_fns[i](nu_star) for i in range(num_pools)]

    return splits, nu_star


# ══════════════════════════════════════════════════════════════════
# NUMERICAL FALLBACK
# ══════════════════════════════════════════════════════════════════

def route_bisection_numerical(
    specs: list,
    states: list,
    side: jnp.ndarray,
    total_delta: jnp.ndarray,
    num_iters_outer: int = 32,
    num_iters_inner: int = 32,
) -> tuple[list[jnp.ndarray], jnp.ndarray]:
    """
    Fallback router using numerical marginal inverse via nested bisection.
    Uses lax.fori_loop throughout for JAX compatibility inside lax.scan.
    """
    is_buy = (side == 0)
    eps    = total_delta * jnp.float32(1e-5) + jnp.float32(1e-8)

    def curve_output(spec, state, delta):
        return jnp.where(is_buy,
                         spec.curve_buy(state,  delta),
                         spec.curve_sell(state, delta))

    def marginal_output(spec, state, delta):
        return (curve_output(spec, state, delta + eps)
                - curve_output(spec, state, delta)) / eps

    def numerical_marginal_inverse(spec, state, nu):
        lo = jnp.float32(0.0)
        hi = total_delta * jnp.float32(0.9999)

        def inner_body(i, carry):
            lo, hi = carry
            mid   = (lo + hi) * jnp.float32(0.5)
            f_mid = marginal_output(spec, state, mid)
            lo = jnp.where(f_mid > nu, mid, lo)
            hi = jnp.where(f_mid > nu, hi,  mid)
            return lo, hi

        lo_f, hi_f = jax.lax.fori_loop(jnp.int32(0), jnp.int32(num_iters_inner), inner_body, (lo, hi))
        return jnp.maximum((lo_f + hi_f) * jnp.float32(0.5), 0.0)

    # Build (nu) → delta_i functions with state closed over
    num_pools = len(specs)
    inv_fns = [
        (lambda s=specs[i], st=states[i]: lambda nu: numerical_marginal_inverse(s, st, nu))()
        for i in range(num_pools)
    ]

    return route_bisection(inv_fns, total_delta, int(num_iters_outer))