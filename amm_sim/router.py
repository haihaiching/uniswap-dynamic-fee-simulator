"""
amm_sim/router.py
=================
Optimal order routing via KKT bisection on the Lagrange multiplier ν.

Theory:
    Problem:  max Σᵢ fᵢ(Δᵢ)  s.t.  Σᵢ Δᵢ = Δ,  Δᵢ ≥ 0
    KKT:      fᵢ'(Δᵢ*) = ν  for active pools (Δᵢ* > 0)
              fᵢ'(0)   ≤ ν  for inactive pools (Δᵢ* = 0)

    Solution: bisect on ν until g(ν) = Σᵢ fᵢ'⁻¹(ν) = Δ

    g(ν) monotonicity differs by side:
        Buy  (fᵢ' increasing in Δ): min cost → larger ν → larger Δᵢ* → g increases.
        Sell (fᵢ' decreasing in Δ): max recv → larger ν → smaller Δᵢ* → g decreases.

    Inner bisection inverts fᵢ'(Δ) = ν using jax.grad — exact marginal,
    no finite-difference approximation.

Uses lax.fori_loop (not lax.scan) to avoid ConcretizationTypeError when called
inside engine.py's retail lax.scan.
"""

import jax
import jax.numpy as jnp

from amm_sim.spec import AMMSpec


def route_bisection(
    specs:           list[AMMSpec],
    states:          list,
    side:            jnp.ndarray,
    total_delta:     jnp.ndarray,
    num_iters_outer: int   = 32,
    num_iters_inner: int   = 32,
    nu_hi:           float = 1e6,
) -> tuple[list[jnp.ndarray], jnp.ndarray]:
    """
    Find optimal allocation across N pools via nested bisection.

    Outer bisection: on shadow price ν until Σᵢ fᵢ'⁻¹(ν) = total_delta.
    Inner bisection: per pool, invert fᵢ'(Δ) = ν using jax.grad marginals.

    Works for any AMM type — no analytic inverse functions required.
    Uses lax.fori_loop throughout for JAX compatibility inside lax.scan.

    Parameters
    ----------
    specs           : list of AMMSpec, one per pool
    states          : list of AMM states, one per pool
    side            : 0 = buy, 1 = sell
    total_delta     : total order size to split across pools
    num_iters_outer : outer bisection iterations on ν (static int)
    num_iters_inner : inner bisection iterations per pool (static int)
    nu_hi           : upper bound on ν (static float)

    Returns
    -------
    splits  : list of per-pool allocations, one per pool, summing to total_delta
    nu_star : optimal shadow price ν*
    """
    is_buy    = (side == 0)
    num_pools = len(specs)   # static — Python loop unrolled at trace time

    def marginal_output(spec, state, delta):
        """Marginal output via jax.grad — only the relevant side is evaluated."""
        return jax.lax.cond(
            is_buy,
            lambda: jax.grad(spec.curve_buy,  argnums=1)(state, delta),
            lambda: jax.grad(spec.curve_sell, argnums=1)(state, delta),
        )

    # Precompute per-pool bisection upper bounds once — these depend only on
    # spec/state/total_delta, not on ν, so recomputing them inside marginal_inverse
    # on every outer iteration is wasteful.
    uppers = [
        jnp.minimum(
            total_delta,
            jnp.where(is_buy, specs[i].max_trade_buy(states[i]),
                              specs[i].max_trade_sell(states[i]))
        )
        for i in range(num_pools)
    ]

    def marginal_inverse(spec, state, upper, nu):
        """
        Find Δ* such that fᵢ'(Δ*) = ν via inner bisection.

        Buy  (fᵢ' increasing): fᵢ'(mid) > ν → mid too large → hi = mid.
        Sell (fᵢ' decreasing): fᵢ'(mid) > ν → mid too small → lo = mid.

        Upper bound: min(total_delta, cap_i) — no pool receives more than
        the full order or its own inventory cap. When the bisection converges
        to the cap, the pool is saturated and the outer bisection absorbs the
        remaining flow into other pools via g(ν) = Σ min(capᵢ, fᵢ'⁻¹(ν)).
        """
        def inner_body(i, carry):
            lo, hi   = carry
            mid      = (lo + hi) * jnp.float32(0.5)
            f_mid    = marginal_output(spec, state, mid)
            above_nu = f_mid > nu
            # Buy: above_nu → hi=mid (too large); Sell: above_nu → lo=mid (too small)
            towards_hi = jnp.where(is_buy, ~above_nu, above_nu)
            lo = jnp.where(towards_hi, mid, lo)
            hi = jnp.where(towards_hi, hi,  mid)
            return lo, hi

        lo_f, hi_f = jax.lax.fori_loop(
            0, num_iters_inner, inner_body,
            (jnp.float32(0.0), upper)
        )
        return jnp.maximum((lo_f + hi_f) * jnp.float32(0.5), jnp.float32(0.0))

    def g(nu):
        """Total allocation at shadow price ν (increasing for buy, decreasing for sell)."""
        total = jnp.float32(0.0)
        for i in range(num_pools):   # unrolled at trace time
            total = total + marginal_inverse(specs[i], states[i], uppers[i], nu)
        return total

    def outer_body(i, carry):
        lo, hi  = carry
        mid     = (lo + hi) * jnp.float32(0.5)
        g_mid   = g(mid)
        g_above = g_mid > total_delta
        # Buy  (g increasing): g > Δ → ν too large  → hi = mid
        # Sell (g decreasing): g > Δ → ν too small  → lo = mid
        towards_hi = jnp.where(is_buy, ~g_above, g_above)
        lo = jnp.where(towards_hi, mid, lo)
        hi = jnp.where(towards_hi, hi,  mid)
        return lo, hi

    lo_f, hi_f = jax.lax.fori_loop(
        0, num_iters_outer, outer_body,
        (jnp.float32(0.0), jnp.float32(nu_hi))
    )
    nu_star = (lo_f + hi_f) * jnp.float32(0.5)
    splits  = [marginal_inverse(specs[i], states[i], uppers[i], nu_star)
               for i in range(num_pools)]

    return splits, nu_star