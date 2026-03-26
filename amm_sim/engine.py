"""
amm_sim/engine.py
=================
Simulation engine. make_engine() returns a jit-compiled block_step.

Cycle:  Market Shift → Arbitrage → Retail Flow

Changes in this version:
    - arb_solver signature: (spec, state, fair_price, epsilon) → (side, dx)
      * epsilon is the oracle price move this step, needed by Linear AMM.
      * CP AMM arb solver accepts epsilon but ignores it.
    - Routing: uses route_fn (one per pair of pools) for retail flow.
      * route_fn(state1, state2, side, total_delta) → (delta1, delta2)
      * This supports the Linear AMM analytical optimal split.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable

from amm_sim.spec import AMMSpec
from amm_sim.types import SimParams, EnvState, CycleRecord, update_metrics
from amm_sim.scoring import compute_edge


# ══════════════════════════════════════════════════════════════════
# DEFAULT PLUGGABLE COMPONENTS
# ══════════════════════════════════════════════════════════════════

def default_oracle(fair_price, key, sim_params):
    """
    Zero-drift GBM.
    Returns (epsilon, new_price, key).
    """
    key, subkey = jax.random.split(key)
    Z         = jax.random.normal(subkey)
    sigma     = sim_params.sigma
    epsilon   = -0.5 * sigma**2 + sigma * Z
    new_price = fair_price * jnp.exp(epsilon)
    return epsilon, new_price, key


def default_retail_sampler(key, sim_params):
    """
    Poisson arrivals, LogNormal sizes.
    Returns fixed-length arrays (sides, sizes) of length max_orders.
    Inactive slots have size=0 (no-op).
    """
    M = sim_params.max_orders
    key, k1, k2, k3 = jax.random.split(key, 4)

    n_raw = jax.random.poisson(k1, sim_params.lam)
    n     = jnp.minimum(n_raw, M).astype(jnp.int32)

    Z     = jax.random.normal(k2, shape=(M,))
    sizes = jnp.exp(sim_params.mu + sim_params.sigma_ln * Z)
    sides = jax.random.bernoulli(k3, p=0.5, shape=(M,)).astype(jnp.int32)

    mask  = (jnp.arange(M) < n).astype(jnp.float32)
    sizes = sizes * mask

    return sides, sizes, key


# ══════════════════════════════════════════════════════════════════
# ENGINE FACTORY
# ══════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
# OPTIMAL ROUTING  (numerical bisection, works for any AMM type)
# ══════════════════════════════════════════════════════════════════

def route_two_pools(spec1, state1, spec2, state2,
                    side: jnp.ndarray,
                    total_delta: jnp.ndarray,
                    n_iter: int = 32):
    """
    Optimal two-pool routing via bisection.

    Maximises total output received by taker:
        Buy  (side=0): max total X received, split Y across pools
        Sell (side=1): max total Y received, split X across pools

    Bisects on marginal output equality:
        marginal_output_pool1(d1) = marginal_output_pool2(d2)

    Works for any AMM — only uses spec.curve_buy and spec.curve_sell.
    Uses lax.fori_loop for JAX compatibility.
    """
    is_buy = (side == 0)
    eps    = total_delta * jnp.float32(1e-5) + jnp.float32(1e-8)

    def get_output(spec, state, delta):
        return jnp.where(is_buy,
                         spec.curve_buy(state,  delta),
                         spec.curve_sell(state, delta))

    def marginal_diff(d1):
        d2 = total_delta - d1
        m1 = (get_output(spec1, state1, d1 + eps) - get_output(spec1, state1, d1)) / eps
        m2 = (get_output(spec2, state2, d2)       - get_output(spec2, state2, d2 - eps)) / eps
        return m1 - m2

    lo = jnp.float32(0.0)
    hi = total_delta * jnp.float32(0.9999)

    def bisect_body(i, carry):
        lo, hi = carry
        mid   = (lo + hi) * jnp.float32(0.5)
        f_mid = marginal_diff(mid)
        lo    = jnp.where(f_mid > 0, mid, lo)
        hi    = jnp.where(f_mid < 0, mid, hi)
        return lo, hi

    lo, hi = jax.lax.fori_loop(0, n_iter, bisect_body, (lo, hi))
    delta1 = jnp.clip((lo + hi) * jnp.float32(0.5), 0.0, total_delta)
    delta2 = total_delta - delta1
    return delta1, delta2


def make_engine(amm_specs:      list[AMMSpec],
                arb_solvers:    list[Callable],
                edge_fns:       list[Callable],
                route_fn:       Callable = None,
                oracle_fn:      Callable = None,
                retail_sampler: Callable = None):
    """
    Build and return a jit-compiled block_step function.

    Parameters
    ----------
    amm_specs      : list of AMMSpec, one per pool (static length = 2)
    arb_solvers    : list of arb solver functions, one per pool
                     signature: (spec, state, fair_price, epsilon) → (side, dx)
    route_fn       : routing function for retail orders
                     signature: (state1, state2, side, total_delta)
                                → (delta1, delta2)
    edge_fns       : list of edge functions, one per pool
                     signature: (state, side, delta_x) → edge
    oracle_fn      : (price, key, params) → (epsilon, new_price, key)
    retail_sampler : (key, params) → (sides, sizes, key)

    Returns
    -------
    block_step : (EnvState, SimParams) → (EnvState, CycleRecord)
    """
    if oracle_fn is None:
        oracle_fn = default_oracle
    if retail_sampler is None:
        retail_sampler = default_retail_sampler

    num_pools = len(amm_specs)   # static, unrolled at trace time

    # If no route_fn provided, use the universal bisection router
    _spec1, _spec2 = amm_specs[0], amm_specs[1]

    if route_fn is None:
        def _route_fn(state1, state2, side, total_delta):
            return route_two_pools(_spec1, state1, _spec2, state2, side, total_delta)
    else:
        def _route_fn(state1, state2, side, total_delta):
            return route_fn(state1, state2, side, total_delta)

    def block_step(state: EnvState, sim_params: SimParams):
        key = state.rng_key
        key, k_oracle, k_retail = jax.random.split(key, 3)

        # ── Step A: Oracle ─────────────────────────────────────────
        epsilon, new_fair, _ = oracle_fn(state.fair_price, k_oracle, sim_params)

        # ── Step B: Arbitrage (one per pool, unrolled) ─────────────
        amm_states     = list(state.amm_states)
        total_arb_edge = jnp.float32(0.0)
        total_arb_vol  = jnp.float32(0.0)

        for i in range(num_pools):
            # Pass epsilon to arb solver — Linear AMM uses it,
            # CP AMM accepts it but ignores it
            arb_side, arb_dx = arb_solvers[i](
                amm_specs[i], amm_states[i], new_fair, epsilon
            )
            new_s_i, arb_dy = amm_specs[i].swap(
                amm_states[i], arb_side, arb_dx
            )
            arb_edge_i = edge_fns[i](amm_states[i], arb_side, arb_dx, new_fair)

            amm_states[i]  = new_s_i
            total_arb_edge = total_arb_edge + arb_edge_i
            total_arb_vol  = total_arb_vol  + arb_dx

        # ── Step C: Retail flow (lax.scan) ─────────────────────────
        sides, sizes, _ = retail_sampler(k_retail, sim_params)

        def process_order(carry, order):
            """
            Process one retail order slot.

            1. route_fn splits total_delta across pools.
            2. Each pool executes its own delta independently.
            3. Edge computed via each pool's edge_fn.
            Inactive slots (size=0) are no-ops.
            """
            sts, ret_edge, ret_vol = carry
            side, total_size = order
            active = total_size > 0.0

            # Route: optimal split across pools
            delta1, delta2 = _route_fn(sts[0], sts[1], side, total_size)

            deltas = [delta1, delta2]

            # Execute on each pool
            new_sts    = list(sts)
            order_edge = jnp.float32(0.0)

            for j in range(num_pools):
                delta_j       = deltas[j]
                new_s_j, dy_j = amm_specs[j].swap(sts[j], side, delta_j)
                edge_j        = edge_fns[j](sts[j], side, delta_j, new_fair)

                # Commit only if slot is active
                new_sts[j] = jax.tree.map(
                    lambda n, o: jnp.where(active, n, o),
                    new_s_j, sts[j]
                )
                order_edge = order_edge + edge_j * active

            return (new_sts, ret_edge + order_edge,
                    ret_vol + total_size * active), None

        init_carry = (amm_states, jnp.float32(0.0), jnp.float32(0.0))
        (amm_states, total_ret_edge, total_ret_vol), _ = jax.lax.scan(
            process_order,
            init_carry,
            (sides, sizes),
        )

        # ── Step D: Package ────────────────────────────────────────
        record = CycleRecord(
            fair_price=new_fair,
            epsilon=epsilon,
            arb_edge=total_arb_edge,
            retail_edge=total_ret_edge,
            total_edge=total_arb_edge + total_ret_edge,
            arb_volume=total_arb_vol,
            retail_volume=total_ret_vol,
        )

        agent_inventory = amm_states[0].reserve_x

        new_state = state.replace(
            amm_states=amm_states,
            fair_price=new_fair,
            step_idx=state.step_idx + jnp.int32(1),
            rng_key=key,
            metrics=update_metrics(state.metrics, record, agent_inventory),
        )

        return new_state, record

    return jit(block_step, static_argnums=(1,))