"""
amm_sim/engine.py
=================
Simulation engine. make_engine() returns a jit-compiled block_step.

Cycle:  Market Shift → Arbitrage → Retail Flow

Changes in this version:
    - arb_solver signature: (spec, state, fair_price, epsilon) → (side, dx)
      epsilon is the oracle price move this step, needed by Linear AMM.
      CP AMM arb solver accepts epsilon but ignores it.
    - Routing: uses route_fn (one per pair of pools) for retail flow.
      route_fn(state1, state2, side, total_delta) → (delta1, delta2)
      This supports the Linear AMM analytical optimal split.
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

def make_engine(amm_specs:      list[AMMSpec],
                arb_solvers:    list[Callable],
                route_fn:       Callable,
                edge_fns:       list[Callable],
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
            arb_edge_i = edge_fns[i](amm_states[i], arb_side, arb_dx)

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
            delta1, delta2 = route_fn(sts[0], sts[1], side, total_size)

            deltas = [delta1, delta2]

            # Execute on each pool
            new_sts    = list(sts)
            order_edge = jnp.float32(0.0)

            for j in range(num_pools):
                delta_j       = deltas[j]
                new_s_j, dy_j = amm_specs[j].swap(sts[j], side, delta_j)
                edge_j        = edge_fns[j](sts[j], side, delta_j)

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

        agent_inventory = amm_states[0].x

        new_state = state.replace(
            amm_states=amm_states,
            fair_price=new_fair,
            step_idx=state.step_idx + jnp.int32(1),
            rng_key=key,
            metrics=update_metrics(state.metrics, record, agent_inventory),
        )

        return new_state, record

    return jit(block_step, static_argnums=(1,))