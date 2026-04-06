"""
amm_sim/engine.py
=================
Simulation engine. make_engine() returns a jit-compiled block_step.

One simulation cycle (block_step) runs three sub-steps in order:

    Step A — Oracle:  fair price moves via GBM, returns epsilon
    Step B — Arb:     each pool corrects stale prices via generic_arb_solver
    Step C — Retail:  taker orders arrive, routed optimally via KKT bisection

Per-AMM arb and edge logic are no longer arguments — the engine calls
generic_arb_solver (arb.py) and compute_edge (scoring.py) for all pools
uniformly using only the AMMSpec interface.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable

from amm_sim.spec import AMMSpec
from amm_sim.types import SimParams, EnvState, CycleRecord, update_metrics
from amm_sim.scoring import compute_edge
from amm_sim.router import route_bisection
from amm_sim.arb import generic_arb_solver


# ── Default oracle ─────────────────────────────────────────────────────────

def default_oracle(fair_price, key, sim_params):
    """
    Zero-drift GBM: p(t+1) = p(t) * exp(-σ²/2 + σZ), Z ~ N(0,1).

    Returns (epsilon, new_price, key) where epsilon is the log return.
    """
    key, subkey = jax.random.split(key)
    Z         = jax.random.normal(subkey)
    sigma     = sim_params.sigma
    epsilon   = -0.5 * sigma**2 + sigma * Z   # log return
    new_price = fair_price * jnp.exp(epsilon)
    return epsilon, new_price, key


# ── Default retail sampler ─────────────────────────────────────────────────

def default_retail_sampler(key, sim_params):
    """
    Generate retail orders for one step.

    Arrival count: N ~ Poisson(lam), clipped to max_orders (static shape).
    Size:          Y ~ LogNormal(mu, sigma_ln).
    Direction:     50% buy (side=0), 50% sell (side=1).

    Always returns arrays of length max_orders. Inactive slots have
    size=0. When delta_x=0, swap is a no-op — reserves unchanged, edge=0.
    This is the JAX-compatible way to handle variable-length sequences.
    """
    M = sim_params.max_orders
    key, k1, k2, k3 = jax.random.split(key, 4)

    n_raw = jax.random.poisson(k1, sim_params.lam)
    n     = jnp.minimum(n_raw, M).astype(jnp.int32)

    Z     = jax.random.normal(k2, shape=(M,))
    sizes = jnp.exp(sim_params.mu + sim_params.sigma_ln * Z)   # LogNormal sizes
    sides = jax.random.bernoulli(k3, p=0.5, shape=(M,)).astype(jnp.int32)

    # Zero out inactive slots — size=0 → no-op swap
    mask  = (jnp.arange(M) < n).astype(jnp.float32)
    sizes = sizes * mask

    return sides, sizes, key


def make_engine(amm_specs:      list[AMMSpec],
                oracle_fn:      Callable = None,
                retail_sampler: Callable = None):
    """
    Build and return a jit-compiled block_step function.

    Parameters
    ----------
    amm_specs      : list of AMMSpec (one per pool)
    oracle_fn      : optional custom oracle — defaults to GBM
    retail_sampler : optional custom sampler — defaults to Poisson-LogNormal

    Returns
    -------
    block_step : jit-compiled (EnvState, SimParams) → (EnvState, CycleRecord)
    """
    if oracle_fn is None:
        oracle_fn = default_oracle
    if retail_sampler is None:
        retail_sampler = default_retail_sampler

    num_pools = len(amm_specs)   # static integer — Python loop unrolled at trace time

    def block_step(state: EnvState, sim_params: SimParams):
        """
        One full simulation cycle. Called N times via lax.scan for a rollout.

        Step A: Oracle moves fair price by GBM.
        Step B: Each pool corrected independently by generic_arb_solver.
        Step C: Retail orders arrive, split optimally, each pool executes.
        Step D: Package results into CycleRecord, update EnvState.
        """
        key = state.rng_key
        key, k_oracle, k_retail = jax.random.split(key, 3)

        # ── Step A: Oracle ─────────────────────────────────────────────────
        epsilon, new_fair, _ = oracle_fn(state.fair_price, k_oracle, sim_params)

        # ── Step B: Arbitrage ──────────────────────────────────────────────
        # Each pool corrected independently — no interaction between pools.
        # Python loop unrolled at trace time (num_pools is static).
        # Edge computed on pre-trade state; dy from swap is on pre-trade state.
        amm_states     = list(state.amm_states)
        total_arb_edge = jnp.float32(0.0)
        total_arb_vol  = jnp.float32(0.0)
        arb_edge_pp    = jnp.zeros(num_pools, dtype=jnp.float32)

        for i in range(num_pools):
            arb_side, arb_dx       = generic_arb_solver(amm_specs[i], amm_states[i], new_fair)
            new_s_i, arb_dy        = amm_specs[i].swap(amm_states[i], arb_side, arb_dx)
            arb_edge_i             = compute_edge(arb_side, arb_dx, arb_dy, new_fair)

            amm_states[i]  = new_s_i
            total_arb_edge = total_arb_edge + arb_edge_i
            total_arb_vol  = total_arb_vol  + arb_dx
            arb_edge_pp    = arb_edge_pp.at[i].set(arb_edge_i)

        # ── Step C: Retail flow ────────────────────────────────────────────
        # lax.scan over fixed-length order array (max_orders slots, inactive masked).
        sides, sizes, _ = retail_sampler(k_retail, sim_params)

        def process_order(carry, order):
            """
            Process one retail order slot.

            1. route_bisection splits total_size optimally across pools.
            2. Each pool executes its allocation independently.
            3. Edge computed using dy returned by swap (pre-trade state).
            4. State updates masked to zero for inactive slots (size=0).
            """
            sts, ret_edge, ret_vol, ret_edge_pp = carry
            side, total_size = order
            active = total_size > 0.0   # False for zero-masked inactive slots

            deltas, _ = route_bisection(amm_specs, list(sts), side, total_size)

            new_sts    = list(sts)
            order_edge = jnp.float32(0.0)

            for j in range(num_pools):
                delta_j       = deltas[j]
                new_s_j, dy_j = amm_specs[j].swap(sts[j], side, delta_j)
                edge_j        = compute_edge(side, delta_j, dy_j, new_fair)

                # Mask inactive slots — keep old state unchanged for size=0 orders
                new_sts[j] = jax.tree.map(
                    lambda n, o: jnp.where(active, n, o),
                    new_s_j, sts[j]
                )
                order_edge  = order_edge + edge_j * active
                ret_edge_pp = ret_edge_pp.at[j].add(edge_j * active)

            return (new_sts, ret_edge + order_edge, ret_vol + total_size * active, ret_edge_pp), None

        init_carry = (amm_states, jnp.float32(0.0), jnp.float32(0.0),
                      jnp.zeros(num_pools, dtype=jnp.float32))
        (amm_states, total_ret_edge, total_ret_vol, ret_edge_pp), _ = jax.lax.scan(
            process_order,
            init_carry,
            (sides, sizes),
        )

        # ── Step D: Package ────────────────────────────────────────────────
        record = CycleRecord(
            fair_price=new_fair,
            epsilon=epsilon,
            arb_edge=total_arb_edge,
            retail_edge=total_ret_edge,
            total_edge=total_arb_edge + total_ret_edge,
            arb_volume=total_arb_vol,
            retail_volume=total_ret_vol,
            arb_edges_per_pool=arb_edge_pp,
            retail_edges_per_pool=ret_edge_pp,
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

    # jit-compile with sim_params as static arg (controls loop lengths)
    return jit(block_step, static_argnums=(1,))