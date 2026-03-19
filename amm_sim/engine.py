"""
amm_sim/engine.py
=================
Simulation engine. make_engine() returns a jit-compiled block_step
function that runs one full simulation cycle:

    Market Shift → Arbitrage → Retail Flow

All three breaking JAX issues from the spec §4 are handled here:
    Issue 2: fixed-length order arrays with zero-mask
    Issue 3: compute-all-select-one pool selection
    Issue 1: num_pools fixed at construction, Python loop unrolled
"""

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Callable

from amm_sim.spec import AMMSpec, marginal_ask, marginal_bid
from amm_sim.types import SimParams, EnvState, CycleRecord, Metrics, update_metrics
from amm_sim.scoring import compute_edge


# ══════════════════════════════════════════════════════════════════
# DEFAULT PLUGGABLE COMPONENTS
# ══════════════════════════════════════════════════════════════════

def default_oracle(fair_price: jnp.ndarray,
                   key: jnp.ndarray,
                   sim_params: SimParams):
    """
    Zero-drift GBM oracle.

        p(t+1) = p(t) * exp(-sigma^2/2 + sigma*Z),  Z ~ N(0,1)

    Returns (epsilon, new_fair_price).
    """
    key, subkey = jax.random.split(key)
    Z         = jax.random.normal(subkey)
    sigma     = sim_params.sigma
    log_ret   = -0.5 * sigma**2 + sigma * Z
    new_price = fair_price * jnp.exp(log_ret)
    return log_ret, new_price, key


def default_retail_sampler(key: jnp.ndarray,
                           sim_params: SimParams):
    """
    Poisson arrivals with LogNormal sizes.

    Arrival count: N ~ Poisson(lam), clipped to max_orders.
    Size:          Y ~ LogNormal(mu, sigma_ln), denominated in X.
    Direction:     side ~ Bernoulli(0.5),  0=buy, 1=sell.

    Always returns arrays of length max_orders.
    Inactive slots have size=0.0 (no-op swap).

    Returns (sides, sizes, new_key).
    """
    M = sim_params.max_orders
    key, k1, k2, k3 = jax.random.split(key, 4)

    # Number of active orders
    n_raw = jax.random.poisson(k1, sim_params.lam)
    n     = jnp.minimum(n_raw, M).astype(jnp.int32)

    # Order sizes: LogNormal in X units
    Z     = jax.random.normal(k2, shape=(M,))
    sizes = jnp.exp(sim_params.mu + sim_params.sigma_ln * Z)

    # Directions
    sides = jax.random.bernoulli(k3, p=0.5, shape=(M,)).astype(jnp.int32)

    # Mask inactive slots
    mask  = (jnp.arange(M) < n).astype(jnp.float32)
    sizes = sizes * mask   # inactive → size=0 → no-op swap

    return sides, sizes, key


# ══════════════════════════════════════════════════════════════════
# ENGINE FACTORY
# ══════════════════════════════════════════════════════════════════

def make_engine(amm_specs:       list[AMMSpec],
                arb_solvers:     list[Callable],
                oracle_fn:       Callable = None,
                retail_sampler:  Callable = None):
    """
    Build and return a jit-compiled block_step function.

    Parameters
    ----------
    amm_specs      : list of AMMSpec, one per pool. Length = num_pools (static).
    arb_solvers    : list of arb solver functions, one per pool.
    oracle_fn      : (fair_price, key, sim_params) → (epsilon, new_price, key)
                     Defaults to GBM.
    retail_sampler : (key, sim_params) → (sides, sizes, key)
                     Defaults to Poisson-LogNormal.

    Returns
    -------
    block_step : Callable
        (EnvState, SimParams) → (EnvState, CycleRecord)
        JIT-compiled. Can be used directly inside lax.scan.
    """
    if oracle_fn is None:
        oracle_fn = default_oracle
    if retail_sampler is None:
        retail_sampler = default_retail_sampler

    num_pools = len(amm_specs)   # static integer — unrolled at trace time

    def block_step(state: EnvState,
                   sim_params: SimParams) -> tuple[EnvState, CycleRecord]:
        """
        One full simulation cycle.

        Step 0: Split PRNG key.
        Step A: Oracle updates fair price.
        Step B: Arbitrage — one arb solver per pool (Python loop, unrolled).
        Step C: Retail flow — fixed-length lax.scan over order array.
        Step D: Update metrics and package state.
        """
        key = state.rng_key

        # ── Step 0: PRNG splits ────────────────────────────────────
        key, k_oracle, k_retail = jax.random.split(key, 3)

        # ── Step A: Market shift (oracle) ──────────────────────────
        epsilon, new_fair, _ = oracle_fn(state.fair_price, k_oracle, sim_params)

        # ── Step B: Arbitrage (unrolled Python loop) ───────────────
        amm_states     = list(state.amm_states)   # mutable Python list
        total_arb_edge = jnp.float32(0.0)
        total_arb_vol  = jnp.float32(0.0)

        for i in range(num_pools):
            spec_i  = amm_specs[i]
            solver_i = arb_solvers[i]

            arb_side, arb_dx = solver_i(spec_i, amm_states[i], new_fair)
            new_s_i, arb_dy  = spec_i.swap(amm_states[i], arb_side, arb_dx)

            arb_edge_i = compute_edge(arb_side, arb_dx, arb_dy, new_fair)

            amm_states[i]  = new_s_i
            total_arb_edge = total_arb_edge + arb_edge_i
            total_arb_vol  = total_arb_vol  + arb_dx

        # ── Step C: Retail flow (lax.scan) ─────────────────────────
        sides, sizes, _ = retail_sampler(k_retail, sim_params)

        def process_order(carry, order):
            """
            Process one retail order slot.

            Routing: best marginal price (argmin ask / argmax bid).
            Execution: compute-all-select-one (Issue 3 fix).
            Inactive slots (size=0) are no-ops.
            """
            sts, ret_edge, ret_vol = carry
            side, size = order
            active = size > 0.0   # False for zero-masked slots

            # Compute marginal prices for all pools
            asks = jnp.array([marginal_ask(amm_specs[j], sts[j])
                               for j in range(num_pools)])
            bids = jnp.array([marginal_bid(amm_specs[j], sts[j])
                               for j in range(num_pools)])

            # Route: buy → lowest ask;  sell → highest bid
            pool_idx = jnp.where(side == 0,
                                 jnp.argmin(asks),
                                 jnp.argmax(bids))

            # Execute on ALL pools, mask to selected (Issue 3: Approach A)
            new_sts    = list(sts)
            order_edge = jnp.float32(0.0)

            for j in range(num_pools):
                new_s_j, dy_j = amm_specs[j].swap(sts[j], side, size)
                selected = (pool_idx == j) & active

                # Commit state only for selected pool
                new_sts[j] = jax.tree.map(
                    lambda n, o: jnp.where(selected, n, o),
                    new_s_j, sts[j]
                )
                order_edge = order_edge + compute_edge(
                    side, size, dy_j, new_fair
                ) * selected

            return (new_sts, ret_edge + order_edge, ret_vol + size * active), None

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

        # Agent inventory = pool 0's X reserve
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
