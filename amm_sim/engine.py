"""
amm_sim/engine.py
=================
Simulation engine. make_engine() returns a jit-compiled block_step.

One simulation cycle (block_step) runs three sub-steps in order:

    Step A — Oracle:    fair price moves via GBM, returns epsilon
    Step B — Arb:       each pool corrects stale prices independently (closed-form)
    Step C — Retail:    taker orders arrive, routed optimally via KKT bisection

Function signatures required by each pluggable component:

    arb_solver : (spec, state, fair_price, epsilon) → (side, delta_x)
        epsilon is the oracle log-return this step.
        Each AMM implements its own arb logic in amms/.

    edge_fn    : (state, side, delta_x, fair_price) → edge
        Each AMM implements its own edge formula in amms/.
        edge = cash received - fair value of inventory given up.

    routing    : handled by router.py, not per-AMM.
        If marginal_inverse_fns provided → route_bisection (KKT-optimal, N pools).
        Otherwise → route_bisection_numerical (works for any AMM via jax.grad).
        Both work for any number of pools — no per-AMM routing logic needed.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable

from amm_sim.spec import AMMSpec
from amm_sim.types import SimParams, EnvState, CycleRecord, update_metrics
from amm_sim.scoring import compute_edge
from amm_sim.router import route_bisection, route_bisection_numerical


# ── Default oracle ─────────────────────────────────────────────────────────

def default_oracle(fair_price, key, sim_params):
    """
    Zero-drift GBM: p(t+1) = p(t) * exp(-σ²/2 + σZ), Z ~ N(0,1).

    Returns (epsilon, new_price, key) where epsilon is the log return.
    epsilon is passed to arb solvers — Linear AMM uses it directly
    to compute how far spreads shifted this step.
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




def make_engine(amm_specs:            list[AMMSpec],
                arb_solvers:          list[Callable],
                edge_fns:             list[Callable],
                marginal_inverse_fns: list = None,
                oracle_fn:            Callable = None,
                retail_sampler:       Callable = None):
    """
    Build and return a jit-compiled block_step function.

    Parameters
    ----------
    amm_specs             : list of AMMSpec (one per pool)
    arb_solvers           : arb solver per pool — (spec, state, fair_price, epsilon) → (side, dx)
    edge_fns              : edge function per pool — (state, side, delta_x, fair_price) → edge
    marginal_inverse_fns  : optional list of (buy_inv_fn, sell_inv_fn) per pool
                            buy_inv_fn(state, nu) → delta
                            sell_inv_fn(state, nu) → delta
                            If provided: uses route_bisection (KKT-optimal).
                            If None: uses route_bisection_numerical (fallback, any AMM).
    oracle_fn             : optional custom oracle — defaults to GBM
    retail_sampler        : optional custom sampler — defaults to Poisson-LogNormal

    Returns
    -------
    block_step : jit-compiled (EnvState, SimParams) → (EnvState, CycleRecord)
    """
    if oracle_fn is None:
        oracle_fn = default_oracle
    if retail_sampler is None:
        retail_sampler = default_retail_sampler

    num_pools = len(amm_specs)   # static integer — Python loop unrolled at trace time

    # ── Build routing function ────────────────────────────────────────────
    # Routing: marginal_inverse_fns (KKT analytic) > numerical fallback
    # Works for N pools — Python loop unrolled at trace time (num_pools static)
 
    if marginal_inverse_fns is not None:
        # Analytic: each pool provides (buy_inv_fn, sell_inv_fn)
        # Returns list of N deltas
        def _route_fn(states, side, total_delta):
            is_buy = (side == 0)
            inv_fns = [
                lambda nu, i=i: jnp.where(
                    is_buy,
                    marginal_inverse_fns[i][0](states[i], nu),   # buy inverse
                    marginal_inverse_fns[i][1](states[i], nu),   # sell inverse
                )
                for i in range(num_pools)
            ]
            splits, _ = route_bisection(inv_fns, total_delta, nu_hi=1e4)
            return splits
 
    else:
        # Numerical fallback: works for any AMM type
        def _route_fn(states, side, total_delta):
            splits, _ = route_bisection_numerical(
                amm_specs, states, side, total_delta
            )
            return splits

    def block_step(state: EnvState, sim_params: SimParams):
        """
        One full simulation cycle. Called N times via lax.scan for a rollout.

        Step A: Oracle moves fair price by GBM.
        Step B: Each pool's arb solver corrects stale prices independently.
        Step C: Retail orders arrive, get split optimally, each pool executes.
        Step D: Package results into CycleRecord, update EnvState.
        """
        key = state.rng_key
        key, k_oracle, k_retail = jax.random.split(key, 3)

        # ── Step A: Oracle ─────────────────────────────────────────────────
        # epsilon = log return this step (used by Linear AMM arb solver)
        epsilon, new_fair, _ = oracle_fn(state.fair_price, k_oracle, sim_params)

        # ── Step B: Arbitrage ──────────────────────────────────────────────
        # Each pool corrected independently — no interaction between pools during arb.
        # Python loop is unrolled at trace time (num_pools is static).
        amm_states     = list(state.amm_states)
        total_arb_edge = jnp.float32(0.0)
        total_arb_vol  = jnp.float32(0.0)

        for i in range(num_pools):
            # epsilon passed to all arb solvers — Linear AMM uses it, CP AMM ignores it
            arb_side, arb_dx = arb_solvers[i](amm_specs[i], amm_states[i], new_fair, epsilon)
            new_s_i, arb_dy  = amm_specs[i].swap(amm_states[i], arb_side, arb_dx)
            # edge computed on pre-trade state (before swap changes reserves)
            arb_edge_i       = edge_fns[i](amm_states[i], arb_side, arb_dx, new_fair)

            amm_states[i]  = new_s_i
            total_arb_edge = total_arb_edge + arb_edge_i
            total_arb_vol  = total_arb_vol  + arb_dx

        # ── Step C: Retail flow ────────────────────────────────────────────
        # lax.scan over fixed-length order array (max_orders slots, inactive ones masked)
        sides, sizes, _ = retail_sampler(k_retail, sim_params)

        def process_order(carry, order):
            """
            Process one retail order slot.

            1. _route_fn finds optimal delta split (d1, d2) for this order.
            2. Each pool executes its delta independently.
            3. Edge computed on pre-trade state for accuracy.
            4. State updates masked to zero for inactive slots (size=0).
            """
            sts, ret_edge, ret_vol = carry
            side, total_size = order
            active = total_size > 0.0   # False for zero-masked (inactive) slots

            # Optimal split: maximises taker's total output
            deltas = _route_fn(sts, side, total_size)

            new_sts    = list(sts)
            order_edge = jnp.float32(0.0)

            for j in range(num_pools):
                delta_j       = deltas[j]
                new_s_j, dy_j = amm_specs[j].swap(sts[j], side, delta_j)
                # Use pre-trade state for edge calculation
                edge_j        = edge_fns[j](sts[j], side, delta_j, new_fair)

                # jnp.where mask: inactive slots keep old state unchanged
                # (JAX Issue 3 fix: no dynamic indexing into Python list)
                new_sts[j] = jax.tree.map(
                    lambda n, o: jnp.where(active, n, o),
                    new_s_j, sts[j]
                )
                order_edge = order_edge + edge_j * active   # zero for inactive slots

            return (new_sts, ret_edge + order_edge, ret_vol + total_size * active), None

        init_carry = (amm_states, jnp.float32(0.0), jnp.float32(0.0))
        (amm_states, total_ret_edge, total_ret_vol), _ = jax.lax.scan(
            process_order,
            init_carry,
            (sides, sizes),   # scanned over max_orders slots
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
        )

        # reserve_x is the common inventory field for both AMM types
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