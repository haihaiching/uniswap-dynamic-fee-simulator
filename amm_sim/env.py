"""
amm_sim/env.py
==============
Environment wrapper. Gymnax-compatible API:

    reset(key, sim_params) → (obs, state)
    step(state, action, sim_params) → (obs, state, reward, done, info)
    rollout(key, sim_params) → (final_state, trajectory)
    batch_rollout(keys, sim_params) → (final_states, trajectories)

make_env() is the main entry point.

Works with any AMM type that has reserve_x and reserve_y fields
(both CP AMM and Linear AMM satisfy this after v2 update).
"""

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from typing import Callable

from amm_sim.spec import AMMSpec
from amm_sim.types import SimParams, EnvState, CycleRecord, zero_metrics
from amm_sim.engine import make_engine


# ══════════════════════════════════════════════════════════════════
# OBSERVATION BUILDER
# ══════════════════════════════════════════════════════════════════

def make_obs(state: EnvState, sim_params: SimParams) -> jnp.ndarray:
    """
    Build a 3-dim observation compatible with all AMM types.

    obs = [fair_price, inventory (reserve_x), time_fraction]

    Both CP AMM and Linear AMM expose reserve_x.
    """
    pool0     = state.amm_states[0]
    time_frac = state.step_idx.astype(jnp.float32) / sim_params.num_steps

    return jnp.array([
        state.fair_price,
        pool0.reserve_x,
        time_frac,
    ], dtype=jnp.float32)


# ══════════════════════════════════════════════════════════════════
# ENV FACTORY
# ══════════════════════════════════════════════════════════════════

def make_env(amm_specs:      list[AMMSpec],
             amm_params:     list,
             arb_solvers:    list[Callable],
             edge_fns:       list[Callable],
             route_fn:       Callable = None,
             oracle_fn:      Callable = None,
             retail_sampler: Callable = None):
    """
    Build the environment.

    Parameters
    ----------
    amm_specs      : list of AMMSpec (one per pool)
    amm_params     : list of params objects for each spec.init
    arb_solvers    : list of arb solver functions (one per pool)
                     signature: (spec, state, fair_price, epsilon) → (side, dx)
    route_fn       : routing function
                     signature: (state1, state2, side, total_delta) → (d1, d2)
    edge_fns       : list of edge functions (one per pool)
                     signature: (state, side, delta_x) → edge
    oracle_fn      : optional custom oracle
    retail_sampler : optional custom retail sampler
    """
    block_step = make_engine(
        amm_specs, arb_solvers, edge_fns, route_fn,
        oracle_fn, retail_sampler
    )
    num_pools = len(amm_specs)

    # ── reset ──────────────────────────────────────────────────────
    def reset(key, sim_params):
        key, subkey = jax.random.split(key)
        init_states = [amm_specs[i].init(amm_params[i]) for i in range(num_pools)]
        s0 = init_states[0]
        p0 = s0.reserve_y / s0.reserve_x   # works for both AMM types

        state = EnvState(
            amm_states=init_states,
            fair_price=jnp.float32(p0),
            step_idx=jnp.int32(0),
            rng_key=subkey,
            metrics=zero_metrics(),
        )
        return make_obs(state, sim_params), state

    # ── step ───────────────────────────────────────────────────────
    def step(state, action, sim_params):
        """
        action is accepted for Gymnax compatibility but not applied
        to AMM state here — use a policy_fn in rollout() to update
        AMM parameters before each step.
        """
        new_state, record = block_step(state, sim_params)

        obs    = make_obs(new_state, sim_params)
        done   = new_state.step_idx >= sim_params.num_steps
        reward = record.total_edge - sim_params.phi * new_state.metrics.inventory**2

        return obs, new_state, reward, done, record

    # ── rollout ────────────────────────────────────────────────────
    def rollout(key, sim_params, policy_fn=None):
        """
        Run a full episode via lax.scan.

        policy_fn(obs, key) → action  (optional)
        If None, AMM parameters are held fixed throughout.
        """
        _, init_state = reset(key, sim_params)

        def scan_step(carry, _):
            state, rng = carry
            rng, act_key = jax.random.split(rng)
            obs    = make_obs(state, sim_params)
            action = policy_fn(obs, act_key) if policy_fn is not None \
                     else jnp.zeros(2, dtype=jnp.float32)
            _, new_state, reward, done, record = step(state, action, sim_params)
            return (new_state, rng), record

        (final_state, _), trajectory = jax.lax.scan(
            scan_step,
            (init_state, key),
            None,
            length=sim_params.num_steps,
        )
        return final_state, trajectory

    # ── batch_rollout ──────────────────────────────────────────────
    def batch_rollout(keys, sim_params, policy_fn=None):
        """Run N independent episodes in parallel via vmap."""
        batched = vmap(
            partial(rollout, sim_params=sim_params, policy_fn=policy_fn),
            in_axes=0
        )
        return batched(keys)

    class Env:
        pass

    env = Env()
    env.reset         = reset
    env.step          = step
    env.rollout       = rollout
    env.batch_rollout = batch_rollout
    env.block_step    = block_step

    return env