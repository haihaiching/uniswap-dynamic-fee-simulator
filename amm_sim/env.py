"""
amm_sim/env.py
==============
Environment wrapper around the engine.
Exposes the standard Gymnax-compatible API:

    reset(key, sim_params) → (obs, state)
    step(state, action, sim_params) → (obs, state, reward, done, info)
    rollout(key, sim_params) → (final_state, trajectory)
    batch_rollout(keys, sim_params) → (final_states, trajectories)

make_env() is the main entry point.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Callable

from amm_sim.spec import AMMSpec
from amm_sim.types import SimParams, EnvState, CycleRecord, Metrics, zero_metrics
from amm_sim.engine import make_engine


# ══════════════════════════════════════════════════════════════════
# OBSERVATION BUILDER
# ══════════════════════════════════════════════════════════════════

def make_obs(state: EnvState, sim_params: SimParams) -> jnp.ndarray:
    """
    Build the observation vector from the current environment state.

    obs = [
        agent_ask,        # marginal ask of pool 0 (agent's pool)
        agent_bid,        # marginal bid of pool 0
        fair_price,       # oracle price
        inventory,        # agent's current X reserve
        time_fraction,    # step_idx / num_steps
    ]
    obs_dim = 5 (fixed, static shape required for vmap)

    Note: marginal ask/bid are approximated as spot price ± spread.
    For CP AMM: spot = reserve_y / reserve_x.
    """
    pool0   = state.amm_states[0]
    spot    = pool0.reserve_y / pool0.reserve_x
    gamma   = 1.0 - pool0.fee

    # Approximate marginal ask/bid from spot and fee
    marginal_ask = spot / gamma    # buyer pays more
    marginal_bid = spot * gamma    # seller receives less

    time_frac = state.step_idx.astype(jnp.float32) / sim_params.num_steps

    return jnp.array([
        marginal_ask,
        marginal_bid,
        state.fair_price,
        pool0.reserve_x,          # inventory proxy
        time_frac,
    ], dtype=jnp.float32)


# ══════════════════════════════════════════════════════════════════
# ENV FACTORY
# ══════════════════════════════════════════════════════════════════

def make_env(amm_specs:      list[AMMSpec],
             amm_params:     list,
             arb_solvers:    list[Callable],
             oracle_fn:      Callable = None,
             retail_sampler: Callable = None):
    """
    Build the environment and return the four standard API functions.

    Parameters
    ----------
    amm_specs      : list of AMMSpec (one per pool, length = num_pools)
    amm_params     : list of params objects passed to each spec.init
    arb_solvers    : list of arb solver functions (one per pool)
    oracle_fn      : optional custom oracle
    retail_sampler : optional custom retail sampler

    Returns
    -------
    env : object with .reset / .step / .rollout / .batch_rollout
    """
    block_step = make_engine(amm_specs, arb_solvers, oracle_fn, retail_sampler)
    num_pools  = len(amm_specs)

    # ── reset ──────────────────────────────────────────────────────
    def reset(key: jnp.ndarray,
              sim_params: SimParams) -> tuple[jnp.ndarray, EnvState]:
        """
        Initialise all AMM states and return (obs, state).
        """
        key, subkey = jax.random.split(key)

        init_states  = [amm_specs[i].init(amm_params[i]) for i in range(num_pools)]
        # Initial fair price = spot of pool 0
        s0 = init_states[0]
        p0 = s0.reserve_y / s0.reserve_x

        state = EnvState(
            amm_states=init_states,
            fair_price=jnp.float32(p0),
            step_idx=jnp.int32(0),
            rng_key=subkey,
            metrics=zero_metrics(),
        )
        obs = make_obs(state, sim_params)
        return obs, state

    # ── step ───────────────────────────────────────────────────────
    def step(state: EnvState,
             action: jnp.ndarray,
             sim_params: SimParams) -> tuple:
        """
        Gymnax-compatible step.

        action = [new_ask, new_bid] for the agent's pool (pool 0).
        Applied BEFORE block_step runs.

        Returns (obs, new_state, reward, done, info).
        """
        # Apply action: update pool 0's fee to reflect the quoted spread.
        # Implied fee from ask/bid:  ask = spot/gamma, bid = spot*gamma
        # → gamma ≈ sqrt(bid/ask),  fee = 1 - gamma
        new_ask = action[0]
        new_bid = action[1]
        implied_gamma = jnp.sqrt(new_bid / (new_ask + 1e-8))
        implied_fee   = jnp.clip(1.0 - implied_gamma, 1e-4, 0.1)

        pool0_new = state.amm_states[0].replace(fee=implied_fee)
        amm_states_new = [pool0_new] + list(state.amm_states[1:])
        state = state.replace(amm_states=amm_states_new)

        # Run one cycle
        new_state, record = block_step(state, sim_params)

        obs    = make_obs(new_state, sim_params)
        done   = new_state.step_idx >= sim_params.num_steps
        reward = record.total_edge - sim_params.phi * new_state.metrics.inventory**2
        info   = record

        return obs, new_state, reward, done, info

    # ── rollout ────────────────────────────────────────────────────
    def rollout(key: jnp.ndarray,
                sim_params: SimParams,
                policy_fn: Callable = None) -> tuple[EnvState, list[CycleRecord]]:
        """
        Run a full episode via lax.scan.

        policy_fn : (obs, key) → action
                    If None, uses a no-op (action=[ask, bid] unchanged).

        Returns (final_state, trajectory) where trajectory is a
        CycleRecord with leading time dimension (num_steps,).
        """
        _, init_state = reset(key, sim_params)

        def scan_step(carry, _):
            state, rng = carry
            rng, act_key = jax.random.split(rng)

            obs = make_obs(state, sim_params)

            if policy_fn is not None:
                action = policy_fn(obs, act_key)
            else:
                # No-op: keep current ask/bid
                pool0 = state.amm_states[0]
                spot  = pool0.reserve_y / pool0.reserve_x
                gamma = 1.0 - pool0.fee
                action = jnp.array([spot / gamma, spot * gamma])

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
    def batch_rollout(keys: jnp.ndarray,
                      sim_params: SimParams,
                      policy_fn: Callable = None) -> tuple:
        """
        Run N independent episodes in parallel via vmap.

        keys : (N, 2) array of PRNG keys — one per episode.

        Returns (final_states, trajectories) where trajectories has
        leading batch dimension (N, num_steps, ...).

        This is where the framework's speed advantage comes from:
        all N episodes run in a single GPU forward pass.
        """
        batched = vmap(
            partial(rollout, sim_params=sim_params, policy_fn=policy_fn),
            in_axes=0
        )
        return batched(keys)

    # ── package as simple namespace ────────────────────────────────
    class Env:
        pass

    env = Env()
    env.reset         = reset
    env.step          = step
    env.rollout       = rollout
    env.batch_rollout = batch_rollout
    env.block_step    = block_step   # expose for direct use

    return env
