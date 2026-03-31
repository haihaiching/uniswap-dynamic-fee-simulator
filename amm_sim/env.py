"""
amm_sim/env.py
==============
Gymnax-compatible environment wrapper around the engine.

API:
    reset(key, sim_params)                    → (obs, state)
    step(state, action, sim_params)           → (obs, state, reward, done, info)
    rollout(key, sim_params, policy_fn=None)  → (final_state, trajectory)
    batch_rollout(keys, sim_params)           → (final_states, trajectories)

Works with any AMM type that exposes reserve_x and reserve_y fields.
Both CP AMM and Linear AMM satisfy this requirement.
"""

import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from typing import Callable

from amm_sim.spec import AMMSpec
from amm_sim.types import SimParams, EnvState, CycleRecord, zero_metrics
from amm_sim.engine import make_engine


# ── Observation builder ────────────────────────────────────────────────────

def make_obs(state: EnvState, sim_params: SimParams) -> jnp.ndarray:
    """
    Build the observation vector from current environment state.

    obs = [fair_price, reserve_x (inventory proxy), time_fraction]
    obs_dim = 3 — fixed shape required for vmap over episodes.

    Intentionally minimal: only fields that exist in both CP and Linear AMM.
    Extend this for RL training by adding AMM-specific fields as needed.
    """
    pool0     = state.amm_states[0]
    time_frac = state.step_idx.astype(jnp.float32) / sim_params.num_steps

    return jnp.array([
        state.fair_price,   # oracle price — the "true" external price
        pool0.reserve_x,    # inventory of pool 0 (agent's pool)
                            # CP AMM: X token reserves
                            # Linear AMM: kept in sync with x (inventory) in swap
        time_frac,          # how far through the episode we are (0 to 1)
    ], dtype=jnp.float32)


# ── Environment factory ────────────────────────────────────────────────────

def make_env(amm_specs:            list[AMMSpec],
             amm_params:            list,
             arb_solvers:           list[Callable],
             edge_fns:              list[Callable],
             marginal_inverse_fns:  list = None,
             oracle_fn:             Callable = None,
             retail_sampler:        Callable = None):
    """
    Build the simulation environment.

    Parameters
    ----------
    amm_specs             : list of AMMSpec (one per pool)
    amm_params            : initial params for each AMM (passed to spec.init)
    arb_solvers           : arb solver per pool — (spec, state, fair_price, epsilon) → (side, dx)
    edge_fns              : edge function per pool — (state, side, delta_x, fair_price) → edge
    marginal_inverse_fns  : optional list of (buy_inv_fn, sell_inv_fn) per pool
                            Enables KKT-optimal routing via route_bisection.
                            If None: falls back to route_bisection_numerical.
    oracle_fn             : optional custom oracle (defaults to GBM)
    retail_sampler        : optional custom order sampler (defaults to Poisson-LogNormal)

    Returns
    -------
    env object with .reset / .step / .rollout / .batch_rollout / .block_step
    """
    # Build the jit-compiled step function from the engine
    block_step = make_engine(
        amm_specs, arb_solvers, edge_fns, marginal_inverse_fns,
        oracle_fn, retail_sampler
    )
    num_pools = len(amm_specs)

    def reset(key, sim_params):
        """
        Initialise all AMM states for a new episode.

        Initial fair price = reserve_y / reserve_x of pool 0.
        This works for both AMM types:
          CP AMM:     reserve_y / reserve_x = initial spot price
          Linear AMM: reserve_y / reserve_x = init_y / init_x (set in params)
        """
        key, subkey = jax.random.split(key)
        # Call each AMM's init function with its params
        init_states = [amm_specs[i].init(amm_params[i]) for i in range(num_pools)]
        s0 = init_states[0]
        p0 = s0.reserve_y / s0.reserve_x   # initial fair price

        state = EnvState(
            amm_states=init_states,
            fair_price=jnp.float32(p0),
            step_idx=jnp.int32(0),
            rng_key=subkey,
            metrics=zero_metrics(),
        )
        return make_obs(state, sim_params), state

    def step(state, action, sim_params):
        """
        Run one simulation cycle.

        action is accepted for Gymnax API compatibility but currently
        not applied — AMM parameters stay fixed unless a policy_fn
        explicitly modifies state before calling block_step.

        Returns (obs, new_state, reward, done, info) where:
          reward = total_edge - phi * inventory²
          info   = CycleRecord (arb_edge, retail_edge, fair_price, etc.)
        """
        new_state, record = block_step(state, sim_params)

        obs    = make_obs(new_state, sim_params)
        done   = new_state.step_idx >= sim_params.num_steps
        # reward = edge minus inventory penalty (phi=0 → pure edge maximisation)
        reward = record.total_edge - sim_params.phi * new_state.metrics.inventory**2

        return obs, new_state, reward, done, record

    def rollout(key, sim_params, policy_fn=None):
        """
        Run a full episode of num_steps cycles via lax.scan.

        lax.scan replaces a Python for loop — the entire episode is
        compiled into a single XLA computation, which is much faster
        than calling step() in a Python loop.

        policy_fn(obs, key) → action  (optional)
        If None, all AMM parameters stay fixed throughout the episode.

        Returns (final_state, trajectory) where trajectory is a
        CycleRecord with each field having shape (num_steps,).
        """
        _, init_state = reset(key, sim_params)

        def scan_step(carry, _):
            state, rng = carry
            rng, act_key = jax.random.split(rng)
            obs    = make_obs(state, sim_params)
            action = policy_fn(obs, act_key) if policy_fn is not None \
                     else jnp.zeros(2, dtype=jnp.float32)
            _, new_state, reward, done, record = step(state, action, sim_params)
            return (new_state, rng), record   # record accumulated by lax.scan

        (final_state, _), trajectory = jax.lax.scan(
            scan_step,
            (init_state, key),
            None,
            length=sim_params.num_steps,   # static length required by lax.scan
        )
        return final_state, trajectory

    def batch_rollout(keys, sim_params, policy_fn=None):
        """
        Run N independent episodes in parallel via vmap.

        keys: (N, 2) array — one PRNG key per episode.

        vmap transforms rollout() to operate over a batch of keys simultaneously.
        All N episodes run in a single GPU forward pass — this is where
        the simulator's speed advantage over CPU-based simulators comes from.

        Returns (final_states, trajectories) where each trajectory field
        has shape (N, num_steps).
        """
        batched = vmap(
            partial(rollout, sim_params=sim_params, policy_fn=policy_fn),
            in_axes=0   # vectorise over the first axis (batch of keys)
        )
        return batched(keys)

    # Package into a simple namespace object
    class Env:
        pass

    env = Env()
    env.reset         = reset
    env.step          = step
    env.rollout       = rollout
    env.batch_rollout = batch_rollout
    env.block_step    = block_step   # expose raw block_step for debugging

    return env