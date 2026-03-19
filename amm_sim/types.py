"""
amm_sim/types.py
================
Shared data structures used across the entire simulator.

All fields must have static shapes at construction time
so that jit / vmap / lax.scan work correctly.
"""

import jax.numpy as jnp
import chex
from typing import Any


# ══════════════════════════════════════════════════════════════════
# STATIC PARAMETERS  (passed as sim_params, never on device)
# ══════════════════════════════════════════════════════════════════

@chex.dataclass(frozen=True)
class SimParams:
    """
    Static simulation parameters fixed at construction time.
    These are NOT JAX arrays — they live in Python and control
    compile-time constants like loop lengths.

    Attributes
    ----------
    sigma       : GBM volatility per step
    num_steps   : episode length (static — controls lax.scan length)
    max_orders  : maximum retail orders per step (static — controls
                  fixed-length order array; Poisson arrivals are masked)
    lam         : Poisson arrival rate for retail orders
    mu          : LogNormal mean for order size (in Y terms)
    sigma_ln    : LogNormal std for order size (fixed at 1.2 per spec)
    phi         : inventory penalty coefficient in reward
    """
    sigma:      float = 0.001
    num_steps:  int   = 1_000
    max_orders: int   = 16
    lam:        float = 0.8
    mu:         float = 20.0
    sigma_ln:   float = 1.2
    phi:        float = 0.0


# ══════════════════════════════════════════════════════════════════
# PER-STEP OUTPUT RECORD
# ══════════════════════════════════════════════════════════════════

@chex.dataclass(frozen=True)
class CycleRecord:
    """
    Output of one simulation cycle (block_step).
    All fields are scalar JAX arrays.

    Attributes
    ----------
    fair_price   : oracle price at end of this cycle
    epsilon      : oracle price move this cycle
    arb_edge     : total arb edge across all pools (≤ 0)
    retail_edge  : total retail edge across all pools (≥ 0 typically)
    total_edge   : arb_edge + retail_edge
    arb_volume   : total X volume from arbitrage
    retail_volume: total X volume from retail orders
    """
    fair_price:    jnp.ndarray   # scalar
    epsilon:       jnp.ndarray   # scalar
    arb_edge:      jnp.ndarray   # scalar, ≤ 0
    retail_edge:   jnp.ndarray   # scalar
    total_edge:    jnp.ndarray   # scalar
    arb_volume:    jnp.ndarray   # scalar
    retail_volume: jnp.ndarray   # scalar


# ══════════════════════════════════════════════════════════════════
# RUNNING METRICS  (accumulated inside EnvState)
# ══════════════════════════════════════════════════════════════════

@chex.dataclass(frozen=True)
class Metrics:
    """
    Scalar accumulators carried inside EnvState.
    Updated every cycle; all fields are scalar JAX arrays.

    Attributes
    ----------
    total_edge      : cumulative total edge since episode start
    total_arb_edge  : cumulative arb edge
    total_ret_edge  : cumulative retail edge
    inventory       : current X inventory of the agent's pool
                      (positive = long X, negative = short X)
    """
    total_edge:     jnp.ndarray   # scalar
    total_arb_edge: jnp.ndarray   # scalar
    total_ret_edge: jnp.ndarray   # scalar
    inventory:      jnp.ndarray   # scalar


def zero_metrics() -> Metrics:
    """Return a zeroed Metrics struct for episode reset."""
    return Metrics(
        total_edge=jnp.float32(0.0),
        total_arb_edge=jnp.float32(0.0),
        total_ret_edge=jnp.float32(0.0),
        inventory=jnp.float32(0.0),
    )


def update_metrics(metrics: Metrics, record: CycleRecord,
                   agent_inventory: jnp.ndarray) -> Metrics:
    """
    Accumulate one cycle's record into the running metrics.

    Parameters
    ----------
    metrics         : current running metrics
    record          : output of this cycle's block_step
    agent_inventory : current X reserve of the agent's pool (pool 0)
    """
    return Metrics(
        total_edge=metrics.total_edge       + record.total_edge,
        total_arb_edge=metrics.total_arb_edge + record.arb_edge,
        total_ret_edge=metrics.total_ret_edge  + record.retail_edge,
        inventory=agent_inventory,
    )


# ══════════════════════════════════════════════════════════════════
# ENVIRONMENT STATE  (full simulator state passed through lax.scan)
# ══════════════════════════════════════════════════════════════════

@chex.dataclass(frozen=True)
class EnvState:
    """
    Complete environment state. Passed into and out of every block_step.

    All fields must have static shapes so that vmap over episodes works.

    Attributes
    ----------
    amm_states  : list of AMM states, one per pool.
                  Length = num_pools (static).
                  For heterogeneous pools this is a Python list of
                  different-shaped pytrees; for homogeneous pools it
                  can be a single stacked pytree.
    fair_price  : current oracle fair price (scalar)
    step_idx    : current step index (scalar int)
    rng_key     : JAX PRNG key for this step
    metrics     : running accumulators
    """
    amm_states: Any              # list[pytree], length = num_pools (static)
    fair_price: jnp.ndarray      # scalar float32
    step_idx:   jnp.ndarray      # scalar int32
    rng_key:    jnp.ndarray      # shape (2,) uint32
    metrics:    Metrics
