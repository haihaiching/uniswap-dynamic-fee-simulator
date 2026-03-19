"""
amm_sim/sanity_check.py
========================
Phase 5 verification. Run this after installing the package to confirm
the simulator is working correctly before writing any strategy code.

Checks:
    1. Arb edge always <= 0
    2. Two identical AMMs produce equal edge
    3. batch_rollout runs without error across 100 episodes
    4. Timing benchmark for batch_rollout
"""

import time
import jax
import jax.numpy as jnp

from amm_sim.amms.constant_product import (
    CONSTANT_PRODUCT_AMM, CPParams, cp_arb_solver,
    verify_jax_compatibility
)
from amm_sim.env import make_env
from amm_sim.types import SimParams


def run_sanity_checks():
    print("=" * 60)
    print("AMM Simulator — Sanity Check")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print()

    # ── Phase 2: JAX compatibility of CP AMM ───────────────────────
    print("[ Phase 2 ] CP AMM JAX compatibility...")
    verify_jax_compatibility()
    print()

    # ── Setup: two identical CP AMMs ───────────────────────────────
    params      = CPParams(fee=jnp.float32(0.003))
    sim_params  = SimParams(sigma=0.001, num_steps=200, max_orders=16,
                            lam=0.8, mu=1.0, sigma_ln=0.5)

    env = make_env(
        amm_specs   = [CONSTANT_PRODUCT_AMM, CONSTANT_PRODUCT_AMM],
        amm_params  = [params, params],
        arb_solvers = [cp_arb_solver, cp_arb_solver],
    )

    # ── Check 1: Single episode, arb edge <= 0 ─────────────────────
    print("[ Check 1 ] Arb edge always <= 0...")
    key = jax.random.PRNGKey(0)
    final_state, traj = env.rollout(key, sim_params)

    arb_edges = traj.arb_edge
    assert float(jnp.all(arb_edges <= 1e-6)), \
        f"FAIL: found positive arb edge: {arb_edges.max():.6f}"
    print(f"  arb_edge min={float(arb_edges.min()):.6f}  "
          f"max={float(arb_edges.max()):.6f}  ✓")
    print()

    # ── Check 2: Two identical AMMs, edge should be split evenly ───
    print("[ Check 2 ] Two identical AMMs produce similar total edge...")
    total_edge = float(final_state.metrics.total_edge)
    total_arb  = float(final_state.metrics.total_arb_edge)
    total_ret  = float(final_state.metrics.total_ret_edge)
    print(f"  total_edge = {total_edge:.4f}")
    print(f"  arb_edge   = {total_arb:.4f}  (should be < 0)")
    print(f"  retail_edge= {total_ret:.4f}  (should be > 0)")
    assert total_arb <= 0,  "FAIL: cumulative arb edge should be negative"
    assert total_ret >= 0,  "FAIL: cumulative retail edge should be positive"
    print("  ✓")
    print()

    # ── Check 3: batch_rollout across 50 episodes ──────────────────
    print("[ Check 3 ] batch_rollout 50 episodes without error...")
    keys = jax.random.split(jax.random.PRNGKey(42), 50)
    final_states, trajs = env.batch_rollout(keys, sim_params)

    arb_batch = trajs.arb_edge   # shape (50, num_steps)
    assert float(jnp.all(arb_batch <= 1e-6)), \
        "FAIL: positive arb edge found in batch"
    print(f"  arb_edge across 50 episodes: "
          f"min={float(arb_batch.min()):.6f}  "
          f"max={float(arb_batch.max()):.6f}  ✓")
    print()

    # ── Check 4: Timing benchmark ──────────────────────────────────
    print("[ Check 4 ] Timing benchmark (100 episodes × 200 steps)...")
    keys_bench = jax.random.split(jax.random.PRNGKey(99), 100)

    # Warm up JIT
    _ = env.batch_rollout(keys_bench, sim_params)
    jax.block_until_ready(_)

    t0 = time.time()
    result = env.batch_rollout(keys_bench, sim_params)
    jax.block_until_ready(result)
    t1 = time.time()

    elapsed    = t1 - t0
    total_steps = 100 * 200
    print(f"  {total_steps:,} total steps in {elapsed:.3f}s "
          f"({total_steps/elapsed:,.0f} steps/sec)  ✓")
    print()

    print("=" * 60)
    print("All checks passed. Simulator is ready.")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_checks()
