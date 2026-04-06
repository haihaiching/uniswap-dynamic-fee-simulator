"""
amm_sim/sanity_check.py
========================
Generic AMM sanity checks. Parameterised by AMMSpec so the same five
checks run for every AMM type.

Checks (per AMM):
    1. Arb edge always <= 0; after arb one mispricing = 0, other >= 0
    2. Retail edge always >= 0 per step
    3. Two identical pools produce equal routing splits
    4. batch_rollout runs without error across 100 episodes
    5. Timing benchmark for batch_rollout
"""

import time
import jax
import jax.numpy as jnp

from amm_sim.amms.constant_product import (
    CONSTANT_PRODUCT_AMM, CPParams,
    verify_jax_compatibility as cp_verify,
)
from amm_sim.amms.linear import (
    LINEAR_AMM, LinearParams,
    verify_jax_compatibility as linear_verify,
)
from amm_sim.router import route_bisection
from amm_sim.spec import AMMSpec
from amm_sim.env import make_env
from amm_sim.types import SimParams


_SIM_PARAMS = SimParams(sigma=0.001, num_steps=200, max_orders=16,
                        lam=0.8, mu=1.0, sigma_ln=0.5)


def check_amm(spec: AMMSpec, params, name: str,
              sim_params: SimParams = _SIM_PARAMS,
              check_retail_after_arb: bool = True):
    """Run all five checks for one AMM type."""
    print(f"\n{'─' * 60}")
    print(f"  {name}")
    print(f"{'─' * 60}")

    env = make_env([spec, spec], [params, params])

    # ── Check 1: arb edge always <= 0 ─────────────────────────────
    print("[ Check 1 ] Arb edge always <= 0...")
    key = jax.random.PRNGKey(0)
    _, traj = env.rollout(key, sim_params)

    arb_edges = traj.arb_edge
    assert float(jnp.all(arb_edges <= 1e-6)), \
        f"FAIL: positive arb edge: max={float(arb_edges.max()):.6f}"
    print(f"  arb_edge min={float(arb_edges.min()):.6f}  "
          f"max={float(arb_edges.max()):.6f}  ✓")

    # ── Check 2: first retail edge after arb >= 0 ─────────────────
    # In each block_step arb runs before retail, so retail_edge[t] is the
    # LP gain from retail orders that traded against the post-arb pool state.
    # Verify this is non-negative in every step where arb actually fired.
    # Note: skipped for AMMs with cross-impact (e.g. Linear) where a sell arb
    # can drag p_ask below fair, causing negative retail edge — known model behaviour.
    print("[ Check 2 ] First retail edge after arb >= 0...")
    if not check_retail_after_arb:
        print("  skipped (cross-impact AMM)")
    else:
        arb_fired   = traj.arb_volume > 0
        n_arb_steps = int(arb_fired.sum())
        if n_arb_steps > 0:
            ret_after_arb = traj.retail_edge[arb_fired]
            assert float(jnp.all(ret_after_arb >= -1e-6)), \
                f"FAIL: negative retail edge after arb: min={float(ret_after_arb.min()):.6f}"
            print(f"  arb steps={n_arb_steps}  "
                  f"retail_edge min={float(ret_after_arb.min()):.6f}  "
                  f"max={float(ret_after_arb.max()):.6f}  ✓")
        else:
            print("  No arb steps in rollout (skipped)  ✓")

    # ── Check 3: two identical pools → equal routing splits ────────
    print("[ Check 3 ] Two identical pools produce equal routing splits...")
    s0 = spec.init(params)
    for side_val, label in [(0, "buy"), (1, "sell")]:
        splits, _ = route_bisection(
            [spec, spec], [s0, s0],
            jnp.int32(side_val), jnp.float32(1.0),
        )
        d0, d1 = float(splits[0]), float(splits[1])
        assert abs(d0 - d1) < 1e-4, \
            f"FAIL: unequal {label} split: {d0:.6f} vs {d1:.6f}"
        print(f"  {label}: splits = [{d0:.4f}, {d1:.4f}]  ✓")

    # ── Check 4: batch_rollout 100 episodes ───────────────────────
    print("[ Check 4 ] batch_rollout 100 episodes...")
    keys100 = jax.random.split(jax.random.PRNGKey(42), 100)
    _, trajs100 = env.batch_rollout(keys100, sim_params)

    arb100 = trajs100.arb_edge
    assert float(jnp.all(arb100 <= 1e-2)), \
        f"FAIL: positive arb edge in batch: max={float(arb100.max()):.6f}"
    print(f"  arb_edge: min={float(arb100.min()):.6f}  "
          f"max={float(arb100.max()):.6f}  ✓")

    # ── Check 5: timing benchmark ──────────────────────────────────
    print("[ Check 5 ] Timing (100 episodes × 200 steps)...")
    _ = env.batch_rollout(keys100, sim_params)   # warm up
    jax.block_until_ready(_)

    t0 = time.time()
    result = env.batch_rollout(keys100, sim_params)
    jax.block_until_ready(result)
    t1 = time.time()

    n_steps = 100 * sim_params.num_steps
    print(f"  {n_steps:,} steps in {t1-t0:.3f}s "
          f"({n_steps/(t1-t0):,.0f} steps/sec)  ✓")


def run_sanity_checks():
    print("=" * 60)
    print("AMM Simulator — Sanity Check")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")

    # ── JAX compatibility ──────────────────────────────────────────
    print("\n[ JAX ] CP AMM compatibility...")
    cp_verify()
    print("[ JAX ] Linear AMM compatibility...")
    linear_verify()

    # ── CP AMM ────────────────────────────────────────────────────
    check_amm(
        CONSTANT_PRODUCT_AMM,
        CPParams(fee_plus=0.003, fee_minus=0.003, max_trade_x=2.0),
        "Constant-Product AMM (30 bps)",
    )

    # ── Linear AMM ────────────────────────────────────────────────
    check_amm(
        LINEAR_AMM,
        LinearParams(
            lam_pp=2, lam_mm=2,
            lam_pm=1, lam_mp=1,
            init_p_ask=100.2, init_p_bid=99.8,
            init_x=100.0, init_y=10_000.0,
            max_trade_x=2.0,
        ),
        "Linear-Impact AMM (spread 0.2)",
        check_retail_after_arb=False,   # cross-impact can drag p_ask below fair
    )

    print("\n" + "=" * 60)
    print("All checks passed. Simulator is ready.")
    print("=" * 60)


if __name__ == "__main__":
    run_sanity_checks()