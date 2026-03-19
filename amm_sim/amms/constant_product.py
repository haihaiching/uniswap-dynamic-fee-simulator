"""
amm_sim/amms/constant_product.py
=================================
Constant-product AMM (x·y = k) with fee-on-input.

Matches the paper spec exactly:
    gamma = 1 - fee    (fee multiplier)

    Sell delta_x of X → effective input = delta_x * gamma
    Buy  delta_x of X → buyer pays delta_y / gamma  (fee on Y)

Arb solver uses the closed-form solution from the paper:
    Arb buy:   x' = sqrt(k / (gamma * fair_price))
    Arb sell:  x' = sqrt(gamma * k / fair_price)
"""

import jax
import jax.numpy as jnp
import chex
from amm_sim.spec import AMMSpec


# ══════════════════════════════════════════════════════════════════
# PARAMS & STATE
# ══════════════════════════════════════════════════════════════════

@chex.dataclass(frozen=True)
class CPParams:
    """
    Static parameters for a Constant-Product AMM.
    Embedded inside CPState so vmap copies them per episode.
    (Overhead: 3 floats × n_episodes — acceptable per spec §Issue 7.)
    """
    fee:    jnp.float32 = jnp.float32(0.003)    # 30 bps default
    init_x: jnp.float32 = jnp.float32(100.0)
    init_y: jnp.float32 = jnp.float32(10_000.0)


@chex.dataclass(frozen=True)
class CPState:
    """
    Mutable state of a Constant-Product AMM.

    Attributes
    ----------
    reserve_x : token X reserve
    reserve_y : token Y reserve
    fee       : fee rate (copied from params for convenience)

    Derived quantities (not stored):
        k     = reserve_x * reserve_y   (invariant)
        spot  = reserve_y / reserve_x
        gamma = 1 - fee
    """
    reserve_x: jnp.float32
    reserve_y: jnp.float32
    fee:       jnp.float32


# ══════════════════════════════════════════════════════════════════
# FOUR CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def cp_init(params: CPParams) -> CPState:
    """Build initial CPState from CPParams."""
    return CPState(
        reserve_x=params.init_x,
        reserve_y=params.init_y,
        fee=params.fee,
    )


def cp_curve_buy(state: CPState, delta_x: jnp.float32) -> jnp.float32:
    """
    READ-ONLY. How much Y must the buyer pay to receive delta_x of X?

    Derivation (fee-on-output, Y side):
        k       = x * y
        x_new   = x - delta_x
        y_new   = k / x_new
        delta_y_gross = y_new - y
        delta_y = delta_y_gross / gamma   ← buyer pays fee on top

    Returns delta_y (Y paid by buyer, received by AMM).
    """
    gamma = 1.0 - state.fee
    k     = state.reserve_x * state.reserve_y
    x_new = jnp.maximum(state.reserve_x - delta_x, 1e-10)
    y_new = k / x_new
    delta_y_gross = y_new - state.reserve_y
    return delta_y_gross / gamma


def cp_curve_sell(state: CPState, delta_x: jnp.float32) -> jnp.float32:
    """
    READ-ONLY. How much Y does the seller receive for selling delta_x of X?

    Derivation (fee-on-input, X side):
        gamma   = 1 - fee
        x_new   = x + delta_x * gamma   ← only gamma fraction enters pool
        y_new   = k / x_new
        delta_y = y - y_new

    Returns delta_y (Y received by seller, paid by AMM).
    """
    gamma = 1.0 - state.fee
    k     = state.reserve_x * state.reserve_y
    x_new = state.reserve_x + delta_x * gamma
    y_new = k / x_new
    return jnp.maximum(state.reserve_y - y_new, 0.0)


def cp_swap(state: CPState, side: jnp.int32,
            delta_x: jnp.float32) -> tuple[CPState, jnp.float32]:
    """
    Execute a trade; return (new_state, delta_y).

    side = 0 → buy  (trader receives X, AMM receives Y)
    side = 1 → sell (trader pays X, AMM pays Y)

    Reserve update:
        Buy:   reserve_x -= delta_x
               reserve_y += delta_y * gamma   (AMM keeps gamma fraction of Y)
        Sell:  reserve_x += delta_x * gamma   (only gamma fraction enters)
               reserve_y -= delta_y

    Uses jnp.where throughout — no Python branching on traced values.
    """
    gamma = 1.0 - state.fee
    is_buy = (side == 0)

    dy_buy  = cp_curve_buy(state, delta_x)
    dy_sell = cp_curve_sell(state, delta_x)
    delta_y = jnp.where(is_buy, dy_buy, dy_sell)

    # Reserve updates for buy path
    new_rx_buy = state.reserve_x - delta_x
    new_ry_buy = state.reserve_y + delta_y * gamma

    # Reserve updates for sell path
    new_rx_sell = state.reserve_x + delta_x * gamma
    new_ry_sell = state.reserve_y - delta_y

    new_rx = jnp.where(is_buy, new_rx_buy, new_rx_sell)
    new_ry = jnp.where(is_buy, new_ry_buy, new_ry_sell)

    new_state = state.replace(reserve_x=new_rx, reserve_y=new_ry)
    return new_state, delta_y


# ══════════════════════════════════════════════════════════════════
# AMMSpec INSTANCE
# ══════════════════════════════════════════════════════════════════

CONSTANT_PRODUCT_AMM = AMMSpec(
    init=cp_init,
    curve_buy=cp_curve_buy,
    curve_sell=cp_curve_sell,
    swap=cp_swap,
)


# ══════════════════════════════════════════════════════════════════
# ARB SOLVER  (closed-form, paper §Asymmetric Arbitrage Dynamics)
# ══════════════════════════════════════════════════════════════════

def cp_arb_solver(spec: AMMSpec, state: CPState,
                  fair_price: jnp.float32) -> tuple[jnp.int32, jnp.float32]:
    """
    Closed-form arbitrage solver for the constant-product AMM.

    Conditions:
        spot = reserve_y / reserve_x
        gamma = 1 - fee

        spot < fair_price → arb buy  (buy X from AMM, spot too low)
            x' = sqrt(k / (gamma * fair_price))
            delta_x = x - x'

        spot > fair_price → arb sell (sell X to AMM, spot too high)
            x' = sqrt(gamma * k / fair_price)
            delta_x = (x' - x) / gamma

    Returns
    -------
    side    : 0 = buy, 1 = sell
    delta_x : quantity to trade (0 if no arb opportunity)

    All operations are jnp — fully jit/vmap/grad compatible.
    """
    x     = state.reserve_x
    y     = state.reserve_y
    k     = x * y
    gamma = 1.0 - state.fee
    spot  = y / x

    # Arb buy: spot < fair_price
    x_star_buy  = jnp.sqrt(k / (gamma * fair_price))
    dx_buy      = jnp.maximum(x - x_star_buy, 0.0)

    # Arb sell: spot > fair_price
    x_star_sell = jnp.sqrt(gamma * k / fair_price)
    dx_sell     = jnp.maximum((x_star_sell - x) / gamma, 0.0)

    is_buy  = spot < fair_price
    is_sell = spot > fair_price
    active  = is_buy | is_sell

    side    = jnp.where(is_buy, jnp.int32(0), jnp.int32(1))
    delta_x = jnp.where(is_buy, dx_buy, dx_sell)
    delta_x = jnp.where(active, delta_x, jnp.float32(0.0))

    return side, delta_x


# ══════════════════════════════════════════════════════════════════
# JAX VERIFICATION
# ══════════════════════════════════════════════════════════════════

def verify_jax_compatibility():
    """
    Run basic checks that the CP AMM is jit / vmap / grad compatible.
    Call this once after import to catch issues early.
    """
    import jax

    params = CPParams()
    state  = cp_init(params)

    # ── jit ────────────────────────────────────────────────────────
    jit_swap = jax.jit(cp_swap)
    s2, dy = jit_swap(state, jnp.int32(0), jnp.float32(1.0))
    assert s2.reserve_x < state.reserve_x, "jit swap buy: reserve_x should decrease"

    # ── vmap over batch of states ───────────────────────────────────
    batch = jax.tree.map(lambda x: jnp.stack([x] * 4), state)
    vmap_buy = jax.vmap(cp_curve_buy, in_axes=(0, None))
    dys = vmap_buy(batch, jnp.float32(1.0))
    assert dys.shape == (4,), f"vmap curve_buy: expected shape (4,), got {dys.shape}"

    # ── grad of curve_buy w.r.t. delta_x ───────────────────────────
    grad_fn  = jax.grad(cp_curve_buy, argnums=1)
    slope    = grad_fn(state, jnp.float32(1.0))
    assert slope > 0, "grad curve_buy: slope should be positive"

    # ── arb solver jit ─────────────────────────────────────────────
    jit_arb = jax.jit(cp_arb_solver, static_argnums=(0,))
    side, dx = jit_arb(CONSTANT_PRODUCT_AMM, state, jnp.float32(101.0))
    assert side == 0, "arb solver: fair > spot should trigger buy"
    assert dx > 0,    "arb solver: delta_x should be positive"

    print("CP AMM — all JAX compatibility checks passed.")


if __name__ == "__main__":
    verify_jax_compatibility()
