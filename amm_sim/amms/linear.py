"""
amm_sim/amms/linear.py
=======================
Linear-Impact AMM following:
    "Optimal Liquidity Provision with Linear Price Impact"

State is described by explicit pool bid/ask prices (P+, P-) and net
inventories (reserve_x, reserve_y). Prices are updated only when trades
occur — the oracle no longer touches AMM state directly.

Key formulas
------------

Trades (§Trades):
    Buy  Δ+: P̄+ = P+ + Δ+/(2λ++),   ΔY = P̄+ · Δ+
    Sell Δ-: P̄- = P- - Δ-/(2λ--),   ΔY = P̄- · Δ-

Price update on trade (§Price update):
    Buy  Δ+: P+_new = P+ + Δ+/λ++,   P-_new = P- + Δ+/λ-+
    Sell Δ-: P-_new = P- - Δ-/λ--,   P+_new = P+ - Δ-/λ+-

Inventory update:
    Buy:  reserve_x -= Δ+,  reserve_y += ΔY
    Sell: reserve_x += Δ-,  reserve_y -= ΔY

Per-trade cap
-------------
Both sides clip delta_x to state.max_trade_x — a hard limit encoded in the AMM
state that bounds price impact regardless of inventory level. arb.py and router.py
use max_trade_buy / max_trade_sell (both returning max_trade_x) as bisection bounds.
"""

import jax
import jax.numpy as jnp
import chex
from amm_sim.spec import AMMSpec


# ══════════════════════════════════════════════════════════════════
# PARAMS & STATE
# ══════════════════════════════════════════════════════════════════

@chex.dataclass(frozen=True)
class LinearParams:
    """
    Static initialisation parameters.

    lam_pp      : λ++  ask-side depth   (buy  impact on ask)
    lam_mm      : λ--  bid-side depth   (sell impact on bid)
    lam_pm      : λ+-  sell cross-impact on ask
    lam_mp      : λ-+  buy  cross-impact on bid
    init_p_ask  : initial pool ask price P+
    init_p_bid  : initial pool bid price P-
    init_x      : initial token X inventory
    init_y      : initial token Y inventory
    max_trade_x : hard per-trade cap on delta_x (both buy and sell)
    """
    lam_pp:      float = 100
    lam_mm:      float = 100
    lam_pm:      float = 200
    lam_mp:      float = 200
    init_p_ask:  float = 100.2
    init_p_bid:  float = 99.8
    init_x:      float = 100.0
    init_y:      float = 10_000.0
    max_trade_x: float = 10.0


@chex.dataclass(frozen=True)
class LinearState:
    """
    Mutable state of a Linear-Impact AMM.

    p_ask       : current pool ask price P+
    p_bid       : current pool bid price P-
    lam_pp      : λ++
    lam_mm      : λ--
    lam_pm      : λ+-
    lam_mp      : λ-+
    reserve_x   : net X inventory
    reserve_y   : net Y inventory
    max_trade_x : hard per-trade cap on delta_x
    """
    p_ask:       jnp.float32
    p_bid:       jnp.float32
    lam_pp:      jnp.float32
    lam_mm:      jnp.float32
    lam_pm:      jnp.float32
    lam_mp:      jnp.float32
    reserve_x:   jnp.float32
    reserve_y:   jnp.float32
    max_trade_x: jnp.float32


# ══════════════════════════════════════════════════════════════════
# FOUR CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def linear_init(params: LinearParams) -> LinearState:
    """Build initial LinearState from LinearParams."""
    return LinearState(
        p_ask=jnp.float32(params.init_p_ask),
        p_bid=jnp.float32(params.init_p_bid),
        lam_pp=jnp.float32(params.lam_pp),
        lam_mm=jnp.float32(params.lam_mm),
        lam_pm=jnp.float32(params.lam_pm),
        lam_mp=jnp.float32(params.lam_mp),
        reserve_x=jnp.float32(params.init_x),
        reserve_y=jnp.float32(params.init_y),
        max_trade_x=jnp.float32(params.max_trade_x),
    )


def linear_curve_buy(state: LinearState,
                     delta_x: jnp.float32) -> jnp.float32:
    """
    READ-ONLY. Y paid by buyer to receive delta_x of X.

    Paper §Trades:
        P̄+ = P+ + Δ+/(2λ++)
        ΔY  = P̄+ · Δ+  =  (P+ + Δ+/(2λ++)) · Δ+

    delta_x is clipped to state.max_trade_x (hard per-trade cap).
    Returns delta_y (Y paid by buyer, received by AMM).
    """
    safe_dx = jnp.minimum(delta_x, state.max_trade_x)
    return (state.p_ask + safe_dx / (2.0 * state.lam_pp)) * safe_dx


def linear_curve_sell(state: LinearState,
                      delta_x: jnp.float32) -> jnp.float32:
    """
    READ-ONLY. Y received by seller for selling delta_x of X.

    Paper §Trades:
        P̄- = P- - Δ-/(2λ--)
        ΔY  = P̄- · Δ-  =  (P- - Δ-/(2λ--)) · Δ-

    delta_x is clipped to state.max_trade_x (hard per-trade cap).
    Returns delta_y (Y received by seller, paid by AMM).
    """
    safe_dx = jnp.minimum(delta_x, state.max_trade_x)
    return jnp.maximum(
        (state.p_bid - safe_dx / (2.0 * state.lam_mm)) * safe_dx,
        0.0
    )


def linear_swap(state: LinearState, side: jnp.int32,
                delta_x: jnp.float32) -> tuple[LinearState, jnp.float32]:
    """
    Execute a trade; return (new_state, delta_y).

    side = 0 → buy  (trader receives X, LP receives Y)
    side = 1 → sell (trader pays X,    LP pays Y)

    Price update (paper §Price update):
        Buy  Δ+: P+_new = P+ + Δ+/λ++,   P-_new = P- + Δ+/λ-+
        Sell Δ-: P-_new = P- - Δ-/λ--,   P+_new = P+ - Δ-/λ+-

    delta_x is clipped to state.max_trade_x before all updates.
    """
    is_buy = (side == 0)

    eff_dx = jnp.minimum(delta_x, state.max_trade_x)

    dy_buy  = linear_curve_buy(state, eff_dx)
    dy_sell = linear_curve_sell(state, eff_dx)
    delta_y = jnp.where(is_buy, dy_buy, dy_sell)

    # ── Price updates ──────────────────────────────────────────────
    new_p_ask = jnp.where(is_buy,
                          state.p_ask + eff_dx / state.lam_pp,
                          state.p_ask - eff_dx / state.lam_pm)

    new_p_bid = jnp.where(is_buy,
                          state.p_bid + eff_dx / state.lam_mp,
                          state.p_bid - eff_dx / state.lam_mm)

    # ── Inventory updates ──────────────────────────────────────────
    new_rx = jnp.where(is_buy, state.reserve_x - eff_dx, state.reserve_x + eff_dx)
    new_ry = jnp.where(is_buy, state.reserve_y + delta_y, state.reserve_y - delta_y)

    return state.replace(
        p_ask=new_p_ask,
        p_bid=new_p_bid,
        reserve_x=new_rx,
        reserve_y=new_ry,
    ), delta_y


# ══════════════════════════════════════════════════════════════════
# AMMSpec INSTANCE
# ══════════════════════════════════════════════════════════════════

def linear_max_trade_buy(state: LinearState):
    """Hard per-trade cap — same for both sides."""
    return state.max_trade_x


def linear_max_trade_sell(state: LinearState):
    """Hard per-trade cap — same for both sides."""
    return state.max_trade_x


LINEAR_AMM = AMMSpec(
    init=linear_init,
    curve_buy=linear_curve_buy,
    curve_sell=linear_curve_sell,
    swap=linear_swap,
    max_trade_buy=linear_max_trade_buy,
    max_trade_sell=linear_max_trade_sell,
)


# ══════════════════════════════════════════════════════════════════
# JAX VERIFICATION
# ══════════════════════════════════════════════════════════════════

def verify_jax_compatibility():
    """Run basic jit / vmap / grad checks on the Linear-Impact AMM."""
    params = LinearParams()
    state  = linear_init(params)

    # jit swap
    jit_swap = jax.jit(linear_swap)
    s2, dy   = jit_swap(state, jnp.int32(0), jnp.float32(1.0))
    assert s2.reserve_x < state.reserve_x,  "buy should reduce X inventory"
    assert s2.p_ask     > state.p_ask,       "buy should push ask price up"
    assert s2.p_bid     > state.p_bid,       "buy should push bid up via cross-impact"

    # vmap
    batch    = jax.tree.map(lambda x: jnp.stack([x] * 4), state)
    vmap_buy = jax.vmap(linear_curve_buy, in_axes=(0, None))
    dys      = vmap_buy(batch, jnp.float32(1.0))
    assert dys.shape == (4,)

    # grad — marginal buy cost should be positive
    grad_fn = jax.grad(linear_curve_buy, argnums=1)
    slope   = grad_fn(state, jnp.float32(1.0))
    assert slope > 0, "marginal buy cost should increase with size"

    print("Linear AMM — all JAX compatibility checks passed.")


if __name__ == "__main__":
    verify_jax_compatibility()