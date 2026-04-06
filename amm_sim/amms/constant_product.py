"""
amm_sim/amms/constant_product.py
=================================
Constant-product AMM (x·y = k) with asymmetric fees.

    gamma_plus  = 1 - fee_plus   (buy-side:  trader buys  X from AMM)
    gamma_minus = 1 - fee_minus  (sell-side: trader sells X to   AMM)

Fees are encoded in the bonding curve formulas directly; cp_swap applies
no additional fee multipliers to the reserve updates.
"""

import jax
import jax.numpy as jnp
import chex
from amm_sim.spec import AMMSpec


@chex.dataclass(frozen=True)
class CPParams:
    fee_plus:    float = 0.003
    fee_minus:   float = 0.003
    init_x:      float = 100.0
    init_y:      float = 10_000.0
    max_trade_x: float = 2.0


@chex.dataclass(frozen=True)
class CPState:
    reserve_x:   jnp.float32
    reserve_y:   jnp.float32
    gamma_plus:  jnp.float32
    gamma_minus: jnp.float32
    max_trade_x: jnp.float32


def cp_init(params: CPParams) -> CPState:
    return CPState(
        reserve_x=jnp.float32(params.init_x),
        reserve_y=jnp.float32(params.init_y),
        gamma_plus=jnp.float32(1.0 - params.fee_plus),
        gamma_minus=jnp.float32(1.0 - params.fee_minus),
        max_trade_x=jnp.float32(params.max_trade_x),
    )


def cp_curve_buy(state: CPState, delta_x: jnp.float32) -> jnp.float32:
    """Y paid by buyer to receive delta_x of X.
    delta_y = y · Δx / ((x − Δx) · γ+)
    delta_x is clipped to state.max_trade_x (hard per-trade cap).
    """
    safe_dx = jnp.minimum(delta_x, state.max_trade_x)
    return state.reserve_y * safe_dx / ((state.reserve_x - safe_dx) * state.gamma_plus)


def cp_curve_sell(state: CPState, delta_x: jnp.float32) -> jnp.float32:
    """Y received by seller for selling delta_x of X.
    delta_y = y · Δx · γ− / (x + Δx · γ−)
    delta_x is clipped to state.max_trade_x (hard per-trade cap).
    """
    safe_dx = jnp.minimum(delta_x, state.max_trade_x)
    gm = state.gamma_minus
    return state.reserve_y * safe_dx * gm / (state.reserve_x + safe_dx * gm)


def cp_swap(state: CPState, side: jnp.int32,
            delta_x: jnp.float32) -> tuple[CPState, jnp.float32]:
    """
    Execute a trade; return (new_state, delta_y).

    side = 0 → buy  (trader receives X, AMM receives Y)
    side = 1 → sell (trader pays X,    AMM pays Y)

    delta_x is clipped to state.max_trade_x before all updates.
    Fees are already embedded in delta_y from the curve functions.
    """
    is_buy = (side == 0)

    eff_dx  = jnp.minimum(delta_x, state.max_trade_x)
    dy_buy  = cp_curve_buy(state, eff_dx)
    dy_sell = cp_curve_sell(state, eff_dx)
    delta_y = jnp.where(is_buy, dy_buy, dy_sell)

    new_rx = jnp.where(is_buy,
                       state.reserve_x - eff_dx,
                       state.reserve_x + eff_dx)
    new_ry = jnp.where(is_buy,
                       state.reserve_y + delta_y,
                       state.reserve_y - delta_y)

    return state.replace(reserve_x=new_rx, reserve_y=new_ry), delta_y


def cp_max_trade_buy(state: CPState):
    """Hard per-trade cap — same for both sides."""
    return state.max_trade_x


def cp_max_trade_sell(state: CPState):
    """Hard per-trade cap — same for both sides."""
    return state.max_trade_x


CONSTANT_PRODUCT_AMM = AMMSpec(
    init=cp_init,
    curve_buy=cp_curve_buy,
    curve_sell=cp_curve_sell,
    swap=cp_swap,
    max_trade_buy=cp_max_trade_buy,
    max_trade_sell=cp_max_trade_sell,
)


def verify_jax_compatibility():
    """Run jit / vmap / grad checks on the CP AMM."""
    params = CPParams(fee_plus=0.002, fee_minus=0.004)
    state  = cp_init(params)

    jit_swap = jax.jit(cp_swap)
    s2, _ = jit_swap(state, jnp.int32(0), jnp.float32(1.0))
    assert s2.reserve_x < state.reserve_x

    batch    = jax.tree.map(lambda x: jnp.stack([x] * 4), state)
    vmap_buy = jax.vmap(cp_curve_buy, in_axes=(0, None))
    dys      = vmap_buy(batch, jnp.float32(1.0))
    assert dys.shape == (4,)

    grad_fn = jax.grad(cp_curve_buy, argnums=1)
    slope   = grad_fn(state, jnp.float32(1.0))
    assert slope > 0

    print("CP AMM (asymmetric fees) — all JAX compatibility checks passed.")


if __name__ == "__main__":
    verify_jax_compatibility()