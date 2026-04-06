"""
amm_sim/arb.py
==============
Generic arbitrage solver for any AMM via AMMSpec.

Buy arb:  marginal_ask(eps) < fair_price → buy until marginal_ask = fair_price
Sell arb: marginal_bid(eps) > fair_price → sell until marginal_bid = fair_price

Bisection bounds: [0, reserve_x * (1 - 1e-6)] for both sides.
Early-stop when interval width < tol. Uses lax.while_loop (scan-compatible).
"""

import jax
import jax.numpy as jnp

from amm_sim.spec import AMMSpec, marginal_ask, marginal_bid


def generic_arb_solver(spec: AMMSpec,
                       state,
                       fair_price: jnp.ndarray,
                       num_iters: int = 100,
                       tol: float = 1e-6) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Parameters
    ----------
    spec       : AMMSpec
    state      : AMM state (must expose reserve_x)
    fair_price : current oracle fair price
    num_iters  : max bisection iterations
    tol        : early-stop on interval width

    Returns
    -------
    (side, delta_x)  — side 0 = buy, 1 = sell; delta_x = 0 if no arb
    """
    _eps = jnp.float32(1e-6)
    buy_arb  = marginal_ask(spec, state, _eps) < fair_price
    sell_arb = marginal_bid(spec, state, _eps) > fair_price

    tol_f    = jnp.float32(tol)
    _safety  = jnp.float32(1.0 - 1e-6)
    hi_buy   = spec.max_trade_buy(state)  * _safety
    hi_sell  = spec.max_trade_sell(state) * _safety

    # ── Buy arb ────────────────────────────────────────────────────────────
    def buy_cond(carry):
        lo, hi, i = carry
        return (hi - lo > tol_f) & (i < num_iters)

    def buy_body(carry):
        lo, hi, i = carry
        mid   = (lo + hi) * jnp.float32(0.5)
        m_mid = marginal_ask(spec, state, mid)
        lo = jnp.where(m_mid < fair_price, mid, lo)
        hi = jnp.where(m_mid < fair_price, hi,  mid)
        return lo, hi, i + 1

    lo_buy, hi_buy, _ = jax.lax.while_loop(
        buy_cond, buy_body, (jnp.float32(0.0), hi_buy, jnp.int32(0))
    )
    dx_buy = (lo_buy + hi_buy) * jnp.float32(0.5)

    # ── Sell arb ───────────────────────────────────────────────────────────
    def sell_cond(carry):
        lo, hi, i = carry
        return (hi - lo > tol_f) & (i < num_iters)

    def sell_body(carry):
        lo, hi, i = carry
        mid   = (lo + hi) * jnp.float32(0.5)
        m_mid = marginal_bid(spec, state, mid)
        lo = jnp.where(m_mid > fair_price, mid, lo)
        hi = jnp.where(m_mid > fair_price, hi,  mid)
        return lo, hi, i + 1

    lo_sell, hi_sell, _ = jax.lax.while_loop(
        sell_cond, sell_body, (jnp.float32(0.0), hi_sell, jnp.int32(0))
    )
    dx_sell = (lo_sell + hi_sell) * jnp.float32(0.5)

    # ── Select ─────────────────────────────────────────────────────────────
    active  = buy_arb | sell_arb
    side    = jnp.where(buy_arb, jnp.int32(0), jnp.int32(1))
    delta_x = jnp.where(buy_arb, dx_buy, dx_sell)
    delta_x = jnp.where(active, delta_x, jnp.float32(0.0))

    return side, delta_x