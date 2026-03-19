"""
amm_sim/spec.py
===============
Core AMM interface definition.

An AMM is fully described by four callables bundled in AMMSpec.
The engine never looks inside AMM state — it only calls these four functions.
"""

from typing import Callable, NamedTuple


class AMMSpec(NamedTuple):
    """
    Defines the interface of a single AMM.

    Functions
    ---------
    init(params) -> state
        Build the initial AMM state from static parameters.

    curve_buy(state, delta_x) -> delta_y
        READ-ONLY. Given current state, how much Y must a buyer pay
        to receive delta_x units of X?

    curve_sell(state, delta_x) -> delta_y
        READ-ONLY. Given current state, how much Y does a seller
        receive for selling delta_x units of X?

    swap(state, side, delta_x) -> (state', delta_y)
        Execute a trade and return the updated state plus the Y amount
        transferred.
            side = 0  →  buy  (trader receives X, pays Y)
            side = 1  →  sell (trader pays X, receives Y)

    Notes
    -----
    - All functions must use jnp operations only (no Python if/else on
      traced values) to remain jit / vmap / grad compatible.
    - state must be a frozen chex.dataclass (immutable pytree).
    """
    init:       Callable   # (params) → state
    curve_buy:  Callable   # (state, delta_x) → delta_y
    curve_sell: Callable   # (state, delta_x) → delta_y
    swap:       Callable   # (state, side, delta_x) → (state', delta_y)


# ── Marginal price utilities (engine-level, not part of AMMSpec) ──────────

def marginal_ask(spec: AMMSpec, state, eps: float = 1e-6):
    """
    Marginal ask price: cost per unit of X for an infinitesimal buy.
    Works for any bonding curve shape.
    """
    return spec.curve_buy(state, eps) / eps


def marginal_bid(spec: AMMSpec, state, eps: float = 1e-6):
    """
    Marginal bid price: revenue per unit of X for an infinitesimal sell.
    Works for any bonding curve shape.
    """
    return spec.curve_sell(state, eps) / eps
