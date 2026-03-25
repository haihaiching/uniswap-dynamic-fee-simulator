"""
amm_sim/scoring.py
==================
Edge calculation — the single source of truth for how performance
is measured throughout the simulator.

Definition (from spec):
    Edge (AMM sells X) = delta_x * fair_price - delta_y
    Edge (AMM buys  X) = delta_y - delta_x * fair_price

Positive edge = captured from uninformed retail flow (spread).
Negative edge = lost to informed arbitrage flow (staleness).
"""

import jax.numpy as jnp


def compute_edge(side: jnp.ndarray,
                 delta_x: jnp.ndarray,
                 delta_y: jnp.ndarray,
                 fair_price: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the AMM's edge on a single trade.

    Parameters
    ----------
    side        : 0 = buy (AMM sells X, receives Y)
                  1 = sell (AMM buys X, pays Y)
    delta_x     : units of X exchanged (always positive)
    delta_y     : units of Y exchanged (always positive)
    fair_price  : current oracle fair price

    Returns
    -------
    edge : scalar float32
        Positive → AMM gained value relative to fair price.
        Negative → AMM lost value (arb extraction).

    Formula
    -------
    Trader sells X (side=0, AMM buys y):  edge = delta_y - delta_x * p
        The AMM gave away delta_x units of X worth delta_x*p,
        and received delta_y Y. If delta_y < delta_x*p, AMM lost.

    Trader buys X (side=1, AMM sells y):  edge = delta_x * p - delta_y
        The AMM paid delta_y Y and received delta_x X worth delta_x*p.
        If delta_y > delta_x*p, AMM overpaid.

    Uses jnp.where so the function is jit/vmap/grad compatible.
    """
    edge_sell = delta_y - delta_x * fair_price   # trader sells X (AMM buys Y)
    edge_buy  = delta_x * fair_price - delta_y   # trader buys  X (AMM sells X)
    return jnp.where(side == 0, edge_sell, edge_buy)


def compute_edge_batch(sides: jnp.ndarray,
                       delta_xs: jnp.ndarray,
                       delta_ys: jnp.ndarray,
                       fair_price: jnp.ndarray) -> jnp.ndarray:
    """
    Vectorised version: compute edge for an array of trades.

    Parameters
    ----------
    sides       : (N,) int array, each element 0 or 1
    delta_xs    : (N,) float array
    delta_ys    : (N,) float array
    fair_price  : scalar

    Returns
    -------
    edges : (N,) float array
    """
    edge_sell = delta_ys - delta_xs * fair_price
    edge_buy  = delta_xs * fair_price - delta_ys
    return jnp.where(sides == 0, edge_sell, edge_buy)
