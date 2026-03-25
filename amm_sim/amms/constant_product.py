"""
amm_sim/amms/constant_product.py
=================================
Constant-product AMM (x·y = k) with ASYMMETRIC fee-on-input.

Key change from symmetric version:
    gamma_plus  = 1 - fee_plus   (buy-side,  trader buys  X from AMM)
    gamma_minus = 1 - fee_minus  (sell-side, trader sells X to   AMM)

These two can be set independently, enabling directional pricing.

No-arbitrage zone (paper §Asymmetric Arbitrage Dynamics):
    z = ln(p / spot)
    No arb when:  ln(gamma_minus) <= z <= -ln(gamma_plus)

Arb solvers (closed-form):
    Arb buy:   x' = sqrt(k / (gamma_plus  * p))
    Arb sell:  x' = sqrt(gamma_minus * k / p)

Routing (paper §Aggregation):
    depth = sqrt(k) = sqrt(x * y)   (Uniswap V2 standard liquidity measure)
    Each pool receives flow proportional to its depth / total depth.
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
    Static parameters for an asymmetric-fee Constant-Product AMM.

    Attributes
    ----------
    fee_plus  : buy-side  fee rate  (trader buys  X, AMM sells X)
    fee_minus : sell-side fee rate  (trader sells X, AMM buys  X)
    init_x    : initial token X reserve
    init_y    : initial token Y reserve
    """
    fee_plus:  float = 0.003   # 30 bps buy-side  default
    fee_minus: float = 0.003   # 30 bps sell-side default
    init_x:    float = 100.0
    init_y:    float = 10_000.0


@chex.dataclass(frozen=True)
class CPState:
    """
    Mutable state of an asymmetric-fee Constant-Product AMM.

    Attributes
    ----------
    reserve_x   : token X reserve
    reserve_y   : token Y reserve
    gamma_plus  : buy-side  fee multiplier = 1 - fee_plus
    gamma_minus : sell-side fee multiplier = 1 - fee_minus

    Derived (not stored):
        k     = reserve_x * reserve_y
        spot  = reserve_y / reserve_x
        depth = sqrt(k)   ← used for routing
        z     = ln(fair_price / spot)   ← log-mispricing
    """
    reserve_x:   jnp.float32
    reserve_y:   jnp.float32
    gamma_plus:  jnp.float32   # buy-side  fee multiplier
    gamma_minus: jnp.float32   # sell-side fee multiplier


# ── Convenience helpers ───────────────────────────────────────────

def cp_depth(state: CPState) -> jnp.float32:
    """
    Uniswap V2 standard liquidity measure: depth = sqrt(x * y).
    Used by engine for proportional routing across pools.
    """
    return jnp.sqrt(state.reserve_x * state.reserve_y)


# ══════════════════════════════════════════════════════════════════
# FOUR CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def cp_init(params: CPParams) -> CPState:
    """Build initial CPState from CPParams."""
    return CPState(
        reserve_x=jnp.float32(params.init_x),
        reserve_y=jnp.float32(params.init_y),
        gamma_plus=jnp.float32(1.0 - params.fee_plus),
        gamma_minus=jnp.float32(1.0 - params.fee_minus),
    )


def cp_curve_buy(state: CPState, delta_x: jnp.float32) -> jnp.float32:
    """
    READ-ONLY. How much Y must the buyer pay to receive delta_x of X?

    Uses gamma_plus (buy-side fee multiplier).

    Derivation (fee-on-output, Y side):
        k             = x * y
        x_new         = x - delta_x
        y_new         = k / x_new
        delta_y_gross = y_new - y
        delta_y       = delta_y_gross / gamma_plus   ← buyer pays fee on top

    Returns delta_y (Y paid by buyer, received by AMM).
    """
    gp    = state.gamma_plus
    k     = state.reserve_x * state.reserve_y
    x_new = jnp.maximum(state.reserve_x - delta_x, 1e-10)
    y_new = k / x_new
    return (y_new - state.reserve_y) / gp


def cp_curve_sell(state: CPState, delta_x: jnp.float32) -> jnp.float32:
    """
    READ-ONLY. How much Y does the seller receive for selling delta_x of X?

    Uses gamma_minus (sell-side fee multiplier).

    Derivation (fee-on-input, X side):
        x_new   = x + delta_x * gamma_minus   ← only gamma fraction enters
        y_new   = k / x_new
        delta_y = y - y_new

    Returns delta_y (Y received by seller, paid by AMM).
    """
    gm    = state.gamma_minus
    k     = state.reserve_x * state.reserve_y
    x_new = state.reserve_x + delta_x * gm
    y_new = k / x_new
    return jnp.maximum(state.reserve_y - y_new, 0.0)


def cp_swap(state: CPState, side: jnp.int32,
            delta_x: jnp.float32) -> tuple[CPState, jnp.float32]:
    """
    Execute a trade; return (new_state, delta_y).

    side = 0 → buy  (trader receives X, AMM receives Y)  uses gamma_plus
    side = 1 → sell (trader pays X,    AMM pays Y)        uses gamma_minus

    Reserve updates:
        Buy:   reserve_x -= delta_x
               reserve_y += delta_y * gamma_plus    (fee retained in pool)
        Sell:  reserve_x += delta_x * gamma_minus   (fee retained in pool)
               reserve_y -= delta_y
    """
    gp    = state.gamma_plus
    gm    = state.gamma_minus
    is_buy = (side == 0)

    dy_buy  = cp_curve_buy(state, delta_x)
    dy_sell = cp_curve_sell(state, delta_x)
    delta_y = jnp.where(is_buy, dy_buy, dy_sell)

    # Buy path
    new_rx_buy = state.reserve_x - delta_x
    new_ry_buy = state.reserve_y + delta_y * gp

    # Sell path
    new_rx_sell = state.reserve_x + delta_x * gm
    new_ry_sell = state.reserve_y - delta_y

    new_rx = jnp.where(is_buy, new_rx_buy,  new_rx_sell)
    new_ry = jnp.where(is_buy, new_ry_buy,  new_ry_sell)

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
# ARB SOLVER  (closed-form, asymmetric fees)
# ══════════════════════════════════════════════════════════════════

def cp_arb_solver(spec: AMMSpec, state: CPState,
                  fair_price: jnp.float32,
                  epsilon: jnp.float32 = jnp.float32(0.0)) -> tuple[jnp.int32, jnp.float32]:
    # epsilon accepted for interface compatibility with Linear AMM arb solver,
    # but not used — CP AMM arb is fully determined by spot vs fair_price.
    """
    Closed-form arbitrage solver for the asymmetric-fee CP AMM.

    No-arbitrage zone:  ln(gamma_minus) <= z <= -ln(gamma_plus)
    where z = ln(fair_price / spot).

    Arb Buy  (z > -ln gamma_plus,  spot < fair):
        x' = sqrt(k / (gamma_plus * fair_price))
        delta_x = x - x'

    Arb Sell (z < ln gamma_minus,  spot > fair):
        x' = sqrt(gamma_minus * k / fair_price)
        delta_x = (x' - x) / gamma_minus

    Returns (side, delta_x).  delta_x = 0 if no arb opportunity.
    """
    x    = state.reserve_x
    y    = state.reserve_y
    k    = x * y
    gp   = state.gamma_plus
    gm   = state.gamma_minus
    spot = y / x
    z    = jnp.log(fair_price / spot)

    hi = -jnp.log(gp)   # upper boundary of no-arb zone
    lo =  jnp.log(gm)   # lower boundary of no-arb zone

    # Arb buy: z > hi  (spot too low)
    x_star_buy = jnp.sqrt(k / (gp * fair_price))
    dx_buy     = jnp.maximum(x - x_star_buy, 0.0)

    # Arb sell: z < lo  (spot too high)
    x_star_sell = jnp.sqrt(gm * k / fair_price)
    dx_sell     = jnp.maximum((x_star_sell - x) / gm, 0.0)

    is_buy  = z > hi
    is_sell = z < lo
    active  = is_buy | is_sell

    side    = jnp.where(is_buy, jnp.int32(0), jnp.int32(1))
    delta_x = jnp.where(is_buy, dx_buy, dx_sell)
    delta_x = jnp.where(active, delta_x, jnp.float32(0.0))

    # Filter floating point noise
    delta_x = jnp.where(delta_x > 1e-6, delta_x, jnp.float32(0.0))

    return side, delta_x


# ══════════════════════════════════════════════════════════════════
# JAX VERIFICATION
# ══════════════════════════════════════════════════════════════════

def verify_jax_compatibility():
    """Run basic jit / vmap / grad checks on the asymmetric-fee CP AMM."""
    params = CPParams(fee_plus=0.002, fee_minus=0.004)
    state  = cp_init(params)

    # jit
    jit_swap = jax.jit(cp_swap)
    s2, dy = jit_swap(state, jnp.int32(0), jnp.float32(1.0))
    assert s2.reserve_x < state.reserve_x

    # vmap
    batch    = jax.tree.map(lambda x: jnp.stack([x] * 4), state)
    vmap_buy = jax.vmap(cp_curve_buy, in_axes=(0, None))
    dys      = vmap_buy(batch, jnp.float32(1.0))
    assert dys.shape == (4,)

    # grad
    grad_fn = jax.grad(cp_curve_buy, argnums=1)
    slope   = grad_fn(state, jnp.float32(1.0))
    assert slope > 0

    # arb solver — asymmetric zone
    jit_arb  = jax.jit(cp_arb_solver, static_argnums=(0,))
    side, dx = jit_arb(CONSTANT_PRODUCT_AMM, state, jnp.float32(101.0))
    assert side == 0
    assert dx   > 0

    # depth
    d = cp_depth(state)
    assert d > 0

    print("CP AMM (asymmetric fees) — all JAX compatibility checks passed.")


if __name__ == "__main__":
    verify_jax_compatibility()


# ══════════════════════════════════════════════════════════════════
# EDGE FUNCTION  (interface-compatible with linear_edge)
# ══════════════════════════════════════════════════════════════════

def cp_edge(state: CPState, side: jnp.int32,
            delta_x: jnp.float32) -> jnp.float32:
    """
    Compute LP edge for a CP AMM trade using scoring.compute_edge.

    This wraps compute_edge in the same interface as linear_edge:
        (state, side, delta_x) → edge

    delta_y is computed from the bonding curve before the swap,
    using the current state (pre-trade).
    fair_price is approximated as the current spot price —
    the engine passes the actual fair_price via the CycleRecord,
    but for per-trade edge we use spot as proxy here.

    Note: for rigorous edge calculation, fair_price should be
    passed explicitly. This is a known approximation.
    """
    from amm_sim.scoring import compute_edge
    is_buy  = (side == 0)
    dy_buy  = cp_curve_buy(state, delta_x)
    dy_sell = cp_curve_sell(state, delta_x)
    delta_y = jnp.where(is_buy, dy_buy, dy_sell)
    fair_price = state.reserve_y / state.reserve_x   # spot as proxy
    return compute_edge(side, delta_x, delta_y, fair_price)