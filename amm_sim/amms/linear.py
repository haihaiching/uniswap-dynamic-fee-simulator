"""
amm_sim/amms/linear.py
=======================
Linear-Impact AMM following:
    "Optimal Liquidity Provision with Linear Price Impact"

State is described by spread state (Z+, Z-) and inventory (X, Y).

Key formulas (all from the paper)
----------------------------------

Spread state:
    Z+_k = P+_k - S_k    (ask spread)
    Z-_k = S_k - P-_k    (bid spread)
    Z > 0: takers pay spread (LP earns)
    Z < 0: arbitrage opportunity (LP loses)

Trades (§Trades):
    Buy  Δ+: P̄+ = P+ + Δ+/(2λ++),   ΔY_received = P̄+ · Δ+
    Sell Δ-: P̄- = P- - Δ-/(2λ--),   ΔY_paid     = P̄- · Δ-

    In spread terms (Z+ = P+ - S):
        P̄+ = S + Z+ + Δ+/(2λ++)

Price update (§Price update):
    Buy  Δ+: Z+_new = Z+ + Δ+/λ++,   Z-_new = Z- + Δ+/λ-+
    Sell Δ-: Z-_new = Z- - Δ-/λ--,   Z+_new = Z+ - Δ-/λ+-

Inventory (§Inventory):
    X_new = X + Δ- - Δ+
    Y_new = Y + P̄+·Δ+ - P̄-·Δ-

Arbitrage (§Step A):
    Oracle moves ε: Z̃+ = Z+ - ε,  Z̃- = Z- + ε
    Ask stale (Z̃+ < 0): Δ+_arb = -λ++ · Z̃+
    Bid stale (Z̃- < 0): Δ-_arb = -λ-- · Z̃-

    Post-arb spread state (paper formula):
        Case ask stale:  z_arb = (0,  Z̃- + (λ++/λ-+)·Z̃+)
        Case bid stale:  z_arb = (Z̃+ + (λ--/λ+-)·Z̃-, 0)
        Case no arb:     z_arb = (Z̃+, Z̃-)

Routing (§Aggregation, §Equal best asks):
    When two pools have equal best ask P+:
        Δ_i+ = (λ_i+ / Λ+) · Δ+,   Λ+ = Σ λ_i+
    General case (different P+):
        Buy sweeps cheapest pool first.
    For two pools: solved analytically by equalising marginal cost.

Edge (§The Edge):
    Buy:  Edge+ = Δ+(Z+ + Δ+/(2λ++))
    Sell: Edge- = Δ-(Z- + Δ-/(2λ--))
    Arb:  Edge_arb = -λ++(Z̃+)²/2  or  -λ--(Z̃-)²/2
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

    lam_pp   : λ++  ask-side depth  (buy  impact on ask)
    lam_mm   : λ--  bid-side depth  (sell impact on bid)
    lam_pm   : λ+-  sell cross-impact on ask
    lam_mp   : λ-+  buy  cross-impact on bid
    init_z_plus  : initial ask spread Z+ = P+ - S
    init_z_minus : initial bid spread Z- = S - P-
    """
    lam_pp:       float = 0.01
    lam_mm:       float = 0.01
    lam_pm:       float = 0.005
    lam_mp:       float = 0.005
    init_z_plus:  float = 0.3
    init_z_minus: float = 0.3
    init_x:       float = 100.0
    init_y:       float = 10_000.0


@chex.dataclass(frozen=True)
class LinearState:
    """
    Mutable state of a Linear-Impact AMM.

    z_plus  : Z+ = P+ - S  (ask spread, >0 takers pay, <0 arb opportunity)
    z_minus : Z- = S - P-  (bid spread, >0 takers pay, <0 arb opportunity)
    x       : risky asset inventory (LP holds)
    y       : numéraire cash balance (LP holds)
    lam_pp  : λ++  (in state so strategy can update dynamically)
    lam_mm  : λ--
    lam_pm  : λ+-
    lam_mp  : λ-+
    """
    z_plus:  jnp.float32
    z_minus: jnp.float32
    x:       jnp.float32
    y:       jnp.float32
    lam_pp:    jnp.float32
    lam_mm:  jnp.float32
    lam_pm:  jnp.float32
    lam_mp:    jnp.float32
    reserve_x: jnp.float32   # = init_x, for env.py fair price init
    reserve_y: jnp.float32   # = init_y, reserve_y/reserve_x = init fair price


# ══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def linear_init(params: LinearParams) -> LinearState:
    """Build initial LinearState from LinearParams."""
    return LinearState(
        z_plus=jnp.float32(params.init_z_plus),
        z_minus=jnp.float32(params.init_z_minus),
        x=jnp.float32(0.0),
        y=jnp.float32(0.0),
        lam_pp=jnp.float32(params.lam_pp),
        lam_mm=jnp.float32(params.lam_mm),
        lam_pm=jnp.float32(params.lam_pm),
        lam_mp=jnp.float32(params.lam_mp),
        reserve_x=jnp.float32(params.init_x),
        reserve_y=jnp.float32(params.init_y),
    )


def linear_curve_buy(state: LinearState,
                     delta_x: jnp.float32) -> jnp.float32:
    """
    READ-ONLY. Y paid by buyer to receive delta_x of X.

    Paper §Trades:
        P̄+ = P+ + Δ+/(2λ++)  =  (S + Z+) + Δ+/(2λ++)
        ΔY  = P̄+ · Δ+

    We express everything in spread-state terms.
    S cancels out in edge calculations, so we return
    the Y flow relative to fair price:
        ΔY = (Z+ + Δ+/(2λ++)) · Δ+

    This is consistent with the edge formula:
        Edge+ = Δ+(Z+ + Δ+/(2λ++))
    """
    return (state.z_plus + delta_x / (2.0 * state.lam_pp)) * delta_x


def linear_curve_sell(state: LinearState,
                      delta_x: jnp.float32) -> jnp.float32:
    """
    READ-ONLY. Y received by seller for selling delta_x of X.

    Paper §Trades:
        P̄- = P- - Δ-/(2λ--)  =  (S - Z-) - Δ-/(2λ--)
        ΔY  = P̄- · Δ-

    In spread-state terms:
        ΔY = (Z- - Δ-/(2λ--)) · Δ-

    Clipped at 0 to avoid negative payments.
    """
    return jnp.maximum(
        (state.z_minus - delta_x / (2.0 * state.lam_mm)) * delta_x,
        0.0
    )


def linear_swap(state: LinearState, side: jnp.int32,
                delta_x: jnp.float32) -> tuple[LinearState, jnp.float32]:
    """
    Execute a trade; return (new_state, delta_y).

    side = 0 → buy  (trader receives X, LP receives Y)
    side = 1 → sell (trader pays X,    LP pays Y)

    Price update (paper §Price update):
        Buy  Δ+: Z+_new = Z+ + Δ+/λ++,  Z-_new = Z- + Δ+/λ-+
        Sell Δ-: Z-_new = Z- - Δ-/λ--,  Z+_new = Z+ - Δ-/λ+-

    Inventory update (paper §Inventory):
        Buy:  X -= Δ+,  Y += ΔY
        Sell: X += Δ-,  Y -= ΔY
    """
    is_buy = (side == 0)

    dy_buy  = linear_curve_buy(state, delta_x)
    dy_sell = linear_curve_sell(state, delta_x)
    delta_y = jnp.where(is_buy, dy_buy, dy_sell)

    # ── Spread state updates ───────────────────────────────────────
    # Buy path
    new_z_plus_buy  = state.z_plus  + delta_x / state.lam_pp
    new_z_minus_buy = state.z_minus - delta_x / state.lam_mp

    # Sell path
    new_z_minus_sell = state.z_minus + delta_x / state.lam_mm
    new_z_plus_sell  = state.z_plus  - delta_x / state.lam_pm

    new_z_plus  = jnp.where(is_buy, new_z_plus_buy,  new_z_plus_sell)
    new_z_minus = jnp.where(is_buy, new_z_minus_buy, new_z_minus_sell)

    # ── Inventory updates ──────────────────────────────────────────
    new_x = jnp.where(is_buy, state.x - delta_x, state.x + delta_x)
    new_y = jnp.where(is_buy, state.y + delta_y, state.y - delta_y)

    new_state = state.replace(
        z_plus=new_z_plus,
        z_minus=new_z_minus,
        x=new_x,
        y=new_y,
    )
    return new_state, delta_y


# ══════════════════════════════════════════════════════════════════
# AMMSpec INSTANCE
# ══════════════════════════════════════════════════════════════════

LINEAR_AMM = AMMSpec(
    init=linear_init,
    curve_buy=linear_curve_buy,
    curve_sell=linear_curve_sell,
    swap=linear_swap,
)


# ══════════════════════════════════════════════════════════════════
# DEPTH FUNCTION  (for routing)
# ══════════════════════════════════════════════════════════════════

def linear_depth(state: LinearState) -> jnp.float32:
    """
    Routing depth = λ++ (buy-side depth).

    Matches paper §Equal best asks:
        Δ_i+ = (λ_i++ / Λ+) · Δ+,  Λ+ = Σ λ_j++
    """
    return state.lam_pp


# ══════════════════════════════════════════════════════════════════
# ARB SOLVER  (closed-form, paper §Step A)
# ══════════════════════════════════════════════════════════════════

def linear_arb_solver(spec: AMMSpec, state: LinearState,
                      fair_price: jnp.float32,
                      epsilon: jnp.float32) -> tuple[jnp.int32, jnp.float32]:
    """
    Closed-form arbitrage solver for the Linear-Impact AMM.

    Paper §Step A: Oracle moves by ε, then arb corrects.

    Post-oracle spreads:
        Z̃+ = Z+ - ε
        Z̃- = Z- + ε

    Ask stale (Z̃+ < 0):
        Δ+_arb = -λ++ · Z̃+
        Post-arb: z+ = 0,  z- = Z̃- + (λ++/λ-+) · Z̃+

    Bid stale (Z̃- < 0):
        Δ-_arb = -λ-- · Z̃-
        Post-arb: z- = 0,  z+ = Z̃+ + (λ--/λ+-) · Z̃-

    No arb:
        z± = Z̃±  (no trade)

    Returns (side, delta_x) where delta_x=0 means no arb.
    Note: fair_price is accepted for interface compatibility
          but not used — arb is fully determined by Z± and ε.
    """
    z_tilde_plus  = state.z_plus  - epsilon
    z_tilde_minus = state.z_minus + epsilon

    # Ask stale: arb buys X from AMM
    dx_buy  = jnp.maximum(-state.lam_pp * z_tilde_plus,  0.0)

    # Bid stale: arb sells X to AMM
    dx_sell = jnp.maximum(-state.lam_mm * z_tilde_minus, 0.0)

    ask_stale = z_tilde_plus  < 0.0
    bid_stale = z_tilde_minus < 0.0
    active    = ask_stale | bid_stale

    side    = jnp.where(ask_stale, jnp.int32(0), jnp.int32(1))
    delta_x = jnp.where(ask_stale, dx_buy, dx_sell)
    delta_x = jnp.where(active, delta_x, jnp.float32(0.0))

    return side, delta_x


# ══════════════════════════════════════════════════════════════════
# EDGE FUNCTION  (paper §The Edge)
# ══════════════════════════════════════════════════════════════════

def linear_edge(state: LinearState, side: jnp.int32,
                delta_x: jnp.float32) -> jnp.float32:
    """
    Compute LP edge for a trade, using pre-trade spread state.

    Paper §The Edge:
        Buy:  Edge+ = Δ+(Z+ + Δ+/(2λ++))
                    = spread_component + impact_component
        Sell: Edge- = Δ-(Z- + Δ-/(2λ--))

    Arb case (Z+ < 0, Δ+ = -λ++·Z+):
        Edge_arb = -λ++(Z+)²/2  ← always negative

    Retail case (Z+ > 0):
        Both spread and impact terms positive ← LP earns

    Note: this is the LINEAR AMM specific edge formula.
    It replaces scoring.compute_edge for this AMM type.
    """
    is_buy = (side == 0)

    edge_buy  = delta_x * (state.z_plus  + delta_x / (2.0 * state.lam_pp))
    edge_sell = delta_x * (state.z_minus + delta_x / (2.0 * state.lam_mm))

    return jnp.where(is_buy, edge_buy, edge_sell)


# ══════════════════════════════════════════════════════════════════
# ROUTING  (paper §Aggregation, two-pool case)
# ══════════════════════════════════════════════════════════════════

def linear_route_two_pools(state1: LinearState, state2: LinearState,
                           side: jnp.int32,
                           total_delta: jnp.float32
                           ) -> tuple[jnp.float32, jnp.float32]:
    """
    Optimal two-pool routing for the Linear-Impact AMM.

    Retail taker maximises revenue (sell) or minimises cost (buy)
    by splitting total_delta across two pools.

    Optimal condition: marginal cost/revenue equal across pools.

    Buy side (taker minimises total Y paid):
        Marginal cost of pool i: Z_i+ + Δ_i/λ_i++
        Setting equal and using Δ1 + Δ2 = Δ:

        Δ1 = (λ1/(λ1+λ2)) · Δ  +  (λ1·λ2/(λ1+λ2)) · (Z2+ - Z1+)

    Sell side (taker maximises total Y received):
        Marginal revenue of pool i: Z_i- - Δ_i/λ_i--
        Setting equal:

        Δ1 = (λ1/(λ1+λ2)) · Δ  +  (λ1·λ2/(λ1+λ2)) · (Z1- - Z2-)

    Both expressions clipped to [0, total_delta] to handle
    corner cases where one pool is strictly dominated.

    Parameters
    ----------
    state1, state2  : LinearState of the two pools
    side            : 0 = buy, 1 = sell
    total_delta     : total order size

    Returns
    -------
    (delta1, delta2) : allocation to each pool, sum = total_delta
    """
    is_buy = (side == 0)

    # ── Buy routing ────────────────────────────────────────────────
    lam1_buy = state1.lam_pp
    lam2_buy = state2.lam_pp
    L_buy    = lam1_buy + lam2_buy + 1e-10

    # Base split (proportional to depth)
    base_buy = (lam1_buy / L_buy) * total_delta
    # Spread correction: pool with lower Z+ gets more flow
    corr_buy = (lam1_buy * lam2_buy / L_buy) * (state2.z_plus - state1.z_plus)
    delta1_buy = base_buy + corr_buy

    # ── Sell routing ───────────────────────────────────────────────
    lam1_sell = state1.lam_mm
    lam2_sell = state2.lam_mm
    L_sell    = lam1_sell + lam2_sell + 1e-10

    base_sell = (lam1_sell / L_sell) * total_delta
    # Spread correction: pool with higher Z- gets more flow
    corr_sell = (lam1_sell * lam2_sell / L_sell) * (state1.z_minus - state2.z_minus)
    delta1_sell = base_sell + corr_sell

    # Select by side
    delta1 = jnp.where(is_buy, delta1_buy, delta1_sell)

    # Clip to [0, total_delta]
    delta1 = jnp.clip(delta1, 0.0, total_delta)
    delta2 = total_delta - delta1

    return delta1, delta2


# ══════════════════════════════════════════════════════════════════
# JAX VERIFICATION
# ══════════════════════════════════════════════════════════════════

def verify_jax_compatibility():
    """Run basic jit / vmap / grad checks."""
    params = LinearParams()
    state  = linear_init(params)

    # jit swap
    jit_swap = jax.jit(linear_swap)
    s2, dy   = jit_swap(state, jnp.int32(0), jnp.float32(1.0))
    assert s2.x < state.x,           "buy should reduce X inventory"
    assert s2.z_plus > state.z_plus,  "buy should push Z+ up"

    # vmap
    batch    = jax.tree.map(lambda x: jnp.stack([x] * 4), state)
    vmap_buy = jax.vmap(linear_curve_buy, in_axes=(0, None))
    dys      = vmap_buy(batch, jnp.float32(1.0))
    assert dys.shape == (4,)

    # grad
    grad_fn = jax.grad(linear_curve_buy, argnums=1)
    slope   = grad_fn(state, jnp.float32(1.0))
    assert slope > 0, "buy cost should increase with size"

    # arb solver — ask stale
    jit_arb = jax.jit(linear_arb_solver, static_argnums=(0,))
    side, dx = jit_arb(LINEAR_AMM, state,
                       jnp.float32(100.0), jnp.float32(0.5))
    assert side == 0, "large positive ε should make ask stale → arb buy"
    assert dx   > 0

    # routing
    state2   = linear_init(LinearParams(init_z_plus=0.5))
    d1, d2   = linear_route_two_pools(state, state2,
                                       jnp.int32(0), jnp.float32(10.0))
    assert abs(float(d1 + d2) - 10.0) < 1e-4, "routing must conserve total flow"
    assert float(d1) > float(d2), "lower Z+ pool should receive more buy flow"

    # edge sign
    e_retail = linear_edge(state, jnp.int32(0), jnp.float32(1.0))
    assert float(e_retail) > 0, "retail edge (Z+>0) should be positive"

    stale = state.replace(z_plus=jnp.float32(-0.3))
    # arb delta = -lambda++ * Z+ = -0.01 * (-0.3) = 0.003
    arb_delta = -float(stale.lam_pp) * float(stale.z_plus)
    e_arb = linear_edge(stale, jnp.int32(0), jnp.float32(arb_delta))
    assert float(e_arb) < 0, "arb edge (Z+<0) should be negative"


if __name__ == "__main__":
    verify_jax_compatibility()
