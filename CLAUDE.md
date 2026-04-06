# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run sanity checks and benchmarks:**
```bash
python -m amm_sim.sanity_check
```
Runs per-AMM checks via `check_amm(spec, params, name, check_retail_after_arb=True)`:
1. **Arb edge always ≤ 0** — verified on every step of a single rollout.
2. **First retail edge after arb ≥ 0** — checked only in steps where `arb_volume > 0`. Skipped for AMMs with cross-impact (pass `check_retail_after_arb=False`) where a sell arb can drag the ask below fair — known model behaviour.
3. Two identical pools produce equal routing splits.
4. `batch_rollout` 100 episodes without error.
5. Timing benchmark.

**Run a notebook:**
```bash
jupyter notebook amm_sim_linear.ipynb        # Linear AMM demo
jupyter notebook amm_sim_multi_pool.ipynb    # Multi-pool routing demo
```

There are no separate build, lint, or install commands — `sanity_check.py` is the primary verification tool.

---

## Module dependency graph

```
sanity_check.py
    └── env.py              ← primary user-facing entry point
            ├── engine.py
            │       ├── spec.py
            │       ├── types.py
            │       ├── scoring.py
            │       ├── router.py
            │       └── arb.py
            └── types.py

amms/constant_product.py  ←─┐
amms/linear.py            ←─┤  implement AMMSpec from spec.py
                            │  and are wired into env.py via make_env()
spec.py  ──────────────────┘
```

**Call sequence for one episode:**

```
make_env(amm_specs, amm_params, ...)
  └─ make_engine(...)           # engine.py → returns jit-compiled block_step
       └─ block_step(state, sim_params)    # called num_steps times via lax.scan
              ├─ oracle_fn(fair_price, key, sim_params)      # Step A: GBM price
              ├─ generic_arb_solver(spec, state, fair_price) # Step B: per-pool arb
              │       └─ spec.curve_buy / curve_sell         #   via jax.grad bisection
              │       └─ spec.swap(state, side, delta_x)
              │       └─ scoring.compute_edge(...)
              └─ retail_sampler(key, sim_params)             # Step C: retail orders
                      └─ route_bisection(specs, states, side, total_delta)  # router.py
                              └─ jax.grad(spec.curve_buy/sell)  # marginal via grad
                      └─ spec.swap(state, side, delta_x)    # per-pool per-order
                      └─ scoring.compute_edge(...)
```

**Design principle:** `engine.py`, `arb.py`, and `router.py` depend only on `AMMSpec` (the four-function interface). Per-AMM files (`constant_product.py`, `linear.py`) contain only the bonding curve and state — no arb, edge, or routing logic.

---

## Module reference

### [amm_sim/spec.py](amm_sim/spec.py) — AMM interface

**`AMMSpec`** *(NamedTuple)*  
The only interface the engine, arb, and router use. All six fields are callables:

| Field | Signature | Description |
|---|---|---|
| `init` | `(params) → state` | Build initial AMM state |
| `curve_buy` | `(state, delta_x: f32) → delta_y: f32` | Read-only: Y cost for buying delta_x of X |
| `curve_sell` | `(state, delta_x: f32) → delta_y: f32` | Read-only: Y received for selling delta_x of X |
| `swap` | `(state, side: i32, delta_x: f32) → (state', delta_y: f32)` | Execute trade, update state. side=0 buy, side=1 sell |
| `max_trade_buy` | `(state) → f32` | Max delta_x the AMM accepts on the buy side |
| `max_trade_sell` | `(state) → f32` | Max delta_x the AMM accepts on the sell side |

**`marginal_ask(spec, state, delta_x=0.0) → f32`**  
Exact marginal ask price after purchasing `delta_x` units of X, via `jax.grad(spec.curve_buy, argnums=1)(state, delta_x)`. Defaults to `delta_x=0` (spot price before any trade).

**`marginal_bid(spec, state, delta_x=0.0) → f32`**  
Exact marginal bid price after selling `delta_x` units of X, via `jax.grad(spec.curve_sell, argnums=1)(state, delta_x)`. Defaults to `delta_x=0` (spot price before any trade).

---

### [amm_sim/types.py](amm_sim/types.py) — Data structures

All are frozen `chex.dataclass` (immutable JAX pytrees).

**`SimParams`**  
Static parameters. Passed as a static arg to `jit` — controls compile-time constants like loop bounds.

| Field | Type | Default | Description |
|---|---|---|---|
| `sigma` | float | 0.001 | GBM volatility per step |
| `num_steps` | int | 1000 | Episode length (controls `lax.scan` length) |
| `max_orders` | int | 16 | Max retail orders per step (controls fixed-length order array) |
| `lam` | float | 0.8 | Poisson arrival rate for retail orders |
| `mu` | float | 20.0 | LogNormal mean for order size (in Y terms) |
| `sigma_ln` | float | 1.2 | LogNormal std for order size |
| `phi` | float | 0.0 | Inventory penalty coefficient in reward |

**`EnvState`**  
Full simulator state passed through every `block_step` call.

| Field | Type | Description |
|---|---|---|
| `amm_states` | list[pytree] | One state per pool (length = num_pools, static) |
| `fair_price` | f32 scalar | Current oracle price |
| `step_idx` | i32 scalar | Current step index |
| `rng_key` | u32 (2,) | JAX PRNG key |
| `metrics` | `Metrics` | Running accumulators |

**`CycleRecord`**  
Output of one `block_step`. Scalar fields are stacked by `lax.scan` into shape `(num_steps,)`; per-pool array fields into shape `(num_steps, num_pools)`.

| Field | Shape | Sign | Description |
|---|---|---|---|
| `fair_price` | scalar | — | Oracle price at end of cycle |
| `epsilon` | scalar | — | Oracle log-return this cycle |
| `arb_edge` | scalar | ≤ 0 | LP loss to arbitrage across all pools |
| `retail_edge` | scalar | — | LP gain from retail flow across all pools |
| `total_edge` | scalar | — | `arb_edge + retail_edge` |
| `arb_volume` | scalar | — | Total X volume from arb |
| `retail_volume` | scalar | — | Total X volume from retail |
| `arb_edges_per_pool` | `(num_pools,)` | — | Per-pool arb edge |
| `retail_edges_per_pool` | `(num_pools,)` | — | Per-pool retail edge |

**`Metrics`**  
Cumulative accumulators updated every step inside `EnvState`.

| Field | Description |
|---|---|
| `total_edge` | Cumulative total edge since episode start |
| `total_arb_edge` | Cumulative arb edge |
| `total_ret_edge` | Cumulative retail edge |
| `inventory` | Current X inventory of pool 0 (agent's pool) |

**`zero_metrics() → Metrics`**  
Returns zeroed `Metrics` struct for episode reset.

**`update_metrics(metrics, record, agent_inventory) → Metrics`**  
Adds one `CycleRecord` into running `Metrics`. Called at end of each `block_step`.

---

### [amm_sim/arb.py](amm_sim/arb.py) — Generic arbitrage solver

**`generic_arb_solver(spec, state, fair_price, num_iters=100, tol=1e-6) → (side: i32, delta_x: f32)`**

Single arb solver that works for any AMM using only `AMMSpec`.

- **Buy arb** when `marginal_ask(spec, state, ε) < fair_price`: bisect on `delta_x ∈ [0, spec.max_trade_buy(state)·(1−ε)]` until `marginal_ask = fair_price`.
- **Sell arb** when `marginal_bid(spec, state, ε) > fair_price`: bisect on `delta_x ∈ [0, spec.max_trade_sell(state)·(1−ε)]` until `marginal_bid = fair_price`.
- Bisection bounds come from `spec.max_trade_buy/sell` — both now return `state.max_trade_x`, a hard per-trade cap that bounds bisection to a well-conditioned range.
- Trigger evaluated at `delta_x = ε = 1e-6` (not 0) for numerical stability.
- Returns `dx=0` when fair is inside the spread (no arb opportunity).
- Implemented with `lax.while_loop` — stops early when `hi - lo < tol` (default `1e-6`), with `num_iters=100` as a hard cap. Compatible with engine's `lax.scan`.

---

### [amm_sim/engine.py](amm_sim/engine.py) — Simulation engine

**`make_engine(amm_specs, oracle_fn=None, retail_sampler=None) → block_step`**

Builds and returns a `jit`-compiled `block_step`. Per-AMM arb and edge logic are no longer arguments — the engine calls `generic_arb_solver` from `arb.py` and `scoring.compute_edge` from `scoring.py` for all pools uniformly.

| Argument | Type | Description |
|---|---|---|
| `amm_specs` | `list[AMMSpec]` | One per pool |
| `oracle_fn` | Callable or None | Custom oracle; defaults to `default_oracle` |
| `retail_sampler` | Callable or None | Custom sampler; defaults to `default_retail_sampler` |

Returns `block_step(state: EnvState, sim_params: SimParams) → (EnvState, CycleRecord)` — jit-compiled with `sim_params` as static arg.

**`default_oracle(fair_price, key, sim_params) → (epsilon, new_price, key)`**  
Zero-drift GBM: `p(t+1) = p(t) * exp(-σ²/2 + σZ)`, `Z ~ N(0,1)`. `epsilon` is the log-return for this step.

**`default_retail_sampler(key, sim_params) → (sides, sizes, key)`**  
Generates `max_orders`-length arrays. Count `N ~ Poisson(lam)` clipped to `max_orders`. Sizes `~ LogNormal(mu, sigma_ln)`. Directions 50/50. Inactive slots have `size=0` (masked no-op).

**`block_step(state, sim_params) → (EnvState, CycleRecord)`** *(inner, jit-compiled)*
- **Step A**: `oracle_fn` → `epsilon`, `new_fair`
- **Step B**: Python loop over pools (unrolled at trace time): `generic_arb_solver` → `spec.swap` → `scoring.compute_edge`; per-pool arb edge stored in `arb_edge_pp[i]`
- **Step C**: `retail_sampler` → `lax.scan` over `max_orders` slots; each slot calls `route_bisection` then per-pool `spec.swap` + `scoring.compute_edge`; per-pool retail edge accumulated in `ret_edge_pp[j]`
- **Step D**: Assembles `CycleRecord` (including `arb_edges_per_pool`, `retail_edges_per_pool`), calls `update_metrics`, returns new `EnvState`

Edge always computed on **pre-trade** state.

---

### [amm_sim/env.py](amm_sim/env.py) — Environment wrapper

**`make_env(amm_specs, amm_params, oracle_fn=None, retail_sampler=None) → Env`**

Top-level factory. Calls `make_engine(amm_specs, ...)` internally. Returns an `Env` object with:

| Method | Signature | Description |
|---|---|---|
| `reset` | `(key, sim_params) → (obs, state)` | Init all AMM states; sets `fair_price = reserve_y/reserve_x` of pool 0 |
| `step` | `(state, action, sim_params) → (obs, state, reward, done, CycleRecord)` | One cycle. `reward = total_edge - phi * inventory²`. `action` accepted for Gymnax API compatibility. |
| `rollout` | `(key, sim_params, policy_fn=None) → (final_state, trajectory)` | Full episode via `lax.scan`. Each `CycleRecord` field has shape `(num_steps,)`. |
| `batch_rollout` | `(keys, sim_params, policy_fn=None) → (final_states, trajectories)` | N parallel episodes via `vmap`. `keys` shape `(N, 2)`. Each field shape `(N, num_steps)`. |
| `block_step` | `(state, sim_params) → (EnvState, CycleRecord)` | Raw compiled step, exposed for debugging. |

**`make_obs(state, sim_params) → jnp.ndarray[3]`**  
`[fair_price, pool0.reserve_x, time_fraction]`. Fixed shape 3 required for `vmap`.

---

### [amm_sim/router.py](amm_sim/router.py) — Order routing

Trader-optimal routing (best execution): each side solved as a separate convex program.

- **Buy** `(side=0)`: `min Σᵢ curve_buy_i(Δᵢ)` s.t. `Σᵢ Δᵢ = Δ` — minimise total Y paid by buyer. `curve_buy` is convex → KKT equates marginal costs → `g(ν) = Σ fᵢ'⁻¹(ν)` is **increasing** in ν.
- **Sell** `(side=1)`: `max Σᵢ curve_sell_i(Δᵢ)` s.t. `Σᵢ Δᵢ = Δ` — maximise total Y received by seller. `curve_sell` is concave → KKT equates marginal receipts → `g(ν)` is **decreasing** in ν.

KKT condition: `fᵢ'(Δᵢ*) = ν` for active pools. Solution: outer bisection on `ν` (direction flipped for buy vs sell); inner bisection on `Δ` to invert `fᵢ'(Δ) = ν` using `jax.grad`.

**`route_bisection(specs, states, side, total_delta, num_iters_outer=32, num_iters_inner=32, nu_hi=1e6) → (splits: list[f32], nu_star: f32)`**

Single routing function for any AMM type. Uses `jax.grad(spec.curve_buy/sell, argnums=1)` for exact marginal prices — selected via `lax.cond` so only the relevant gradient is evaluated at runtime. Per-pool bisection upper bounds (`min(total_delta, max_trade_x)`) are precomputed once before the outer loop and passed into `marginal_inverse` — a saturated pool is clamped at its cap and remaining flow absorbed by other pools. Uses `lax.fori_loop` throughout to avoid `ConcretizationTypeError` when nested inside engine's retail `lax.scan`.

---

### [amm_sim/scoring.py](amm_sim/scoring.py) — Edge calculation

**`compute_edge(side, delta_x, delta_y, fair_price) → f32`**  
Single-trade LP edge. `side=0` (AMM sells X): `edge = delta_y - delta_x * fair_price`. `side=1` (AMM buys X): `edge = delta_x * fair_price - delta_y`. Used by engine for both arb and retail trades across all AMM types.

**`compute_edge_batch(sides, delta_xs, delta_ys, fair_price) → f32 (N,)`**  
Vectorised version for arrays of N trades. Used for offline analysis.

---

### [amm_sim/amms/constant_product.py](amm_sim/amms/constant_product.py) — CP AMM

**Data classes:**

`CPParams` — `fee_plus`, `fee_minus`, `init_x=100`, `init_y=10000`, `max_trade_x=2.0`  
`CPState` — `reserve_x`, `reserve_y`, `gamma_plus=1-fee_plus`, `gamma_minus=1-fee_minus`, `max_trade_x`

**Functions (wired into `CONSTANT_PRODUCT_AMM: AMMSpec`):**

| Function | Signature | Formula / Rule |
|---|---|---|
| `cp_init` | `(CPParams) → CPState` | Sets reserves, gamma multipliers, and max_trade_x |
| `cp_curve_buy` | `(state, delta_x) → delta_y` | `y · Δx / ((x − Δx) · γ+)` — clips Δx ≤ `max_trade_x` |
| `cp_curve_sell` | `(state, delta_x) → delta_y` | `y · Δx · γ− / (x + Δx · γ−)` — clips Δx ≤ `max_trade_x` |
| `cp_swap` | `(state, side, delta_x) → (state', delta_y)` | Clips to `max_trade_x`, then: Buy `rx -= Δx, ry += Δy`; Sell `rx += Δx, ry -= Δy` |
| `cp_max_trade_buy` | `(state) → f32` | `state.max_trade_x` — hard per-trade cap |
| `cp_max_trade_sell` | `(state) → f32` | `state.max_trade_x` — hard per-trade cap |

Fees are fully encoded in the curve formulas. `max_trade_x` is a hard per-trade cap on delta_x, independent of inventory, stored in state so it can be updated dynamically. Both sides use the same cap.

Arb, edge, routing, and marginal inverse functions have been removed — these are now handled generically by `arb.py`, `scoring.py`, and `router.py`.

`verify_jax_compatibility()` — Runs jit/vmap/grad checks (jit, vmap, grad on `cp_curve_buy`).

---

### [amm_sim/amms/linear.py](amm_sim/amms/linear.py) — Linear-impact AMM

**Data classes:**

`LinearParams` — `lam_pp` (λ++), `lam_mm` (λ--), `lam_pm` (λ+-), `lam_mp` (λ-+), `init_p_ask=100.2`, `init_p_bid=99.8`, `init_x=100`, `init_y=10000`, `max_trade_x=2.0`

`LinearState` — `p_ask`, `p_bid`, `lam_pp`, `lam_mm`, `lam_pm`, `lam_mp`, `reserve_x`, `reserve_y`, `max_trade_x`

`p_ask`/`p_bid` are absolute pool prices updated only when trades occur in `linear_swap` — the oracle no longer touches AMM state. `reserve_x`/`reserve_y` are net inventories; `reserve_y/reserve_x` gives the initial fair price used by `env.py`.

**Functions (wired into `LINEAR_AMM: AMMSpec`):**

| Function | Signature | Formula / Rule |
|---|---|---|
| `linear_init` | `(LinearParams) → LinearState` | Sets all fields from params including `max_trade_x` |
| `linear_curve_buy` | `(state, delta_x) → delta_y` | `(p_ask + Δx/(2λ++))·Δx`; clips Δx ≤ `max_trade_x` |
| `linear_curve_sell` | `(state, delta_x) → delta_y` | `max((p_bid - Δx/(2λ--))·Δx, 0)`; clips Δx ≤ `max_trade_x` |
| `linear_swap` | `(state, side, delta_x) → (state', delta_y)` | Clips to `max_trade_x`, then updates `p_ask`, `p_bid` per cross-impact matrix and `reserve_x`, `reserve_y` |
| `linear_max_trade_buy` | `(state) → f32` | `state.max_trade_x` — hard per-trade cap |
| `linear_max_trade_sell` | `(state) → f32` | `state.max_trade_x` — hard per-trade cap |

Price impact in `linear_swap` (using clipped `eff_dx = min(delta_x, max_trade_x)`):
- Buy `Δ+`: `p_ask += Δ+/λ++`, `p_bid += Δ+/λ-+`
- Sell `Δ-`: `p_bid -= Δ-/λ--`, `p_ask -= Δ-/λ+-`

Note: cross-impact means a large sell arb that corrects `p_bid → fair` may simultaneously drag `p_ask` below fair. This is valid model behaviour, not a bug. The sanity check only verifies the arbed side is corrected.

Default params: `lam_pp=200, lam_mm=200, lam_pm=100, lam_mp=100` (deep book, low impact per unit). Sanity check uses `lam_pp=lam_mm=2, lam_pm=lam_mp=1` (shallower book, larger impact).

Arb, edge, routing, marginal inverse, and depth functions have been removed — handled generically by `arb.py`, `scoring.py`, and `router.py`.

`verify_jax_compatibility()` — Runs jit/vmap/grad checks (jit, vmap, grad on `linear_curve_buy`; asserts `p_ask`/`p_bid`/`reserve_x` update correctly after a buy).

---

## JAX constraints

- All state objects are `chex.dataclass(frozen=True)` — immutable pytrees; never mutate in-place.
- `SimParams` fields used as loop bounds (`num_steps`, `max_orders`) must remain Python ints — they are passed as static args to `jit`.
- Engine uses `lax.scan` for the retail order loop. `arb.py` uses `lax.while_loop`; `router.py` uses `lax.fori_loop` to avoid `ConcretizationTypeError` when nested inside the retail scan.
- Python loops over `num_pools` in the engine are unrolled at trace time — `num_pools` is always a static Python int.
- Edge is always computed on the **pre-trade** state (before `swap` updates reserves).

## Adding a new AMM

1. Create `amm_sim/amms/my_amm.py` with frozen `chex.dataclass` for params and state. State must include `reserve_x`, `reserve_y`, and `max_trade_x` fields.
2. Implement `init`, `curve_buy`, `curve_sell`, `swap` using `jnp` ops only (no Python control flow on traced values). Both curves must clip `delta_x` to `state.max_trade_x`; `swap` must also clip before updating prices/reserves.
3. Implement `max_trade_buy(state) → state.max_trade_x` and `max_trade_sell(state) → state.max_trade_x`. These are used as bisection bounds in `arb.py` and `router.py`.
4. Create `MY_AMM = AMMSpec(init=..., curve_buy=..., curve_sell=..., swap=..., max_trade_buy=..., max_trade_sell=...)`.
5. Pass `MY_AMM` in `amm_specs` to `make_env()`. No arb solver, edge function, or routing function needed — these are handled generically by `arb.py`, `scoring.py`, and `router.py` via `jax.grad` on `curve_buy`/`curve_sell`.
