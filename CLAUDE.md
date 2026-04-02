# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run sanity checks and benchmarks:**
```bash
python -m amm_sim.sanity_check
```
This verifies JAX compatibility (jit/vmap/grad), confirms arb edge ≤ 0 in all steps, runs 50-episode batch rollouts, and prints throughput (steps/sec).

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
The only interface the engine, arb, and router use. All four fields are callables:

| Field | Signature | Description |
|---|---|---|
| `init` | `(params) → state` | Build initial AMM state |
| `curve_buy` | `(state, delta_x: f32) → delta_y: f32` | Read-only: Y cost for buying delta_x of X |
| `curve_sell` | `(state, delta_x: f32) → delta_y: f32` | Read-only: Y received for selling delta_x of X |
| `swap` | `(state, side: i32, delta_x: f32) → (state', delta_y: f32)` | Execute trade, update state. side=0 buy, side=1 sell |

**`marginal_ask(spec, state) → f32`**  
Exact marginal ask price at zero trade size via `jax.grad(spec.curve_buy, argnums=1)(state, 0.0)`.

**`marginal_bid(spec, state) → f32`**  
Exact marginal bid price at zero trade size via `jax.grad(spec.curve_sell, argnums=1)(state, 0.0)`.

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
Output of one `block_step`. All scalar f32 arrays. Stacked by `lax.scan` into shape `(num_steps,)` across a rollout.

| Field | Sign | Description |
|---|---|---|
| `fair_price` | — | Oracle price at end of cycle |
| `epsilon` | — | Oracle log-return this cycle |
| `arb_edge` | ≤ 0 | LP loss to arbitrage across all pools |
| `retail_edge` | ≥ 0 | LP gain from retail flow across all pools |
| `total_edge` | — | `arb_edge + retail_edge` |
| `arb_volume` | — | Total X volume from arb |
| `retail_volume` | — | Total X volume from retail |

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

### [amm_sim/arb.py](amm_sim/arb.py) — Generic arbitrage solver *(new)*

**`generic_arb_solver(spec, state, fair_price, epsilon=0.0) → (side: i32, delta_x: f32)`**

Single arb solver that works for any AMM using only `AMMSpec`. No per-AMM code.

- **Buy arb** when `marginal_ask(spec, state) < fair_price`: find `delta_x` via bisection on `delta_x` such that the post-trade marginal ask equals `fair_price`.
- **Sell arb** when `marginal_bid(spec, state) > fair_price`: find `delta_x` via bisection such that the post-trade marginal bid equals `fair_price`.
- Uses `jax.grad(spec.curve_buy/sell, argnums=1)` for exact marginal prices at each bisection step.
- `epsilon` accepted in signature for interface uniformity; unused (arb is fully determined by `fair_price` vs marginal prices).
- Implemented with `lax.fori_loop` for JAX compatibility inside engine's `lax.scan`.

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
- **Step B**: Python loop over pools (unrolled at trace time): `generic_arb_solver` → `spec.swap` → `scoring.compute_edge`
- **Step C**: `retail_sampler` → `lax.scan` over `max_orders` slots; each slot calls `route_bisection` then per-pool `spec.swap` + `scoring.compute_edge`
- **Step D**: Assembles `CycleRecord`, calls `update_metrics`, returns new `EnvState`

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

Solves: `max Σᵢ fᵢ(Δᵢ)` s.t. `Σᵢ Δᵢ = Δ, Δᵢ ≥ 0`. KKT condition: `fᵢ'(Δᵢ*) = ν` for active pools. Solution: outer bisection on `ν`; inner bisection on `Δ` to invert `fᵢ'(Δ) = ν` using `jax.grad`.

**`route_bisection(specs, states, side, total_delta, num_iters_outer=32, num_iters_inner=32) → (splits: list[f32], nu_star: f32)`**

Single routing function for any AMM type. Uses `jax.grad(spec.curve_buy/sell, argnums=1)` for exact marginal prices — replacing both the old analytic `marginal_inverse_fns` path and the finite-difference fallback. Uses `lax.fori_loop` throughout to avoid `ConcretizationTypeError` when nested inside engine's retail `lax.scan`.

---

### [amm_sim/scoring.py](amm_sim/scoring.py) — Edge calculation

**`compute_edge(side, delta_x, delta_y, fair_price) → f32`**  
Single-trade LP edge. `side=0` (AMM sells X): `edge = delta_y - delta_x * fair_price`. `side=1` (AMM buys X): `edge = delta_x * fair_price - delta_y`. Used by engine for both arb and retail trades across all AMM types.

**`compute_edge_batch(sides, delta_xs, delta_ys, fair_price) → f32 (N,)`**  
Vectorised version for arrays of N trades. Used for offline analysis.

---

### [amm_sim/amms/constant_product.py](amm_sim/amms/constant_product.py) — CP AMM

**Data classes:**

`CPParams` — `fee_plus`, `fee_minus`, `init_x=100`, `init_y=10000`  
`CPState` — `reserve_x`, `reserve_y`, `gamma_plus=1-fee_plus`, `gamma_minus=1-fee_minus`

**Functions (wired into `CONSTANT_PRODUCT_AMM: AMMSpec`):**

| Function | Signature | Formula |
|---|---|---|
| `cp_init` | `(CPParams) → CPState` | Sets reserves and gamma multipliers |
| `cp_curve_buy` | `(state, delta_x) → delta_y` | `(k / (x - delta_x) - y) / gamma_plus` |
| `cp_curve_sell` | `(state, delta_x) → delta_y` | `y - k / (x + delta_x * gamma_minus)` |
| `cp_swap` | `(state, side, delta_x) → (state', delta_y)` | Updates `reserve_x`, `reserve_y`; fee retained in pool |

`verify_jax_compatibility()` — Runs jit/vmap/grad checks.

---

### [amm_sim/amms/linear.py](amm_sim/amms/linear.py) — Linear-impact AMM

**Data classes:**

`LinearParams` — `lam_pp` (λ++), `lam_mm` (λ--), `lam_pm` (λ+-), `lam_mp` (λ-+), `init_p_ask`, `init_p_bid`, `init_x=100`, `init_y=10000`

`LinearState` — `p_ask` (current ask price P+), `p_bid` (current bid price P-), `lam_pp`, `lam_mm`, `lam_pm`, `lam_mp`, `reserve_x`, `reserve_y`

`p_ask` and `p_bid` are the pool's absolute bid/ask prices, updated only when trades occur in `linear_swap`. `reserve_x`/`reserve_y` track net inventory and are required by `env.py` for fair price initialization and `make_obs`.

**Functions (wired into `LINEAR_AMM: AMMSpec`):**

| Function | Signature | Formula |
|---|---|---|
| `linear_init` | `(LinearParams) → LinearState` | Sets `p_ask`, `p_bid` from params; `reserve_x/y` from `init_x/y` |
| `linear_curve_buy` | `(state, delta_x) → delta_y` | `(p_ask + delta_x / (2λ++)) * delta_x` |
| `linear_curve_sell` | `(state, delta_x) → delta_y` | `max((p_bid - delta_x / (2λ--)) * delta_x, 0)` |
| `linear_swap` | `(state, side, delta_x) → (state', delta_y)` | Updates `p_ask`, `p_bid` per cross-impact matrix; updates `reserve_x`, `reserve_y` |

Price impact in `linear_swap`:
- Buy `Δ+`: `p_ask += Δ+/λ++`, `p_bid += Δ+/λ-+`
- Sell `Δ-`: `p_bid -= Δ-/λ--`, `p_ask -= Δ-/λ+-`

`verify_jax_compatibility()` — Runs jit/vmap/grad checks.

---

## JAX constraints

- All state objects are `chex.dataclass(frozen=True)` — immutable pytrees; never mutate in-place.
- `SimParams` fields used as loop bounds (`num_steps`, `max_orders`) must remain Python ints — they are passed as static args to `jit`.
- Engine uses `lax.scan` for the retail order loop. `arb.py` and `router.py` use `lax.fori_loop` to avoid `ConcretizationTypeError` when nested inside the retail scan.
- Python loops over `num_pools` in the engine are unrolled at trace time — `num_pools` is always a static Python int.
- Edge is always computed on the **pre-trade** state (before `swap` updates reserves).

## Adding a new AMM

1. Create `amm_sim/amms/my_amm.py` with frozen `chex.dataclass` for params and state. State must include `reserve_x` and `reserve_y` fields for `env.py` compatibility.
2. Implement `init`, `curve_buy`, `curve_sell`, `swap` using `jnp` ops only (no Python control flow on traced values).
3. Create `MY_AMM = AMMSpec(init=..., curve_buy=..., curve_sell=..., swap=...)`.
4. Pass `MY_AMM` in `amm_specs` to `make_env()`. No arb solver, edge function, or routing function needed — these are handled generically by `arb.py`, `scoring.py`, and `router.py` via `jax.grad` on `curve_buy`/`curve_sell`.
