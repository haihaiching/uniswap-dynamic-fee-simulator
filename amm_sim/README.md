# uniswap-dynamic-fee-simulation

This document explains every file in the simulator, what it does, and how the pieces connect.

---

## Repository Structure

```text
amm_sim/
├── spec.py                  # AMM interface contract
├── types.py                 # Shared data structures
├── scoring.py               # Edge calculation
├── engine.py                # Simulation loop
├── env.py                   # Environment wrapper (Gymnax API)
├── sanity_check.py          # Phase 5 verification and benchmarking
└── amms/
    ├── constant_product.py  # Uniswap V2 style AMM (asymmetric fees)
    └── linear.py            # Linear-Impact AMM
```

---

## `spec.py` — AMM Interface

Defines `AMMSpec`, a `NamedTuple` of four callables that every AMM must implement.

```python
class AMMSpec(NamedTuple):
    init:       Callable   # (params) → state
    curve_buy:  Callable   # (state, delta_x) → delta_y  [read-only]
    curve_sell: Callable   # (state, delta_x) → delta_y  [read-only]
    swap:       Callable   # (state, side, delta_x) → (state', delta_y)
```

**Why this design:**
The engine never reads AMM state fields directly — it only calls these four functions. This means adding a new AMM type only requires a new file in `amms/`. No changes to `engine.py`, `env.py`, or any other core file.

**`marginal_ask` / `marginal_bid`:**
Engine-level helpers that compute marginal price from the bonding curve by sending an infinitesimal order (`curve_buy(state, eps) / eps`). Works for any AMM shape.

---

## `types.py` — Data Structures

Defines four data structures used throughout the simulator.

### `SimParams`
Static configuration fixed at construction time. Lives in Python, never on GPU. Controls compile-time constants like `lax.scan` loop lengths.

| Field | Description |
|-------|-------------|
| `sigma` | GBM volatility per step |
| `num_steps` | Episode length — **static**, controls `lax.scan` length |
| `max_orders` | Max retail orders per step — **static**, Poisson arrivals clipped here |
| `lam` | Poisson arrival rate (avg orders per step) |
| `mu` | LogNormal mean for order size |
| `sigma_ln` | LogNormal std for order size |
| `phi` | Inventory penalty in reward (0 = pure edge maximisation) |

### `CycleRecord`
Output of one `block_step` call. After a `rollout()`, each field has shape `(num_steps,)`.

| Field | Description |
|-------|-------------|
| `fair_price` | Oracle price at end of step |
| `epsilon` | Oracle log-return this step |
| `arb_edge` | Total arb edge (≤ 0, cost to LP) |
| `retail_edge` | Total retail edge (≥ 0, income to LP) |
| `total_edge` | `arb_edge + retail_edge` |
| `arb_volume` | X volume traded by arbitrageurs |
| `retail_volume` | X volume traded by retail takers |

### `Metrics`
Cumulative accumulators carried inside `EnvState` across `lax.scan` steps.

### `EnvState`
Complete simulator state — the carry object for `lax.scan`. All fields must have static shapes for `vmap` over episodes to work.

---

## `scoring.py` — Edge Calculation

Single source of truth for LP performance measurement.

**Edge definition:**
```text
Edge = cash received by AMM − fair value of inventory given up
```

| Trade direction | Formula |
|----------------|---------|
| AMM sells X (side=0) | `delta_y − delta_x * fair_price` |
| AMM buys X  (side=1) | `delta_x * fair_price − delta_y` |

**Why `fair_price` not `spot`:**
Spot is the AMM's own price — using it gives edge ≈ 0 always. Fair price is the external oracle — the gap between fair price and spot is exactly what arbitrageurs exploit.

---

## `engine.py` — Simulation Loop

Contains `make_engine()` which returns a `jit`-compiled `block_step` function.

### One cycle (`block_step`)

```text
Step A — Oracle:   fair price moves via GBM, returns epsilon
Step B — Arb:      each pool's arb solver corrects stale prices independently
Step C — Retail:   taker orders arrive, split optimally, each pool executes
Step D — Package:  results → CycleRecord, update EnvState
```

### `route_two_pools` (numerical bisection)

Finds the optimal split of a retail order across two pools.
**Objective:** maximise total output received by the taker.
**Method:** bisect on marginal output equality. If `marginal_output(pool1, d1) > marginal_output(pool2, d2)`, shift more to pool1. Works for any bonding curve.

---

## `env.py` — Environment Wrapper

Wraps `engine.py` in a Gymnax-compatible API.

- `reset(key, sim_params)`: Initialises all AMM states.
- `step(state, action, sim_params)`: Runs one `block_step`. Reward = `total_edge − phi * inventory²`.
- `rollout(key, sim_params, policy_fn=None)`: Runs a full episode via `lax.scan`.
- `batch_rollout(keys, sim_params)`: Runs N episodes in parallel via `vmap`.

---

## `sanity_check.py` — Verification & Benchmarking

Verification script to confirm the simulator is working correctly before writing any strategy code.

**Checks:**
1. **Arb edge always <= 0**: Ensures AMMs always lose to arbitrageurs.
2. **Symmetry**: Two identical AMMs produce the exact same edge.
3. **Stability**: `batch_rollout` runs without error across 100 episodes.
4. **Timing benchmark**: Measures execution speed (steps/sec) after JIT warmup.

---

## `amms/constant_product.py` — CP AMM

Constant-product AMM (`x·y=k`) with asymmetric fees.

**Asymmetric fees:**
`gamma_plus = 1 - fee_plus` (buy-side)
`gamma_minus = 1 - fee_minus` (sell-side)

**Arb solver (closed-form):**
- Arb buy: `x* = sqrt(k / (gamma_plus * fair_price))`
- Arb sell: `x* = sqrt(gamma_minus * k / fair_price)`

---

## `amms/linear.py` — Linear-Impact AMM

AMM with linear price impact. Tracks spread state and inventory.

**Bonding curve:**
- Buy Δ+: ΔY paid = (Z+ + Δ+/(2λ++)) · Δ+
- Sell Δ-: ΔY recv = (Z- − Δ-/(2λ--)) · Δ-

**Arb solver:**
Adjusts spreads by oracle move `epsilon`. Triggers arb if `Z_new < 0`.

---

## How the pieces connect

```text
make_env(amm_specs, amm_params, arb_solvers, edge_fns)
    └── make_engine(...)
            └── block_step (jit-compiled)
                    ├── default_oracle      (GBM price update)
                    ├── arb_solvers[i]      (one per pool, from amms/)
                    ├── route_two_pools     (bisection)
                    ├── amm_specs[i].swap   (one per pool, from amms/)
                    └── edge_fns[i]         (one per pool, from amms/)
```

---

## Adding a new AMM

1. Create `amm_sim/amms/my_amm.py`.
2. Define `MyParams`, `MyState` (`chex.dataclass`).
3. Implement `init`, `curve_buy`, `curve_sell`, `swap`.
4. Provide `arb_solver` and `edge_fn`.
5. Plug it into `make_env()`.