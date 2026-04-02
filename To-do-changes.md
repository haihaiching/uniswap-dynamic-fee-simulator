# Planned changes

## 1. `spec.py` — use `jax.grad` for marginal prices

Replace finite-difference approximations in `marginal_ask` and `marginal_bid` with `jax.grad`:

```python
def marginal_ask(spec, state):
    return jax.grad(spec.curve_buy, argnums=1)(state, jnp.float32(0.0))

def marginal_bid(spec, state):
    return jax.grad(spec.curve_sell, argnums=1)(state, jnp.float32(0.0))
```

Both `cp_curve_buy` and `linear_curve_buy` are smooth in `delta_x`, so `jax.grad` applies cleanly. This gives exact derivatives, compiles to a single XLA op, and eliminates the O(eps) approximation error.

---

## 2. `engine.py` — modular arb via generic solver (Option A)

Move arbitrage logic out of `engine.py` into a new **`arb.py`** module. Implement a single `generic_arb_solver` that works for any AMM using only `AMMSpec`:

- **Buy arb** condition: `marginal_ask(spec, state) < fair_price`  
  Find `delta_x` such that `marginal_ask` after the trade equals `fair_price` — solved by bisection on `delta_x` using `jax.grad`-based marginals.
- **Sell arb** condition: `marginal_bid(spec, state) > fair_price`  
  Find `delta_x` such that `marginal_bid` after the trade equals `fair_price`.

Signature: `generic_arb_solver(spec, state, fair_price, epsilon) → (side: i32, delta_x: f32)`  
(`epsilon` retained for interface compatibility; unused since arb is determined by `fair_price` vs marginal prices.)

`engine.py` calls this single function for all pools — no per-AMM arb logic remains in the engine. The `arb_solvers` argument to `make_engine` is replaced by this single generic function.

Similarly, the edge computed in the engine can use `scoring.compute_edge` generically (see point 4), removing `edge_fns` as an argument.

---

## 3. `router.py` — replace finite differences with `jax.grad`

The `route_bisection_numerical` fallback currently uses finite differences to approximate marginal prices. Replace with `jax.grad`:

```python
def marginal_output(spec, state, delta):
    return jax.grad(
        lambda d: jnp.where(is_buy, spec.curve_buy(state, d), spec.curve_sell(state, d)),
        argnums=0
    )(delta)
```

Remove the analytic `marginal_inverse_fns` path entirely. All routing goes through a single `jax.grad`-based bisection path:

- Outer bisection: on shadow price `ν`
- Inner bisection: on `delta` to invert `f'(delta) = ν`

This simplifies `make_engine` (no `marginal_inverse_fns` argument) and router.py (one code path). The speed cost of inner bisection is acceptable since the whole routine is inside a jit-compiled `lax.scan`.

---

## 4. `amms/constant_product.py` — remove AMM-specific solvers

Remove:
- `cp_arb_solver` — replaced by `generic_arb_solver` in `arb.py`
- `cp_edge` — engine uses `scoring.compute_edge` directly (delta_y from `curve_buy`/`curve_sell`, which is generic)
- `cp_marginal_inverse_buy`, `cp_marginal_inverse_sell` — replaced by `jax.grad`-based routing
- `cp_depth` — was only used for proportional routing

Keep: `CPParams`, `CPState`, `cp_init`, `cp_curve_buy`, `cp_curve_sell`, `cp_swap`, `CONSTANT_PRODUCT_AMM`, `verify_jax_compatibility`.

---

## 5. `amms/linear.py` — explicit bid/ask price state + remove AMM-specific solvers

### 5a. State representation: store explicit bid/ask prices

Replace spread state `(z_plus, z_minus)` with explicit pool prices `(p_ask, p_bid)`:

- `p_ask = P+` — current ask price (LP sells X at this price)  
- `p_bid = P-` — current bid price (LP buys X at this price)

These are updated only when trades occur (in `linear_swap`). Since the generic arb solver (point 2) determines arb trades via `marginal_ask`/`marginal_bid` — which read `p_ask`/`p_bid` through `curve_buy`/`curve_sell` — the arb no longer needs to touch spread state directly. The oracle epsilon no longer needs to flow into the AMM state.

Updated bonding curve formulas in terms of `p_ask`, `p_bid`:
- Buy: `delta_y = (p_ask + delta_x / (2λ++)) * delta_x`
- Sell: `delta_y = (p_bid - delta_x / (2λ--)) * delta_x`

Updated `linear_swap` price impact:
- Buy `Δ+`: `p_ask_new = p_ask + Δ+/λ++`, `p_bid_new = p_bid + Δ+/λ-+`
- Sell `Δ-`: `p_bid_new = p_bid - Δ-/λ--`, `p_ask_new = p_ask - Δ-/λ+-`

New `LinearState` fields: `p_ask`, `p_bid`, `lam_pp`, `lam_mm`, `lam_pm`, `lam_mp`, `reserve_x`, `reserve_y`.

### 5b. Remove AMM-specific solvers

Remove:
- `linear_arb_solver` — replaced by `generic_arb_solver` in `arb.py`
- `linear_edge` — engine uses `scoring.compute_edge` directly (generic)
- `linear_route_two_pools` — replaced by `jax.grad`-based routing in `router.py`
- `linear_marginal_inverse_buy`, `linear_marginal_inverse_sell` — replaced by `jax.grad`
- `linear_depth` — was only used for proportional routing

Keep: `LinearParams`, `LinearState`, `linear_init`, `linear_curve_buy`, `linear_curve_sell`, `linear_swap`, `LINEAR_AMM`, `verify_jax_compatibility`.

---

## Implementation order

1. `spec.py` — swap finite diff for `jax.grad` (self-contained, no dependencies)
2. `amms/constant_product.py` and `amms/linear.py` — strip to core 4 functions + state change for Linear
3. `router.py` — replace numerical fallback with `jax.grad`, remove analytic path
4. `arb.py` (new) — implement `generic_arb_solver`
5. `engine.py` — remove `arb_solvers` and `edge_fns` arguments, call `generic_arb_solver` and `scoring.compute_edge` directly
6. `env.py` — update `make_env` signature to match new `make_engine`
7. `sanity_check.py` — update to use new API
