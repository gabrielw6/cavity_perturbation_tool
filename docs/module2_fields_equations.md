# Module 2 — Field Provider Abstraction: Equations & Implementation Plan

Scope: `FieldProvider` (ABC), `AnalyticalField`, the `RitzField` stub, per the interface
already fixed in `architecture_modules_1-5.md`. No code here — equations and build order only.

Module 2 owns almost no physics of its own — Module 1 already produced exact, closed-form
$E$, $H$. What Module 2 owns is turning "$E$ and $H$ evaluated at a point" into "$\int_{V_s}|E|^2\,dV$
and $\int_{V_s}|H|^2\,dV$ over an arbitrary sample region," which is a numerical-integration
problem, not an electromagnetics problem. Keep that boundary sharp: if you find yourself
deriving a new field formula in this module, it belongs in Module 1 instead.

---

## 0. What Module 2 owns vs. what it consumes from Module 3

`region.quadrature_points(n)` — the actual point/weight generation — belongs to Module 3
(`SampleRegion`), not here. Module 2 is a **consumer** of that contract, not an implementer
of it. But Module 2 should not blindly trust it either: Section 1.4 below specifies a
defensive check Module 2 runs on every call, regardless of which `SampleRegion` supplied
the points.

---

## 1. The core integral

### 1.1 Definition (continuous)

For a field $F\in\{E,H\}$ and a sample region $V_s$:
$$I_F = \int_{V_s} |F(\mathbf r)|^2\, dV$$

This single quantity, computed for both $E$ and $H$, is everything Module 4 needs from
Module 2 — the perturbation formula (already fixed in the architecture doc) only ever
consumes $I_E$, $I_H$, and `total_stored_energy()`.

### 1.2 The Hermitian integrand

$F(\mathbf r)$ is a complex 3-vector. The correct pointwise integrand is the Hermitian norm:
$$|F(\mathbf r)|^2 = F(\mathbf r)\cdot F(\mathbf r)^* = \sum_{k\in\{x,y,z\}} F_k(\mathbf r)\,F_k(\mathbf r)^*= \sum_k |F_k(\mathbf r)|^2$$

This is real and $\ge 0$ at every point, by construction — there is no valid field
configuration for which this integrand is complex. Any implementation that produces a
complex-valued running sum here has a bug (a missing conjugate, or an accidental
`np.sum(vals**2)` instead of `np.sum(np.abs(vals)**2)` — the former sums $F_k^2$, which is
complex and physically meaningless; the latter sums $|F_k|^2$, which is what's wanted).

### 1.3 Numerical quadrature approximation (discrete)

Given points $\{\mathbf r_i\}$ and weights $\{w_i\}$ from `region.quadrature_points(n)`:
$$I_F \approx \sum_{i=1}^{M} w_i \,|F(\mathbf r_i)|^2$$

This is the entire numerical method — a weighted point sum. Nothing more sophisticated
(no adaptive mesh refinement, no spectral methods) is needed at this stage; correctness and
convergence rate are Module 3's responsibility (via how it chooses $\{\mathbf r_i, w_i\}$),
not Module 2's. Module 2's job is to *consume* the sum correctly and to *detect* when it
hasn't converged (Section 1.5).

### 1.4 Consistency identities (defensive checks, run on every call)

Two identities should hold for any valid `(FieldProvider, SampleRegion)` pair, and Module 2
should check both rather than assume Module 3 got them right:

**Volume consistency**: $\sum_i w_i = \text{Volume}(V_s)$, to within quadrature tolerance.
If this fails, the weights returned by `region.quadrature_points` are wrong, and the energy
integral built from them is meaningless regardless of how correct the field evaluation is.
Check this once per call (cheap — $O(M)$ against a number Module 3 already computes
independently via `region.volume()`) and raise rather than silently return a bad number.

**Scale invariance**: for any complex constant $c$, replacing $F\to cF$ must give
$I_F \to |c|^2 I_F$. This isn't a runtime check (it would double every call's cost for no
production benefit) — it's a **unit test**, run once against `AnalyticalField`, that
validates the whole chain: field evaluation (Module 1) → integrand (1.2) → quadrature sum
(1.3) all preserve this identity together. If Module 1's curl-residual fix from the earlier
review broke anything about field magnitude consistency, this is the test that would catch
the downstream symptom.

### 1.5 Convergence control

Since `AnalyticalField.E`/`H` are exact (Module 1 has no discretization error of its own),
the *only* error source in the default path is the quadrature sum itself. Don't fix
`n_points` at a single hardcoded value and hope — run a doubling check:

1. Evaluate $I_F$ at $n$ points and again at $2n$ points (both via
   `region.quadrature_points`).
2. If $\left|\dfrac{I_F^{(2n)} - I_F^{(n)}}{I_F^{(2n)}}\right| > \varepsilon_{\text{tol}}$
   (suggested default $10^{-4}$), double again, up to a hard cap (suggested 10 doublings,
   i.e. up to $\sim n\times 1024$ points) before giving up and raising rather than silently
   returning an unconverged value.
3. Return $I_F^{(2n)}$ (the finer estimate) once converged.

This makes `n_points` a *starting* point, not a promise — the caller shouldn't have to know
in advance how many points a given region/mode combination needs for $10^{-4}$ relative
accuracy, and small or oddly-shaped regions genuinely do need more points than large ones.

---

## 2. `FieldProvider` default path — step by step

1. Implement the Hermitian integrand (1.2) as a small standalone helper —
   `hermitian_density(field_values) -> np.ndarray` (real-valued, one number per point) —
   and unit test it directly on hand-picked complex vectors before wiring it into any
   quadrature loop.
2. Implement the doubling/convergence loop (1.5) as its own helper, parametrized by a
   callable `(n) -> I_F_estimate` — keeps the convergence *logic* independent of what's
   being integrated, so the same helper drives both the $E$ and $H$ integrals without
   duplication.
3. Implement `integrate_field_energy(region, field, n_points=2000)`:
   a. Look up the field callable (`self.E` or `self.H`) from the `field` argument.
   b. Wrap "generate points at resolution $n$, evaluate field, apply 1.2, apply 1.3" as the
      single-resolution estimator that step 2's convergence loop calls.
   c. Run the volume-consistency check (1.4) once, at the first (coarsest) point set —
      it doesn't need re-checking at every doubling, since a wrong weight-generation
      formula in Module 3 will fail it identically at any resolution.
   d. Return a **real** `float` (see Section 5 — this corrects a type mismatch in the
      original interface draft).
4. Do not implement the analytic fast-path dispatch yet — Section 4 specifies it, but the
   architecture doc already calls it a later optimization; build and validate the default
   path completely first.

---

## 3. `AnalyticalField` — thin wrapper, step by step

1. Store a `CavityMode` instance; delegate `E`, `H`, `f0`, `Q_wall`, `total_stored_energy`
   directly — no transformation, no re-scaling.
2. Do **not** override `integrate_field_energy` yet (Section 4 covers when/how). The base
   class's default quadrature path is correct and sufficient for the first working version.
3. Unit test: wrap each of the three Module 1 cavity types in turn, confirm
   `total_stored_energy()` passes through unchanged, and confirm `integrate_field_energy`
   over a region covering the *entire* cavity volume converges to a value consistent with
   `total_stored_energy()` (see Section 8 for the exact relation).

---

## 4. Optional analytic fast path (deferred — specified now so it's not re-derived later)

Per the architecture doc, this is an optimization to add only if profiling shows the
default quadrature path is too slow for Module 5's repeated-evaluation inner loop. It is
specified here, in full, so that decision can be made later without re-deriving anything —
but it should **not** be built before Sections 2–3 are complete and validated.

### 4.1 Rectangular cavity × axis-aligned box region

If the `SampleRegion` is an axis-aligned box $[x_0,x_1]\times[y_0,y_1]\times[z_0,z_1]$
inside a `RectangularCavity`, every field component is a product of one $\cos$/$\sin$ factor
per axis (Module 1, Section 1.4), so $|F|^2$ integrates to a product of three 1-D integrals,
each a **partial-interval** version of Module 1's full-interval identities:
$$\int_{x_0}^{x_1}\cos^2\!\left(\frac{k\pi x}{a}\right)dx = \frac{x_1-x_0}{2} + \frac{a}{4k\pi}\Big[\sin\!\Big(\tfrac{2k\pi x_1}{a}\Big)-\sin\!\Big(\tfrac{2k\pi x_0}{a}\Big)\Big] \quad (k\neq0)$$
$$\int_{x_0}^{x_1}\sin^2\!\left(\frac{k\pi x}{a}\right)dx = \frac{x_1-x_0}{2} - \frac{a}{4k\pi}\Big[\sin\!\Big(\tfrac{2k\pi x_1}{a}\Big)-\sin\!\Big(\tfrac{2k\pi x_0}{a}\Big)\Big] \quad (k\neq0)$$
(for $k=0$: $\cos^2\equiv1\Rightarrow$ integral $=x_1-x_0$; $\sin^2\equiv0\Rightarrow$ integral $=0$ —
same $k=0$ handling as Module 1). Setting $x_0=0,x_1=a$ recovers Module 1's full-domain
identities exactly, which is the correctness check to run the moment this is implemented
(Section 8).

### 4.2 Circular cavity × concentric annular/cylindrical region

For a `SampleRegion` that's rotationally symmetric about the cavity axis (annulus or
concentric cylinder) inside a `CylindricalCavity`, the radial integral no longer has the
simple closed form used in Module 1 (which only needed the *full* $[0,a]$ integral, evaluated
at a Bessel zero). Over a partial interval $[\rho_0,\rho_1]$, use **Lommel's integral**
(a standard closed-form indefinite integral for the square of a Bessel function):
$$\int \rho\, J_n(k_c\rho)^2\, d\rho = \frac{\rho^2}{2}\Big[J_n(k_c\rho)^2 - J_{n-1}(k_c\rho)\,J_{n+1}(k_c\rho)\Big] + C$$
evaluated at $\rho_1$ minus its value at $\rho_0$. **Consistency check**: at $\rho_1=a$ with
$k_c=X_{np}/a$ (a zero of $J_n$, so $J_n(k_c a)=0$), and using the recurrence
$J_{n-1}(x)+J_{n+1}(x)=\frac{2n}{x}J_n(x)$ (which forces $J_{n-1}(X_{np})=-J_{n+1}(X_{np})$ at
a zero of $J_n$), this reduces to $\frac{a^2}{2}J_{n+1}(X_{np})^2$ — exactly Module 1's
full-domain Bessel identity. If a future implementation of this doesn't reduce to that at
$\rho_0=0,\rho_1=a$, the fast path has a bug; this is the first thing to check.

### 4.3 Dispatch mechanism

`AnalyticalField.integrate_field_energy` would override the base class: look up
`(type(self._mode), region.shape_kind)` in a small registry mapping to a closed-form
callable; if found, call it (4.1 or 4.2's formula, evaluated component-by-component and
summed, mirroring Module 1's stored-energy pattern exactly); if not found, fall back to
`super().integrate_field_energy(...)` (the default quadrature path from Section 2). Callers
never need to know which path ran.

---

## 5. Design correction: return type of `integrate_field_energy`

The architecture doc's draft signature is `integrate_field_energy(...) -> complex`. Per
Section 1.2, the quantity being computed is manifestly real and non-negative — there's no
legitimate field configuration that produces a complex energy integral. Keeping `complex` as
the declared return type invites a class of bug where a stray non-zero imaginary part
(floating-point noise, or an actual missing-conjugate bug) silently propagates into Module 4,
which expects real inputs to its $f_{\text{calc}}, Q_{\text{calc}}$ formulas.

**Fix**: declare the return type `float`. At the end of the summation, assert
$|\mathrm{Im}(\text{raw sum})| < \varepsilon_{\text{tol}}$ (suggested $10^{-9}$ relative to
the magnitude of the real part) before returning `raw_sum.real` — this turns "silently
propagate a bug" into "loudly fail close to where the bug would actually be," which is the
whole point of catching it here rather than three modules downstream.

---

## 6. `RitzField` stub — contract only

No new equations here — the Rayleigh–Ritz coefficient-solving machinery is explicitly
deferred (per `CLAUDE.md`). The only thing Module 2 needs to fix now is that
`RitzField.E`/`H`/`total_stored_energy` will, whenever implemented, need to satisfy exactly
the same `FieldProvider` contract validated in Sections 1–3 (real energy integrals, scale
invariance, volume-consistent quadrature) — so the Section 1.4/1.5 checks should be written
generically enough to run unchanged against `RitzField` once it exists, not hardcoded to
assume an underlying `CavityMode`.

---

## 7. Step-by-step build order

1. `hermitian_density` helper (1.2) — standalone, unit tested first.
2. Convergence/doubling helper (1.5) — standalone, tested against a toy integrand with a
   known closed-form answer (e.g. integrate $1$ over a region to recover its volume) before
   it's ever pointed at a real field.
3. `FieldProvider.integrate_field_energy` default path (Section 2), including the volume
   consistency check (1.4).
4. `AnalyticalField` (Section 3), wrapping each of the three Module 1 cavity types.
5. Run the full Section 8 validation suite.
6. **Stop here** unless profiling from Module 5 shows the default path is a bottleneck —
   only then implement Section 4's fast paths, using the consistency checks in 4.1/4.2 as
   the acceptance criteria before trusting them over the (already-validated) default path.

## 8. Validation targets

- **Whole-cavity consistency**: for a region covering the entire cavity volume,
  `integrate_field_energy(region, 'E')` should equal $\dfrac{2}{\epsilon}\times$
  `total_stored_energy()` (from the master formula $W=\frac{\epsilon}{2}\int|E|^2dV$ in
  Module 1 §0.2) — this is the single strongest end-to-end check available, since it
  compares Module 2's numerical path against Module 1's independent closed form.
- **Scale invariance** (1.4): required unit test, not optional.
- **Volume-consistency check** (1.4): confirm it actually fires (raises) on a deliberately
  broken test double of `SampleRegion` that returns wrong weights, before trusting that it
  protects anything on a correct one.
- **Convergence loop termination**: confirm it converges within a small number of doublings
  for a region entirely in the smooth interior of the cavity, and confirm it raises (rather
  than looping forever or silently returning) for a pathological test case if you construct
  one (e.g. a region deliberately sized to alias against the mode's own periodicity).
- **Fast-path reduction identities** (4.1, 4.2), *if and when Section 4 is built*: confirm
  each reduces exactly to the corresponding Module 1 full-domain identity at the whole-cavity
  limit, and confirm the fast path and default quadrature path agree with each other (to
  quadrature tolerance) on at least one non-trivial partial region before trusting the fast
  path in production.
