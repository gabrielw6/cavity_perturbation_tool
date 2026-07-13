# Module 4 — Perturbation: Forward Model — Equations & Implementation Plan

Scope: `PerturbationModel`, `PerturbationResult`, per the interface in
`architecture_modules_1-5.md`. No code — equations and build order only.

**Read Section 0 first.** Working through the equations carefully surfaced three real
issues in the architecture doc's sketch — a units mismatch between Modules 1–2 (absolute,
SI) and Module 3 (relative, dimensionless), a latent cache-key bug, and an awkward
double signature — all fixed here before the rest of the doc is written against them.

---

## 0. Design corrections up front

### 0.1 The units seam: this module is where absolute and relative permittivity meet

Module 1's `CavityMode` computes $f_0$, $Q_{\text{wall}}$, and `total_stored_energy()` using
the cavity's **absolute** background fill medium ($\epsilon_{bg},\mu_{bg}$, SI units — vacuum
for an air-filled cavity, but not assumed to be exactly $\epsilon_0$ in general). Module 3's
`Material` stores **relative**, dimensionless $\epsilon_r,\mu_r$ (background normalized to 1 —
see `module3_sample_equations.md` §1.1/1.3). The architecture doc's draft formula,
$(\epsilon-\epsilon_0)$, silently mixes these — it reads as an absolute difference, but
`Material.eps` is relative. For an air-filled cavity this bug is numerically almost invisible
(since $\epsilon_{bg}\approx\epsilon_0$ and the ratio is close to 1), which is exactly what
makes it dangerous — it wouldn't show up until someone uses a non-vacuum background, and by
then it's buried three modules deep.

**Fix**: Section 1 below is written entirely in terms of dimensionless filling factors, so the
$\epsilon_{bg},\mu_{bg}$ conversion happens explicitly, once, rather than being silently
assumed away. This requires **two new properties on `FieldProvider` (and `CavityMode`)**:
`epsilon_bg: complex` and `mu_bg: complex` — the absolute background values each concrete
cavity class was constructed with. This is a retroactive addition to Modules 1 and 2's
interfaces (their docs should be updated to include these as abstract properties, with each
concrete `CavityMode` returning whatever it was constructed with, defaulting to vacuum
$\epsilon_0,\mu_0$).

### 0.2 `depolarization_factor`'s `field_direction` — resolved internally, no new external parameter

Module 3 fixed `Sample.depolarization_factor(field_type, field_direction)` to need a field
direction, with the note that "Module 4 is responsible for evaluating and passing this in."
Concretely: every `SampleRegion` subclass has a `.center`, and `PerturbationModel` already
holds `self._fp` (the `FieldProvider`). So `evaluate()` computes
`self._fp.E(region.center)` / `self._fp.H(region.center)` itself, internally, and passes the
result straight into `depolarization_factor` — this does **not** require adding a new
parameter to `PerturbationModel.evaluate()`'s own signature; it's an internal implementation
step, invisible to Module 5.

### 0.3 Cache-key bug: `id(region)` is not a safe dictionary key on its own

Python reuses `id()` values after an object is garbage-collected. The architecture doc's
sketch caches `{id(region): (IE, IH)}` without holding a reference to `region` itself — if a
`PerturbationModel` outlives one `region` object and a *different* region later gets allocated
at the same freed address, the cache would silently return stale shape integrals for the
wrong geometry. This is a real risk specifically in long-lived sessions (e.g. exploratory
work) where a `PerturbationModel` is reused across many different `Measurement`/`Sample`
objects — less of a risk within a single Module 5 fit (which holds one region alive
throughout), but not something to leave latent.

**Fix**: cache `{id(region): (region, IE, IH)}` — storing the region object itself in the
cache entry holds a strong reference to it, which guarantees its `id()` cannot be reused for
as long as the cache entry exists. Cheap, and removes the bug entirely.

### 0.4 Simplified `evaluate()` signature

The sketch's `evaluate(region, material, sample=None)` leaves it ambiguous which argument
wins if both are supplied. Simplify to a single required parameter:
`evaluate(sample: Sample) -> PerturbationResult`. Module 5 constructs a fresh (cheap, frozen)
`Sample(region=fixed_region, material=trial_material)` on every fit iteration; the internal
cache still keys off `sample.region` (0.3), so the performance benefit described in the
architecture doc is unaffected — only the external signature gets cleaner.

---

## 1. Filling-factor formulation (the forward model itself)

### 1.1 Starting point

The general first-order cavity-perturbation result (the same variational stationarity
argument this whole project starts from — only *explicit* material/boundary changes survive
to first order, never $\delta E$):

$$\frac{\tilde\omega-\omega_0}{\omega_0} = -\frac{\displaystyle\int_{V_s}\big[\Delta\epsilon\, E\cdot E_0^* + \Delta\mu\, H\cdot H_0^*\big]dV}{\displaystyle\int_V\big[\epsilon_{bg}|E_0|^2+\mu_{bg}|H_0|^2\big]dV}$$

where $E,H$ are the *actual* (perturbed) fields inside the sample and $E_0,H_0$ are the
unperturbed fields (`FieldProvider.E`/`H`). The denominator is, by Module 1's master energy
formula (§0.2 there), exactly $2W$.

### 1.2 Quasi-static substitution

The unknown internal field is what Module 3's depolarization factor exists to approximate:
$E \approx \kappa_E E_0$, $H\approx\kappa_H H_0$ (both a single complex scalar, uniform over
the sample — consistent with the same small-sample assumption already built into using this
formula at all). Substituting, and writing $\Delta\epsilon=\epsilon_{bg}(\epsilon_r-1)$
(the units-seam conversion from 0.1):

$$\int_{V_s}\Delta\epsilon\,E\cdot E_0^*\,dV \approx \epsilon_{bg}(\epsilon_r-1)\,\kappa_E\int_{V_s}|E_0|^2\,dV = \epsilon_{bg}(\epsilon_r-1)\,\kappa_E\,I_E$$

where $I_E=\int_{V_s}|E_0|^2\,dV$ is exactly `integrate_field_energy(region,'E')` from
Module 2 — material-independent, the quantity worth caching (0.3/0.4). Same structure for
$H$ with $I_H,\kappa_H,\mu_r$.

**Important**: $\kappa_E$ enters here to the *first power* (it approximates the field $E$
itself, not an energy), so it is not squared anywhere in this formula — a natural place to
introduce an error would be assuming an "energy correction" needs $|\kappa_E|^2$; it doesn't.

### 1.3 Dimensionless filling factors

Define:
$$p_E \equiv \frac{\epsilon_{bg}\,\kappa_E\,I_E}{W}, \qquad p_H \equiv \frac{\mu_{bg}\,\kappa_H\,I_H}{W}$$

Both are complex, dimensionless, and — because $I_E, I_H, W$ all come from the same
`FieldProvider` instance at the same arbitrary field scale — automatically satisfy the
project-wide scale-invariance requirement (Module 0) without any extra normalization step.
$\kappa_E,\kappa_H$ are themselves generally complex (Module 3 §2.1's depolarization formula
evaluated at complex $\epsilon_r,\mu_r$), so $p_E,p_H$ carry loss information even before
being multiplied by the material contrast.

### 1.4 Complex frequency-shift formula

$$\Delta \equiv \frac{\tilde\omega_{\text{sample-only}}-\omega_0}{\omega_0} = -\frac{1}{2}\Big[(\epsilon_r-1)\,p_E + (\mu_r-1)\,p_H\Big]$$

Compute this as a single complex expression directly (complex $\epsilon_r,\mu_r,p_E,p_H$
throughout) — do not attempt to hand-expand into separate real/imaginary sub-formulas before
implementing; that expansion is exactly the kind of manual-algebra step that's introduced
errors elsewhere in this project. Take $\mathrm{Re}$/$\mathrm{Im}$ only at the very last step
(Section 2.3).

---

## 2. Combining with wall loss

### 2.1 Why the two loss mechanisms simply add

Wall loss and sample loss are two independent, small, first-order perturbations to the same
ideal lossless cavity — and first-order perturbation theory is linear in independent small
perturbations to the same base state, by construction (this is the same reasoning
underlying every result in this project: to first order, effects of distinct explicit changes
superpose). Module 1's `Q_wall(Rs)` is itself the statement of one such perturbation, applied
on its own: $\tilde\omega_{\text{wall-only}} = \omega_0\big(1-\tfrac{j}{2Q_{\text{wall}}}\big)$.
Adding the sample's own perturbation $\Delta$ (1.4) to this, rather than multiplying or
substituting, is the physically correct combination:

$$\tilde\omega_{\text{loaded}} = \omega_0\left(1 - \frac{j}{2Q_{\text{wall}}} + \Delta\right)$$

If `Rs_walls is None` (`PerturbationModel` constructed with no wall-loss model), drop the
$-j/(2Q_{\text{wall}})$ term entirely — equivalent to $Q_{\text{wall}}\to\infty$.

### 2.2 Extracting $f_{\text{calc}}$, $Q_{\text{calc}}$

From the project-wide convention $\tilde\omega=\omega_r(1-j/(2Q))$, solved for a general
complex $\tilde\omega_{\text{loaded}}$:

$$f_{\text{calc}} = \frac{\mathrm{Re}(\tilde\omega_{\text{loaded}})}{2\pi}, \qquad Q_{\text{calc}} = -\frac{\mathrm{Re}(\tilde\omega_{\text{loaded}})}{2\,\mathrm{Im}(\tilde\omega_{\text{loaded}})}$$

### 2.3 Edge cases

- $\mathrm{Im}(\tilde\omega_{\text{loaded}}) = 0$ exactly (both `Rs_walls is None` and a
  lossless material, $\epsilon_r'',\mu_r''=0$): return $Q_{\text{calc}}=\texttt{float('inf')}$
  explicitly — don't let this fall through to a NaN from division by zero.
- Passivity sanity check worth building in (see Section 5): for any passive material
  ($\epsilon_r''\ge0,\mu_r''\ge0$ — Module 3 §1.4), $\mathrm{Im}(\Delta)$ must be $\le0$ — a
  lossy sample can only degrade $Q$, never improve it. If this is ever violated in testing, the
  bug is in the sign chain from Module 1's field recipe (already reviewed once) through this
  formula, not in the test.

---

## 3. Caching strategy

Per 0.3's fix, cache `{id(region): (region, I_E, I_H)}` inside `PerturbationModel`. On every
`evaluate(sample)` call:
1. Look up `sample.region` in the cache by `id`; compute and store `(region, I_E, I_H)` via
   Module 2's `integrate_field_energy` (both 'E' and 'H') only on a cache miss.
2. Always recompute $\kappa_E,\kappa_H$ fresh (material-dependent — these are cheap, no
   quadrature involved, just the Section 2.1-of-Module-3 closed-form evaluated at the current
   trial material).
3. Assemble $p_E,p_H$ (1.3), $\Delta$ (1.4), $\tilde\omega_{\text{loaded}}$ (2.1),
   $f_{\text{calc}},Q_{\text{calc}}$ (2.2), return a `PerturbationResult`.

---

## 4. Step-by-step implementation instructions

1. Add `epsilon_bg`, `mu_bg` abstract properties to `FieldProvider` (and `CavityMode`) —
   this touches `cavity.py` and `fields.py`, not just `perturbation.py`; do this first so
   Module 4 isn't blocked on a stub.
2. Implement `PerturbationModel.__init__(field_provider, Rs_walls=None)` with the corrected
   cache dict from Section 3.
3. Implement the internal `_shape_integrals(region)` helper (cache lookup/populate, per
   Section 3, step 1).
4. Implement the internal field-direction evaluation from 0.2: given `region.center`, call
   `self._fp.E(region.center)` and `self._fp.H(region.center)` once per `evaluate()` call.
5. Implement `evaluate(sample)`:
   a. Get `I_E, I_H` via step 3.
   b. Get $\kappa_E,\kappa_H$ via `sample.depolarization_factor('E', field_dir_E)` and
      `('H', field_dir_H)` from step 4.
   c. Assemble $p_E,p_H$ (1.3) using `self._fp.epsilon_bg`, `self._fp.mu_bg`,
      `self._fp.total_stored_energy()`.
   d. Assemble $\Delta$ (1.4) directly as a complex expression.
   e. Assemble $\tilde\omega_{\text{loaded}}$ (2.1), including the `Rs_walls is None` branch.
   f. Extract $f_{\text{calc}}, Q_{\text{calc}}$ (2.2), with the zero-imaginary-part guard
      (2.3).
   g. **Passivity guard** (per `CLAUDE.md`): check `sample.material.is_passive` at this
      boundary before returning a result; raise or flag rather than silently returning a
      negative or otherwise unphysical $Q_{\text{calc}}$.
   h. Return `PerturbationResult(f_calc, Q_calc, omega_tilde=tilde_omega_loaded)`.
6. Run the Section 5 validation suite.

---

## 5. Validation targets

- **Small-sample limit**: shrink a `Sphere` region toward a point (fixed material) and
  confirm `evaluate()` converges to the closed-form point-dipole result — i.e. $p_E \to
  \dfrac{\epsilon_{bg}\cdot\frac{3}{\epsilon_r+2}\cdot|E_0(\text{center})|^2\cdot V_s}{W}$ as
  $V_s\to0$, matching the classical small-sphere perturbation formula directly.
- **Passivity ⇒ $Q$ can only degrade**: for a sweep of passive materials
  ($\epsilon_r''\ge0$), confirm $\mathrm{Im}(\Delta)\le0$ always, i.e. $Q_{\text{calc}} \le
  Q_{\text{wall}}$ whenever `Rs_walls` is set. Treat any violation as a sign-chain bug, not
  numerical noise.
- **Reciprocal-$Q$ additivity**: compute $Q_{\text{calc}}$ two ways — (a) directly via the
  combined formula (Section 2.1), (b) by separately computing $Q_{\text{wall}}$ and a
  sample-only $Q_{\text{sample}}$ (Section 1.4's $\Delta$ alone, no wall term) and forming
  $1/Q_{\text{wall}}+1/Q_{\text{sample}}$ — confirm they agree.
- **Background-medium sensitivity (catches the 0.1 bug class specifically)**: construct two
  `FieldProvider`s differing only in `epsilon_bg` (e.g. vacuum vs. a synthetic
  $\epsilon_{bg}=2\epsilon_0$ background), same sample and material, and confirm
  $\Delta$ scales as expected through the $\epsilon_{bg}$ factor in $p_E$ — this is the test
  that would have caught the original architecture doc's mixed-units bug, since an air-filled
  cavity alone wouldn't expose it.
- **Scale invariance**: multiply the underlying field's arbitrary amplitude by a constant;
  confirm $f_{\text{calc}},Q_{\text{calc}}$ are unchanged (should follow automatically from
  1.3, but verify — this is a required regression test on every `FieldProvider`, not optional).
- **Cache correctness**: construct two distinct `Sample`/`region` objects in a way that
  forces the first to be garbage-collected before the second is created (a deliberate stress
  test for 0.3's fix), and confirm the cache never returns the first region's shape integrals
  for the second.
