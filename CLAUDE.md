# Cavity Perturbation Measurement Software

Design-support tool for a thesis chapter on measuring ε, μ, and loss tangent via the
cavity perturbation method. Sibling chapters (not in this repo) cover MoM, FEM, and FDTD
for other measurement problems.

## Status

Modules 1–5 are implemented (`cavity.py`, `fields.py`, `sample.py`, `perturbation.py`,
`inverse.py`) and pass their own doc's validation suite plus the end-to-end synthetic-recovery
test (`tests/test_integration.py`). Full specs live in `docs/` — read the relevant one before
touching code in that area, don't re-derive from memory, since several of them document fixes
that diverge from the literal doc prose:

- `docs/architecture_modules_1-5.md` — module boundaries, interfaces (`CavityMode`,
  `FieldProvider`, `Sample`, `PerturbationModel`, `InverseSolver`), data contracts. Note:
  `FieldProvider.integrate_field_energy` return type is corrected to `float` — see
  `docs/module2_fields_equations.md` §5, supersedes the `-> complex` shown there.
- `docs/module1_cavity_equations.md` — every equation for `cavity.py`
  (rectangular/cylindrical/coaxial), plus step-by-step build order and validation targets.
  Section 0.1's TE_z/TM_z recipe has a corrected sign relative to the original write-up —
  use the recipe as given in this file, and keep the curl-residual test (§1.8 step 2) as
  the standing guard against this class of error recurring.
- `docs/module2_fields_equations.md` — every equation for `fields.py` (quadrature
  integration contract, convergence control, deferred analytic fast paths), plus
  step-by-step build order and validation targets.
- `docs/module3_sample_equations.md` — every equation for `sample.py` (`Material`,
  `SampleRegion` geometry/quadrature, depolarization factors), plus step-by-step build order
  and validation targets. **Supersedes two things in `architecture_modules_1-5.md`**:
  `Sample.depolarization_factor` takes an additional required `field_direction` argument
  (Module 4 must evaluate and pass this in — see module3 doc §0), and `SampleRegion.shape_kind`
  is redefined to be purely geometric (`{'sphere','thin_rod','thin_disk','generic'}`), with
  all field-alignment logic moved into `depolarization_factor` itself.
- `docs/module4_perturbation_equations.md` — every equation for `perturbation.py` (filling-
  factor formulation, wall-loss combination), plus step-by-step build order and validation
  targets. **Requires retroactive additions to `cavity.py` and `fields.py`**: `CavityMode`
  and `FieldProvider` both need new abstract properties `epsilon_bg: complex`, `mu_bg: complex`
  (the absolute background permittivity/permeability each cavity was constructed with) — not
  in the original architecture doc's interface, needed because Module 3's `Material` is
  relative/dimensionless while Module 1/2 work in absolute SI units (module4 doc §0.1). Also
  simplifies `PerturbationModel.evaluate(region, material, sample=None)` to a single
  `evaluate(sample)`, and fixes an `id(region)`-keyed cache that didn't hold a strong
  reference to the cached region (module4 doc §0.3–0.4).
- `docs/module5_inverse_equations.md` — every equation for `inverse.py` (residual formulation,
  closed-form initial-guess derivation, covariance/identifiability diagnostics), plus
  step-by-step build order and validation targets. **Supersedes/fixes several things in
  `architecture_modules_1-5.md`**: `_residuals` must build a `Sample` and call Module 4's
  corrected `evaluate(sample)`, not the old `evaluate(region, material)`; `sigma_invQ` is
  renamed `sigma_Q` and both it and `sigma_f` are fractional/relative uncertainties, not
  absolute (module5 doc §0.2 — the residual formulas changed accordingly); optimizer bounds
  move from hardcoded to `InverseSolver.__init__` parameters (§0.3); the previously-stubbed
  `_closed_form_seed` is now fully derived (§2), **but §2.2–2.4's derivation is itself wrong**
  — it inverts Module 4's *literal* Delta formula, not the conjugate-corrected one
  `perturbation.py` actually implements (see the Conventions entry on the Delta conjugate,
  below); `_closed_form_seed` in `inverse.py` conjugates the solved `(eps_r-1, mu_r-1)` to
  match the real forward model, the doc does not; `FitResult` gains `covariance` and
  `condition_number` fields, computed from the Jacobian `scipy` already returns at the
  solution — no analytic Jacobian required for this (§0.5, §4). **Also requires retroactive
  additions to `perturbation.py`**: `PerturbationModel` needs public `field_provider` and
  `Rs_walls` read-only properties (not in module4's original design) — module5's closed-form
  seed (§2.1–2.2) needs direct access to the unperturbed field solution and to know whether a
  wall-loss term is even present, both bypassing `evaluate()`'s general path.

All five modules are implemented, per the build order the cross-module dependencies in each
doc implied: Module 1 (`cavity.py`) first — including the `epsilon_bg`/`mu_bg` properties
Module 4 needs — then Modules 2 (`fields.py`) and 3 (`sample.py`) in either order (3 has no
dependency on 1/2 at all), then Module 4 (`perturbation.py`, depends on 1–3), then Module 5
(`inverse.py`, depends on all four, including the retroactive `field_provider`/`Rs_walls`
properties above). The end-to-end synthetic-recovery test lives in
`tests/test_integration.py` (Testing philosophy, below).

## Tech stack

Python 3.11+, NumPy, SciPy (`special`, `optimize`, `integrate`), pytest. No other
dependencies without a good reason — this is numerically-precise scientific code, not a
web app; prefer closed-form/vectorized NumPy over adding a library.

## Commands

- Test: `pytest`
- Test one module: `pytest tests/test_cavity.py -v`
- Type check: `mypy src/`

## Repo layout

```
src/cavity_perturbation/
    numerics.py       # shared trig/Bessel identities, TE_z/TM_z recipe helper (Module 1 §0)
    cavity.py         # CavityMode + RectangularCavity, CylindricalCavity, CoaxialCavity
    fields.py         # FieldProvider, AnalyticalField, (RitzField stub, deferred)
    sample.py         # SampleRegion (Sphere/Cylinder/Slab), Material, depolarization factors
    perturbation.py   # PerturbationModel, PerturbationResult
    inverse.py        # InverseSolver, Measurement, FitResult
tests/                # one test file per module above, mirrored 1:1
docs/                 # architecture + equations references, see Status above
```

## Conventions (do not deviate without updating `docs/` too)

- **Time convention**: $e^{+j\omega t}$. Lossy media: $\epsilon=\epsilon'-j\epsilon''$,
  $\mu=\mu'-j\mu''$, both double-primed parts $\ge 0$. Complex resonance
  $\tilde\omega=\omega(1-j/2Q)$.
- **Units at every public API boundary**: Hz in, Hz out. Convert to rad/s only inside a
  function body, never across a module boundary.
- **Field normalization is arbitrary but self-consistent per instance** — never rescale to
  unit energy. Every `CavityMode`/`FieldProvider` must satisfy: scaling its raw `E`,`H` by
  any constant leaves `f0`, `Q_wall`, and any perturbation result unchanged. This is a
  required unit test on every concrete class, not just a design note.
- **Vectorization contract**: every `E(r)`, `H(r)` accepts `r` of shape `(3,)` or `(N,3)`
  and returns matching leading shape, complex dtype. No Python-level loops over points —
  quadrature elsewhere in the pipeline calls these with thousands of points at once.
- **Passivity guard**: reject/flag $\epsilon''<0$ or $\mu''<0$ at the `perturbation.py`
  evaluate boundary, not only in the inverse solver's bounds. `Material.is_passive` (module3
  doc §1.4) is the fundamental check ($\epsilon''\ge0,\mu''\ge0,\epsilon'>0,\mu'>0$) —
  distinct from Module 5's fitting bounds (e.g. $\epsilon'\ge1$), which are a prior, not a
  physical law. Don't conflate the two.
- **Energy integrals are real, not complex**: `FieldProvider.integrate_field_energy` and
  anything built on it returns `float`. Assert the imaginary part is near-zero and drop it
  explicitly (module2 doc §5) — never declare or silently propagate a `complex` energy.
- **`SampleRegion` is field-agnostic; `Sample.depolarization_factor` is not.** Geometry
  classes never take or hold a field/`FieldProvider` reference — `shape_kind` is computed
  from dimensions alone. Depolarization is the one place field direction enters Module 3,
  and it does so as an explicit argument supplied by the caller (Module 4), per module3 doc §0.
- **Absolute vs. relative permittivity — don't mix them.** `cavity.py`/`fields.py` work in
  absolute SI $\epsilon,\mu$ (exposed as `epsilon_bg`/`mu_bg`); `sample.py`'s `Material` is
  always relative/dimensionless ($\epsilon_r,\mu_r$, background normalized to 1). Any formula
  combining the two (currently only `perturbation.py`) must convert explicitly — this bug is
  invisible for an air-filled cavity and only shows up with a non-vacuum background, so the
  background-medium-sensitivity test (module4 doc §5) exists specifically to catch it; don't
  skip it because "it works for the air case."
- **Cache keys derived from object identity must hold a strong reference to what they key
  on.** `id(x)` is reused by Python after `x` is garbage-collected; a cache dict of
  `{id(x): value}` without also storing `x` itself can silently return stale data for an
  unrelated later object. See module4 doc §0.3 for the concrete instance
  (`PerturbationModel`'s shape-integral cache) — apply the same pattern anywhere else an
  `id()`-keyed cache shows up.
- **Fit uncertainties (`sigma_f`, `sigma_Q`) are fractional, not absolute.** A fixed Hz or
  fixed-$1/Q$ tolerance doesn't generalize across cavities/measurements; both are relative
  precisions (e.g. $10^{-4}$ = 100 ppm frequency precision), and the residual formulas in
  `inverse.py` divide by the measured value accordingly (module5 doc §0.2/§1.2). Fractional
  $Q$ precision and fractional $1/Q$ precision are numerically equal — no separate conversion.
- **Optimizer bounds and passivity are two different things.** `Material.is_passive`
  ($\epsilon''\ge0$ etc.) is physical law, checked in `perturbation.py`. `InverseSolver`'s
  bounds (e.g. $\epsilon'\ge1$) are a fitting prior, exposed as constructor parameters, not
  hardcoded — don't bake assumptions about "typical" materials into something that can't be
  overridden for an unusual sample.
- **Covariance/condition-number reporting doesn't require the deferred analytic Jacobian.**
  `scipy.optimize.least_squares` returns a (finite-difference) Jacobian at the solution
  regardless; `inverse.py` uses it directly for $(J^\top J)^{-1}$ and its condition number
  (module5 doc §4). This is unrelated to, and doesn't motivate building, the analytic
  Jacobian — that stays deferred (see Out of scope, below).
- **Any formula built on Module 4's Delta must use the conjugated version, not the doc's
  literal one.** `perturbation.py`'s `evaluate` computes
  `delta = -0.5*(conj(eps_r-1)*p_E + conj(mu_r-1)*p_H)` (the passivity fix — see the
  Delta-sign-error item in memory), not module4 doc §1.4's un-conjugated
  `-0.5*((eps_r-1)*p_E + (mu_r-1)*p_H)`. Module 5's `_closed_form_seed` inverts this relation
  for `eps_r`/`mu_r`, and module5 doc §2.2–2.4 derives that inversion against the doc's
  un-conjugated formula — so a literal transcription of §2.2–2.4 reproduces the same
  passivity-violating sign bug one level removed. `inverse.py`'s `_closed_form_seed` solves
  for `(conj(eps_r-1), conj(mu_r-1))` and conjugates back; don't "fix" it to match the doc text.

## Testing philosophy

**General rule**: each doc's own validation section — module1 §1.9/2.9/3.8, module2 §8,
module3 §5, module4 §5, module5 §6 — *is* the test list for that module, implemented as unit
tests, not treated as a manual sanity check. Don't mark a module "done" until its own doc's
validation section passes exactly; these are closed-form comparisons, so a mismatch is a real
bug, not noise. The handful of items below are cross-module or otherwise easy to lose track of:

- The scale-invariance and curl-residual checks (module1 §0/§1.8) are permanent regression
  tests, run for every cavity type, not one-off validations.
- The whole-cavity consistency check (module2 §8: Module 2's quadrature integral vs. Module
  1's closed-form `total_stored_energy()`) is the strongest available cross-check between
  those two modules — treat a mismatch as a real bug in one of them, not quadrature noise.
- Module 2's analytic fast paths (§4) stay unbuilt until Module 5 profiling justifies them.
- The frame-transform round-trip test and `contains`/quadrature-point agreement check
  (module3 §5) are the two most likely places for a silent geometry bug — run both for every
  `SampleRegion` subclass.
- The background-medium-sensitivity test (module4 §5) is the one test that catches the
  absolute-vs-relative-permittivity bug class (Conventions, above) — an air-filled cavity
  alone won't expose it, so don't skip this test because the air case passes.
- The degenerate-multi-mode / identifiability test (module5 §6) exercises the condition-number
  diagnostic actually firing, not just being computed — a fit that silently converges despite
  poorly-constrained parameters is the failure mode this guards against.
- The end-to-end synthetic-data recovery test (fit known ε, μ from model-generated `f_meas`,
  `Q_meas`) is the regression guard for the whole pipeline now that Modules 1–5 are all
  spec-complete — lives in `tests/test_integration.py`, re-run it after any change to any
  module.

## Explicitly out of scope for this repo (deferred / elsewhere)

- Analytic Jacobian for the inverse solver (currently finite-difference via
  `scipy.optimize.least_squares`) — deferred, contract fixed so it drops in later without
  touching `inverse.py`'s callers. Reporting covariance/condition-number (module5 doc §4)
  does *not* require this — it reuses the finite-difference Jacobian `scipy` already returns.
- `RitzField` full implementation (Rayleigh–Ritz sample-size correction) — interface
  stubbed in `fields.py`, deferred to a later phase.
- Sensitivity maps, validation/sample-size-correction study modules — not yet designed.
- MoM, FEM, FDTD — separate thesis chapters, separate repos.
