# Cavity Perturbation Measurement Software

Design-support tool for a thesis chapter on measuring ε, μ, and loss tangent via the
cavity perturbation method. Sibling chapters (not in this repo) cover MoM and FEM for
other measurement problems. FDTD (`fdtd/`) lives in this repo, as an independent
time-domain cross-validation method for this same cavity-perturbation problem, not a
separate measurement problem.

## Status

Modules 1–5 are implemented (`cavity.py`, `fields.py`, `sample.py`, `perturbation.py`,
`inverse.py`) and pass their own doc's validation suite plus the end-to-end synthetic-recovery
test (`tests/test_integration.py`). The Rayleigh–Ritz sample-size-correction module
(`ritz.py`, per `docs/ritz_module_plan.md`) is also implemented and passes its own Section 7
test plan (`tests/test_ritz.py`). The FDTD module (`fdtd/`, per `docs/fdtd_module_plan.md`)
is also implemented and passes its own Section 7 test plan (`tests/test_fdtd/`).

Full specs live in `docs/`. **Read the relevant doc before touching code in that area — several
diverge from their own original draft prose**, corrected during implementation or during a
later cross-check against the actual code. Don't re-derive from memory; the doc corrections
below (and the Conventions section) are the current source of truth, not the first draft of
each doc:

- `docs/architecture_modules_1-5.md` — module boundaries and interfaces. Superseded in
  several places by the module docs below; treat it as the original sketch, not current truth.
- `docs/module1_cavity_equations.md` — `cavity.py`. The abstract §0.1 TE_z/TM_z recipe and
  the cylindrical §2.4 field formulas both had a sign error (traced to the same recipe
  transcription, fixed together); the coaxial §3.4 stored-energy formula had $\ln(b/a)$
  inverted (should diverge, not vanish, as the conductors merge). All three are corrected in
  the doc now and guarded by `test_curl_residual`/`test_cyl_curl_residual` and a brute-force
  quadrature check on coaxial energy.
- `docs/module2_fields_equations.md` — `fields.py`. Matches the implementation closely;
  no corrections needed.
- `docs/module3_sample_equations.md` — `sample.py`. §2.3's oblique-angle test needed a step
  the original draft omitted: extracting a real spatial direction from `field_direction`
  before the angle test, since a Module 1 mode's field is generally complex with one shared
  phase across all Cartesian components (naively taking `.real` can silently zero out a
  legitimately real mode shape). A numerically-zero field direction (e.g. a sample on a
  symmetry axis where the relevant field vanishes) is a real degeneracy, not an error — falls
  through to the same `'generic'` handling as a genuinely oblique angle.
- `docs/module4_perturbation_equations.md` — `perturbation.py`. §1.4's $\Delta$ formula was
  missing a conjugate on the bare material contrast — see Conventions, below, for the
  standing rule this produced. Also needs `PerturbationModel.field_provider`/`.Rs_walls` as
  public read-only properties (not in the original interface), which Module 5's closed-form
  seed calls directly.
- `docs/module5_inverse_equations.md` — `inverse.py`. §2.3–2.4's closed-form seed inverted
  the *un*-conjugated $\Delta$ formula; `_closed_form_seed` must invert whatever
  `perturbation.py` actually computes (now conjugate-corrected), not the first draft's
  formula. Also: clip the seed into the fitting bounds before calling `least_squares`, which
  requires $lb\le x_0\le ub$ exactly.
- `docs/ritz_module_plan.md` — `ritz.py`. Retired the `RitzField` stub that used to sit in
  `fields.py` (a `PerturbationModel`-shaped `RitzCorrectedModel` sibling class replaced it, not
  the originally-sketched `FieldProvider`). Its own energy-matrix derivation needed the same
  two classes of fix found in Modules 1 and 4 (a factor-of-2 caught by cross-checking Module
  1's `total_stored_energy()` convention, and the same missing conjugate as Module 4) — both
  corrected in the doc. Two more found actually implementing it: (1) §3.2's mode-tracking
  weight, read literally ("$|A_k^{(1)}|$ after normalizing each eigenvector"), is not
  basis-scale-invariant — different basis modes (e.g. a cube's exactly-degenerate $TE_{111}$
  vs $TM_{111}$) can differ in `total_stored_energy()` by many orders of magnitude at the same
  nominal amplitude, which silently corrupts mode tracking unless each component is rescaled
  by its own $\sqrt{M_{ii}}$ first (now in the doc, §3.2); (2) §7.2/§7.4's claim that
  `RitzCorrectedModel` converges toward `PerturbationModel`'s depolarization-corrected answer
  (for a `Sphere`, not a `'generic'`-shaped region) as $N$ grows is **not observed** — swept to
  $N=40$, the gap stays flat/small rather than shrinking (doc corrected to test only what's
  actually verified: nonzero at $N=1$, bounded at larger $N$). See
  `docs/ritz_module_plan.md`'s own inline corrections and the Conventions entry below for why
  this doesn't undermine the module's correctness (both are Ritz-vs-*a-different-approximation*
  comparison issues, not violations of a physical invariant — every physical-invariant check,
  passivity/Q-degradation, lossless-Hermitian, scale-invariance, N=1-exact-reduction, passes).
- `docs/fdtd_module_plan.md` — `fdtd/`. Independent time-domain (Yee-grid leapfrog + ringdown
  extraction) cross-check of Modules 1–4, not built on any of their machinery except
  `contains()` (Section 0.3) and `FieldProvider`/`Material` as read-only inputs. Corrected in
  place, found actually implementing/exercising it (not from the original design prose):
  (1) §6.1's suggested full-length Hann window *overestimated* Q by 30–40% on a known-Q
  synthetic ringdown, because a ringdown is already a physically decaying transient (unlike
  the steady-state signal Hann is meant for) — a full raised-cosine taper reshapes the decay
  itself rather than just smoothing the record boundary; fixed with a mild `('tukey', 0.2)`
  taper. The independent time-domain (Hilbert-envelope) extraction route has no such tension
  and recovered the same synthetic case to a few parts in $10^{-6}$. (2) §6.3's rough-Q guess
  for sizing the record length — "Module 1's `Q_wall` or a coarse pre-run" — needed both
  halves fixed: `Q_wall` alone misses a separately lossy sample entirely, and the sample-only
  fallback originally used ($1/\tan\delta$, reasoned as "conservative") is actually *not*
  conservative — for a small filling factor the true $Q$ can be tens of times $1/\tan\delta$,
  so the record ended tens of times too early (caught directly: ~97% low $Q$ on a small
  low-loss sample). Fixed by reusing `PerturbationModel` itself as the "coarse pre-run,"
  cheap and already filling-factor-aware. (3) A genuinely loss-free run (no wall loss, no
  sample loss) has unbounded true $Q$, so "several $\tau$" has no finite target — an
  intermediate fallback expressing this as an assumed large $Q$ produced multi-million-step
  runs at GHz frequencies; fixed by expressing the fallback directly as a fixed oscillation-
  period count instead. (4) The small-sample cross-check against Module 4 (§7.5) needs the
  sample *itself* resolved by several grid cells — independent of how finely
  `cells_per_wavelength` resolves the cavity mode — or rasterization error dominates (a
  1.5mm-radius sample at a ~3mm cell size gave ~196% Q error from this alone). See
  `docs/fdtd_module_plan.md`'s own inline corrections (§6.1, §6.3, §7.4, §7.5) for the full
  detail.

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
    fields.py         # FieldProvider, AnalyticalField, integrate_field_cross_overlap
    sample.py         # SampleRegion (Sphere/Cylinder/Slab), Material, depolarization factors
    perturbation.py   # PerturbationModel, PerturbationResult, omega_tilde_to_result
    inverse.py        # InverseSolver, Measurement, FitResult
    ritz.py           # RitzCorrectedModel, nearest_basis_modes, converged_ritz_model
    fdtd/             # FDTDModel: PerturbationModel-shaped sibling, time-domain cross-check
        grid/         # yee.py (staggering), rasterize.py (contains()-based material masks)
        stability.py  # CFL dt computation + enforcement
        extract.py    # scipy.fft/scipy.signal ringdown -> (f_r, Q)
        materials.py  # single-frequency conductivity match, E-update coefficient arrays
        source.py     # mode-shaped soft-source excitation, probe placement
        stepper.py    # leapfrog E/H update loop
        model.py      # FDTDModel itself, orchestrates the above
tests/                # one test file per module above, mirrored 1:1, + shared conftest.py
                       # (tests/test_fdtd/ mirrors fdtd/ the same way)
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
  any constant leaves `f0`, `Q_wall`, and any perturbation result unchanged. Required unit
  test on every concrete class, not just a design note.
- **Vectorization contract**: every `E(r)`, `H(r)` accepts `r` of shape `(3,)` or `(N,3)`
  and returns matching leading shape, complex dtype. No Python-level loops over points.
- **Passivity guard**: reject/flag $\epsilon''<0$ or $\mu''<0$ at the `perturbation.py`
  evaluate boundary, not only in the inverse solver's bounds. `Material.is_passive`
  ($\epsilon''\ge0,\mu''\ge0,\epsilon'>0,\mu'>0$) is the fundamental check — distinct from
  Module 5's fitting bounds (e.g. $\epsilon'\ge1$), which are a prior, not a physical law.
- **Energy integrals are real, not complex**: `FieldProvider.integrate_field_energy` and
  anything built on it returns `float`. Assert the imaginary part is near-zero and drop it
  explicitly — never declare or silently propagate a `complex` energy.
- **`SampleRegion` is field-agnostic; `Sample.depolarization_factor` is not.** Geometry
  classes never take or hold a field/`FieldProvider` reference — `shape_kind` is computed
  from dimensions alone. Depolarization is the one place field direction enters, and it does
  so as an explicit argument supplied by the caller (Module 4) — which must itself extract a
  real direction from a generally-complex field vector before using it (module3 doc §2.3).
- **Absolute vs. relative permittivity — don't mix them.** `cavity.py`/`fields.py` work in
  absolute SI $\epsilon,\mu$ (exposed as `epsilon_bg`/`mu_bg`); `sample.py`'s `Material` is
  always relative/dimensionless. Any formula combining the two must convert explicitly — this
  bug is invisible for an air-filled cavity, so the background-medium-sensitivity test exists
  specifically to catch it.
- **Cache keys derived from object identity must hold a strong reference to what they key
  on.** `id(x)` is reused by Python after `x` is garbage-collected; `{id(x): value}` without
  also storing `x` can silently return stale data for an unrelated later object
  (`PerturbationModel`'s shape-integral cache is the concrete instance).
- **Fit uncertainties (`sigma_f`, `sigma_Q`) are fractional, not absolute.** Both are
  relative precisions (e.g. $10^{-4}$ = 100 ppm), and the residual formulas divide by the
  measured value accordingly. Fractional $Q$ precision and fractional $1/Q$ precision are
  numerically equal — no separate conversion.
- **Optimizer bounds and passivity are two different things** — see Passivity guard, above.
  Bounds are constructor parameters, not hardcoded, for exactly this reason.
- **Covariance/condition-number reporting doesn't require the deferred analytic Jacobian.**
  `scipy.optimize.least_squares` returns a Jacobian at the solution regardless; `inverse.py`
  uses it directly. Unrelated to, and doesn't motivate building, the analytic Jacobian.
- **Module 4's $\Delta$ is conjugated on the bare material contrast, and this propagates.**
  `perturbation.py` computes `delta = -0.5*(conj(eps_r-1)*p_E + conj(mu_r-1)*p_H)`, not the
  un-conjugated form — a low-loss passive material violates $Q_{\text{calc}}\le Q_{\text{wall}}$
  otherwise (verified directly, not just asserted; see `test_passive_material_never_improves_q`).
  Only the bare $(\epsilon_r-1)$/$(\mu_r-1)$ factors are conjugated — $p_E,p_H$ (and the
  $\kappa_E,\kappa_H$ inside them) are not. **Anything built on top of this formula inherits
  the same requirement**: Module 5's closed-form seed inverts the conjugated relation and
  conjugates its solution back; `ritz.py`'s sample-correction matrix ($\Delta M$) carries
  the identical conjugate for the identical reason. If a future derivation reduces to the
  un-conjugated form in some limit, that's the signal something upstream regressed, not a
  sign that the un-conjugated form was fine after all.

## Testing philosophy

**General rule**: each doc's own validation section is the test list for that module,
implemented as unit tests. Don't mark a module "done" until its own doc's validation section
passes exactly — these are closed-form comparisons, so a mismatch is a real bug, not noise.
Cross-module items worth not losing track of:

- Scale-invariance and curl-residual checks (module1 doc) are permanent regression tests for
  every cavity type, not one-off validations.
- The whole-cavity consistency check (module2 doc: Module 2's quadrature integral vs. Module
  1's closed-form `total_stored_energy()`) is the strongest available cross-check between
  those two modules.
- Module 2's analytic fast paths stay unbuilt until Module 5 profiling justifies them.
- The frame-transform round-trip test and `contains`/quadrature-point agreement check
  (module3 doc) are the two most likely places for a silent geometry bug.
- The background-medium-sensitivity test (module4 doc) is the one test that catches the
  absolute-vs-relative-permittivity bug class — don't skip it because the air case passes.
- `assert_passive_material_never_improves_q` (`tests/conftest.py`, shared by
  `test_perturbation.py` and `test_ritz.py`) is the check that actually caught the
  $\Delta$-conjugate bug in Module 4 and would catch a regression of the identical fix in
  `ritz.py`'s $\Delta M$ — run it immediately after touching either module's material-contrast
  formula, not just at the end.
- The degenerate-multi-mode / identifiability test (module5 doc) exercises the
  condition-number diagnostic actually firing, not just being computed.
- The end-to-end synthetic-data recovery test (`tests/test_integration.py`) is the regression
  guard for the whole pipeline — re-run it after any change to any module.
- `ritz.py`'s own N=1-exact-reduction test (`test_n1_reduction_matches_point_dipole_formula_exactly`)
  isolates a matrix-assembly/conjugate bug from an eigensolve or mode-tracking bug, the same
  role the small-sample-limit test plays for Module 4 — cheaper to debug than the full
  multi-mode comparison, run it first after touching `RitzCorrectedModel`.
- `fdtd/extract.py`'s synthetic-signal tests (`tests/test_fdtd/test_extract.py`, §7.3) are
  built and pass *before* any Maxwell time step exists, same rationale as `ritz.py`'s N=1
  test: isolate the signal-processing from the physics so a bug in one doesn't masquerade as
  a bug in the other.
- `fdtd/test_small_sample.py` (§7.5) needs the sample resolved by several grid cells, a
  *different* resolution requirement from `cells_per_wavelength` (which only governs the
  cavity-mode wavelength) — an under-resolved sample dominates the error long before the
  cavity-mode resolution does; don't shrink the test sample below a few cells across when
  tuning this test's runtime.
- `fdtd/model.py`'s record-length sizing (`_record_duration`) directly reuses
  `PerturbationModel` as its own rough-Q pre-estimate — if `PerturbationModel`'s formula ever
  regresses (e.g. the Module 4 $\Delta$-conjugate bug recurring), `FDTDModel` runs would
  silently mis-size their record length too, a coupling worth remembering when debugging
  either module.

## Explicitly out of scope for this repo (deferred / elsewhere)

- Analytic Jacobian for the inverse solver (currently finite-difference via
  `scipy.optimize.least_squares`) — deferred, contract fixed so it drops in later without
  touching `inverse.py`'s callers. Covariance/condition-number reporting does not require this.
- Adaptive frequency-window basis selection for `ritz.py` (currently a fixed-`n_basis`,
  nearest-frequency default) — documented as a future refinement in `docs/ritz_module_plan.md`
  §1, not a first-pass requirement.
- A general irregular-geometry Ritz/FEM solver — `docs/ritz_module_plan.md` §0.4 explicitly
  scopes this out (irregular geometries are the FEM thesis chapter's job); `RitzCorrectedModel`
  only ever mixes exact modes of one of the three canonical (rectangular/cylindrical/coaxial)
  cavities.
- Sensitivity maps, broader validation studies beyond the Ritz sample-size-correction sweep
  (`docs/ritz_module_plan.md` §7.3) — not yet designed.
- Full dispersive (Debye/Lorentz) material modeling in `fdtd/materials.py` — the single-
  frequency-matched conductivity (`docs/fdtd_module_plan.md` §4.3) is a deliberate first-pass
  scope boundary, not an oversight.
- Sub-cell/conformal boundary correction for `fdtd/`'s staircased curved-cavity boundaries,
  and PML/adaptive grid-sizing-from-sample-extent (`docs/fdtd_module_plan.md` §7.5's note) —
  documented future refinements, not first-pass requirements.
- A general irregular-geometry FDTD grid — `fdtd/` only ever rasterizes onto a regular
  Cartesian grid via `contains()` (Section 0.3), the same canonical-cavity-only scope as Ritz.
- MoM, FEM — separate thesis chapters, separate repos.
