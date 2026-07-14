# FDTD Module — Architecture Definition

Scope: a new sub-package, `fdtd/`, providing an `FDTDModel` that predicts $(f, Q)$ for a
cavity + sample by time-domain simulation and ringdown analysis, as a `PerturbationModel`-shaped
sibling. No code — contracts, file breakdown, and test plan only.

**Read Section 0 first.** FDTD is structurally different from every other method in this
project: it is an *initial-value* time-stepping problem, not an eigenvalue solve, and $f$/$Q$
come from signal processing of a recorded time series rather than from an algebraic solution.
That difference shapes the whole module.

---

## 0. Design decisions up front

### 0.1 Interface compatibility — a `PerturbationModel` sibling, verified against the real contract

`FDTDModel` must be a drop-in for `PerturbationModel` wherever `Measurement.model` (Module 5)
holds one. Checked against the actual current code, that means matching three things exactly,
not just the `evaluate` name:

- `evaluate(sample: Sample) -> PerturbationResult`, where `PerturbationResult` is the existing
  frozen dataclass `(f_calc: float, Q_calc: float, omega_tilde: complex)` — reused verbatim,
  not re-declared.
- a `field_provider` property returning a `FieldProvider`.
- an `Rs_walls` property returning `float | None`.

The last two exist because Module 5's closed-form seed calls them directly, bypassing
`evaluate()` — matching `evaluate`'s signature alone is insufficient, the same lesson already
recorded for the Ritz plan. `FDTDModel.field_provider` returns the `AnalyticalField` wrapping
the mode of interest (used for excitation and probing, Section 3); `FDTDModel.Rs_walls`
returns the wall-loss setting it was constructed with.

### 0.2 The Cartesian grid is its own module, with no EM knowledge

Per the atomization requirement, grid construction and the "which material is at cell
$(i,j,k)$" rasterization are a standalone `grid/` sub-package that knows nothing about
Maxwell's equations, time-stepping, or resonance — only geometry and staggering. It answers
two questions: where does each field component live in space (Yee staggering, Section 1), and
what material fills each cell (rasterization, Section 2). It is separately testable without
any field ever being stepped (Section 7 verifies the grid module in complete isolation from
the solver).

### 0.3 Geometry comes from `contains()`, not from a mesh

Unlike the meshing module, FDTD needs no external geometry/CAD library. A regular Cartesian
grid represents any shape by testing cell membership, which is exactly what Module 1's
`CavityMode.contains()` and Module 3's `SampleRegion.contains()` already provide. This module
*consumes* those two methods (reads their boolean output) and adds no geometry capability of
its own. Curved boundaries are therefore staircased — a real, method-specific error source
(Section 6.4), controlled by grid-refinement convergence, not eliminated.

### 0.4 Lossy sample via single-frequency-matched conductivity

Per the decision to use the single-frequency match: the sample is modeled as a real
$\epsilon_r'$ plus a real conductivity $\sigma$ chosen so the loss matches the specified
$\tan\delta_e$ at the mode frequency $f_0$ (Section 4). This is consistent with how the rest
of the project treats $\epsilon_r$ — one complex value at one frequency — rather than an extra
approximation. Full dispersive (Debye/Lorentz) modeling is explicitly deferred (Section 4.3).

### 0.5 Passivity is structural here, not a guarded invariant

Because loss enters as a non-negative real $\sigma$ in a real-valued update equation, the
scheme cannot add energy — passivity ($Q_{\text{FDTD}}\le Q_{\text{wall}}$) is guaranteed by
construction, not by a sign convention that has to be gotten right and tested (contrast Module
4's $\Delta$-conjugate bug, which would have recurred in Ritz and FEM). The passivity *test*
(Section 6) is retained anyway as a cheap regression guard, but the failure mode it caught
elsewhere structurally cannot occur here — worth stating so nobody spends time hunting for it.

### 0.6 FFT from a scientific library, never hand-rolled

Per the instruction: the frequency-domain extraction uses `scipy.fft` (or `numpy.fft`) and
`scipy.signal` for peak-finding/windowing — no custom DFT. This module's own contribution to
the extraction is the physics (which peak, how linewidth maps to $Q$), not the transform.

---

## 1. `grid/` — Yee staggering (`grid/yee.py`)

Standard Yee arrangement: on a cell of size $(\Delta x,\Delta y,\Delta z)$, the three $E$
components sit at edge midpoints and the three $H$ components at face centers, each component
offset by half a cell from the cell corner along the appropriate axes. Concretely, relative to
integer cell index $(i,j,k)$ at position $(i\Delta x, j\Delta y, k\Delta z)$:

| Component | Spatial offset from corner |
|---|---|
| $E_x$ | $(+\tfrac12\Delta x,\ 0,\ 0)$ |
| $E_y$ | $(0,\ +\tfrac12\Delta y,\ 0)$ |
| $E_z$ | $(0,\ 0,\ +\tfrac12\Delta z)$ |
| $H_x$ | $(0,\ +\tfrac12\Delta y,\ +\tfrac12\Delta z)$ |
| $H_y$ | $(+\tfrac12\Delta x,\ 0,\ +\tfrac12\Delta z)$ |
| $H_z$ | $(+\tfrac12\Delta x,\ +\tfrac12\Delta y,\ 0)$ |

This staggering is what makes the discrete curl a centered difference (second-order accurate)
and is the same edge/face field placement that makes edge elements work in FEM — but here
it's a fixed grid, not a mesh. `grid/yee.py` provides the component→physical-coordinate map
for each field array, vectorized (`(N,3)` coordinate output for any component's whole array),
so that Section 2's rasterization and Section 3's excitation/probing can ask "where in space
is $E_z[i,j,k]$?" without re-deriving offsets.

`E` and `H` are stored on grids offset by half a time step as well (leapfrog, Section 5) — the
temporal stagger, distinct from the spatial stagger above.

---

## 2. `grid/` — material rasterization (`grid/rasterize.py`)

For each of the six field-component grids, build the material coefficient arrays the update
equations (Section 5) need, by evaluating membership at that component's staggered location:

- **Cavity interior mask**: `cavity_mode.contains(component_coords)` → which update cells are
  inside the cavity at all (cells outside are held at the PEC boundary, Section 5.3).
- **Sample mask**: `sample.region.contains(component_coords)` → which cells get the sample's
  $\epsilon_r',\sigma$ versus the background $\epsilon_{bg},\sigma=0$.

The rasterization is evaluated **per component grid**, at that component's own staggered
coordinates — not once at cell centers and reused. Evaluating a material boundary at the wrong
staggered location is a subtle half-cell error that biases the effective sample volume; doing
it per-component avoids it. `contains()` is already vectorized (`(N,3)` in, `(N,)` bool out) in
both Module 1 and Module 3, so each component's whole grid is one call.

**Cell-membership is the only place staircasing enters** — a curved cavity or sample boundary
becomes a binary in/out decision per staggered point, with no sub-cell treatment in a first
implementation (sub-cell/conformal correction is a documented later refinement, not a
first-pass requirement, mirroring how the meshing module deferred interface-refined element
sizing).

---

## 3. Excitation and probing (`fdtd/source.py`)

**Excitation**: a soft source with the *spatial* profile of the mode of interest —
`field_provider.E(coords)` sampled onto the $E$-component grids — modulated by a temporally
short pulse (Gaussian-modulated sinusoid centered at $f_0$, with bandwidth wide enough to
cover the expected perturbed frequency but narrow enough not to strongly excite neighboring
modes). Using Module 1's own mode shape selectively rings up the single mode of interest,
keeping the ringdown a clean single decaying sinusoid rather than a superposition — the same
deliberate reuse of validated code as the Ritz basis choice.

**Probing**: record a scalar time series — the dominant $E$ component sampled at a fixed point
where that mode's field is large (read from `field_provider.E` to pick a near-maximum,
node-avoiding location). One scalar probe is sufficient for $(f,Q)$ extraction; recording full
field snapshots is unnecessary for this module's purpose and is not done by default.

Excitation is turned off after the pulse; the recorded ringdown is the source-free decay used
for extraction (Section 6.2 note: the analysis window must start *after* the source is off, or
the driven portion contaminates the linewidth).

---

## 4. Single-frequency-matched conductivity (`fdtd/materials.py`)

$$\sigma = \omega_0\,\epsilon_0\,\epsilon_r'\,\tan\delta_e = 2\pi f_0\,\epsilon_0\,\epsilon_r'\,\tan\delta_e$$

matching the specified loss tangent at the mode frequency $f_0$ (read from
`field_provider.f0`). $\epsilon_r'=\mathrm{Re}(\epsilon_r)$ and $\tan\delta_e=-\mathrm{Im}(\epsilon_r)/\mathrm{Re}(\epsilon_r)$
come from Module 3's `Material` (its `loss_tangent_e` property already computes the latter).

### 4.1 Wall loss

The cavity walls are modeled with a matching-conductivity or surface-impedance treatment
whose effective loss reproduces `cavity_mode.Q_wall(Rs)` — the same target used everywhere
else. The cleanest first implementation is to *not* model walls with FDTD loss at all, run the
cavity lossless-walled, and add the wall contribution afterward as an independent
reciprocal-$Q$ term (Section 6.2), exactly as Module 4, Ritz, and the abandoned FEM plan all
combined wall loss additively. This keeps the FDTD run modeling only the sample's loss, which
is the quantity of interest, and reuses the already-validated `Q_wall`.

### 4.2 $\mu_r = 1$ scope

Non-magnetic samples only, matching the Ritz plan's scope. $\mu=\mu_{bg}$ uniform simplifies
the $H$ update to a single constant coefficient.

### 4.3 Deferred: full dispersion

A Debye/Lorentz auxiliary-differential-equation material model (frequency-dependent, rigorous
away from $f_0$) is a substantially larger undertaking and is explicitly out of scope for a
first implementation — named here so the single-frequency match is understood as a deliberate
scope boundary, not an oversight.

---

## 5. Time stepping (`fdtd/stepper.py`)

### 5.1 Update equations

Leapfrog: update all $H$ from spatial curl of $E$, advance a half step, update all $E$ from
spatial curl of $H$ (with the conductivity term), advance a half step, repeat. In the lossy
region the $E$ update carries the standard exponentially-damped coefficient form
(semi-implicit treatment of the $\sigma E$ term, which stays stable for any $\sigma\ge0$ — the
structural passivity of 0.5):
$$E^{n+1} = C_a\,E^n + C_b\,(\nabla\times H)^{n+1/2}, \qquad C_a=\frac{1-\frac{\sigma\Delta t}{2\epsilon}}{1+\frac{\sigma\Delta t}{2\epsilon}},\quad C_b=\frac{\Delta t/\epsilon}{1+\frac{\sigma\Delta t}{2\epsilon}}$$
with $C_a=1,\ C_b=\Delta t/\epsilon$ recovered in the lossless ($\sigma=0$) region — a single
code path, coefficients differing per cell via the rasterized arrays from Section 2.

### 5.2 CFL stability (`fdtd/stability.py`)

$$\Delta t \le \frac{1}{c'\sqrt{\Delta x^{-2}+\Delta y^{-2}+\Delta z^{-2}}}, \qquad c'=\frac{1}{\sqrt{\epsilon_{bg}\mu_{bg}}}$$

using the fastest (background) wave speed — a higher-$\epsilon_r$ sample is always slower and
never binds. $\Delta t$ is computed automatically from the grid spacing with a safety factor
(e.g. $0.99\times$ the limit), **never a user-settable value that could be chosen unstable**.
This is a hard invariant, checked and enforced at construction, not documentation.

### 5.3 Boundary condition

PEC cavity walls: tangential $E$ forced to zero on all faces outside the cavity-interior mask
(Section 2). For an axis-aligned rectangular cavity this is exact; for a staircased curved
cavity it is the staircase approximation, whose error is quantified by the Section 6.4
convergence study.

---

## 6. Ringdown extraction (`fdtd/extract.py`)

### 6.1 Two equivalent routes, FFT-first

For a decaying mode $E(t)\propto e^{-t/\tau}\cos(2\pi f_r t)$:
$$Q = \pi f_r \tau \quad\text{(time-domain, envelope decay)} \qquad\equiv\qquad Q = \frac{f_r}{\Delta f_{3\text{dB}}} \quad\text{(frequency-domain, Lorentzian FWHM)}$$
These are the same relation (a decaying sinusoid's spectrum is Lorentzian with
$\Delta f_{3\text{dB}}=1/(\pi\tau)$). **Default: the frequency-domain route** — one `scipy.fft`
of the probe series (Section 0.6) gives both $f_r$ (peak location) and $Q$ (peak location ÷
$-3$ dB linewidth) with no separate envelope fit. The time-domain envelope fit is implemented
as an independent cross-check (they must agree — a discrepancy signals too short a record, an
aliased neighboring mode, or a source-window contamination).

**Implementation correction:** a full-length Hann window (the window originally suggested
here) systematically *overestimated* $Q$ by 30–40% on a synthetic $e^{-t/\tau}\cos(2\pi f_r t)$
test signal with known $Q$ — verified directly, not assumed. The reason: Hann is meant for a
*steady-state* sinusoid truncated by the record boundary, where leakage comes only from the
edges. A ringdown is already a physically decaying transient (near its peak at the record's
start, near zero by the end); multiplying the *entire* record by a full raised-cosine taper
reshapes the already-decaying envelope itself rather than just smoothing the boundary
discontinuity, biasing the extracted linewidth. `extract.py` defaults to a mild Tukey taper
(`('tukey', 0.2)`) instead — it only tapers the extreme edges (still smoothing the periodicity
discontinuity `scipy.fft` implicitly assumes) and leaves the interior, where the actual decay
lives, untouched. A boxcar (no window) is a viable alternative with similar accuracy; Tukey
was chosen for a small edge-cases-only leakage margin. Either way, the "envelope fit as
independent cross-check" language above is doing real work here: on the same synthetic test,
the time-domain (Hilbert-transform) envelope route recovered both $f_r$ and $Q$ to a few parts
in $10^{-6}$, essentially exact — it has no windowing-vs-decay-shape tension since it fits the
raw decay directly rather than its Fourier transform. (The Hilbert transform has its own
artifact, unrelated to windowing: being FFT-based, it implicitly treats the record as
periodic, and a truncated decaying sinusoid has a real discontinuity at the tail-to-head
wraparound that corrupts the reconstructed envelope/phase near *both* record edges — mitigated
by trimming a margin from each end before fitting, `extract.py`'s `edge_trim_front`/
`edge_trim_back`.)

### 6.2 Combining with wall loss

Per 4.1, the FDTD run yields the sample-and-radiation $Q_{\text{FDTD}}$ (walls lossless);
combine with the wall contribution by reciprocal addition, the same additive principle used
across the project:
$$\frac{1}{Q_{\text{loaded}}} = \frac{1}{Q_{\text{FDTD}}} + \frac{1}{Q_{\text{wall}}(R_s)}, \qquad Q_{\text{wall}}\to\infty \text{ if } R_s \text{ is None}$$
then assemble $\tilde\omega=2\pi f_r(1-j/(2Q_{\text{loaded}}))$ and return the existing
`PerturbationResult` fields — reusing Module 4's $f_{\text{calc}}/Q_{\text{calc}}$ convention
verbatim.

### 6.3 Record-length and windowing requirements

The analysis window must (a) start after the source is off (3, so the driven transient doesn't
widen the apparent linewidth) and (b) be long enough that the mode has decayed substantially
but not so long it's buried in numerical noise. A practical rule: record for several $\tau$,
estimated from a rough first $Q$ guess (Module 1's `Q_wall` or a coarse pre-run). Apply a
window (e.g. Hann, from `scipy.signal`) before the FFT to control spectral leakage that would
otherwise bias the linewidth. These are extraction-quality parameters, exposed and
convergence-checkable, not hidden constants.

**Implementation correction — the rough-Q guess:** "Module 1's `Q_wall`" alone, or a
pessimistic sample-only bound ($1/\tan\delta$, reasoning "the true filling-factor-weighted $Q$
can only exceed this, so it's a safe/conservative underestimate"), both turned out wrong in
practice, in opposite directions:

- Using only `Q_wall` (ignoring the sample) undersizes the record whenever the sample is the
  dominant loss: caught directly via a lossless-sample-but-lossy-wall run, where `Q_wall` was
  used correctly for *that* case but a *separately* lossy sample's much lower true $Q$ was
  invisible to a wall-only estimate.
- Using $1/\tan\delta$ (ignoring the filling factor) is not actually a safe lower bound on the
  needed *record length*: for a small sample deep in a large cavity (small filling factor, the
  common case) the true $Q$ can be tens of times $1/\tan\delta$, so the estimated $\tau$ is
  tens of times too short and the record ends before the real decay has meaningfully
  progressed — caught directly: a small, low-loss sample's FDTD-extracted $Q$ came out ~97%
  below `PerturbationModel`'s prediction using this heuristic ("conservative" was backwards —
  *underestimating* $Q$ *underestimates* $\tau$, which *shortens* the record, the unsafe
  direction).

Both are fixed by reusing `PerturbationModel` (Module 4) itself as the "coarse pre-run" this
section already called for — it is already-validated, closed-form, fast, and its filling-
factor/depolarization-aware $Q_{\text{calc}}$ (which already combines $Q_{\text{wall}}$ exactly
as Section 6.2 does) is a far better rough estimate than either partial heuristic. This is not
circular: the rough $Q$ only sizes how long to *run the simulation*; the independent FDTD
extraction is still what gets reported. If the rough estimate is non-finite (both wall and
sample lossless), the true $Q$ is unbounded and "several $\tau$" has no finite target — record
a large-but-fixed number of oscillation periods instead (`fdtd/model.py`'s
`_LOSSLESS_RECORD_PERIODS`), enough for good $f_r$ resolution without an unbounded run (an
earlier version that instead extrapolated toward a very large assumed $Q$ produced
multi-million-step runs at GHz frequencies with a femtosecond-scale CFL $\Delta t$).

### 6.4 Staircasing convergence

Unique to this method: for a curved (cylindrical/coaxial) cavity, the extracted $f_r$ carries a
grid-dependent staircase error. Quantified by re-running at successively finer $\Delta x$ and
confirming $f_r$ converges toward Module 1's closed-form $f_0$ (empty-cavity case). This is the
FDTD analogue of Ritz basis-size and FEM mesh-refinement convergence — the mandatory
self-convergence check before trusting any cross-comparison.

---

## 7. Verification plan

Ordering discipline, same as the other method modules: self-consistency and grid-only tests
before any physics cross-comparison.

### 7.1 `grid/` in complete isolation (no field ever stepped)

- **Yee offsets** (`grid/yee.py`): for each component, confirm the coordinate map places it at
  the correct half-cell offset (Section 1 table) — pure arithmetic, no solver.
- **Rasterization** (`grid/rasterize.py`): rasterize a `Sphere` sample onto the grid, sum the
  in-sample cell volumes, and confirm the total converges to the analytic sphere volume as the
  grid refines — cross-checked against the *closed-form* $\frac43\pi r^3$ computed in the test,
  **not** against `Sphere.volume()` (independence from Module 3, same discipline as the meshing
  module's independent-verification rule). Confirms staircasing behaves as expected and the
  per-component staggering (Section 2) doesn't bias the effective volume.

### 7.2 Stability

- **CFL enforcement** (`fdtd/stability.py`): confirm the computed $\Delta t$ satisfies the
  bound for a range of anisotropic grid spacings; confirm a deliberately over-large $\Delta t$
  is rejected/clamped, never silently used.
- **Long-run boundedness**: an empty lossless cavity, excited then left to run for many
  periods, must not grow without bound (a blunt but decisive stability regression).

### 7.3 Extraction, on synthetic signals (no FDTD run)

- Feed `fdtd/extract.py` a *synthetic* $e^{-t/\tau}\cos(2\pi f_r t)$ with known $f_r,\tau$ and
  confirm both the FFT route and the envelope-fit route recover $f_r$ and $Q=\pi f_r\tau$ to a
  stated tolerance. This isolates the signal-processing from the physics entirely — if
  extraction is wrong, this fails without needing a single Maxwell time step.

### 7.4 Empty-cavity physics check (the strong one)

- Run FDTD with **no sample**, lossless walls, extract $f_r$; confirm it converges to Module
  1's closed-form `f0` as the grid refines (rectangular: should match at modest resolution
  since no staircasing; cylindrical/coaxial: convergence study per 6.4). Observed in practice
  (rectangular, `cells_per_wavelength` 6 → 16): error shrinks from ~4% to ~0.6%, the expected
  standard Yee-grid numerical-dispersion trend (a genuine, distinct error source from
  staircasing, present even with an exactly wall-aligned grid).
- Add the wall-loss model, no sample; confirm extracted $Q$ matches `cavity_mode.Q_wall(Rs)`.
  This check touches neither the perturbation formula nor the sample — it isolates whether
  ringdown extraction and wall-loss modeling are correct on their own, a more targeted
  empty-system check than Ritz or the abandoned FEM plan had available. **Accuracy caveat**
  (see 6.2/6.3's rough-Q correction): because wall loss is never time-stepped, the combined
  $Q_{\text{loaded}}$ is only as accurate as the FDTD run's own finite-record noise floor is
  *far above* $Q_{\text{wall}}$ — for a first-pass (short-record, `record_periods`-scaled,
  staircasing-included) implementation this gives right-order-of-magnitude agreement (tens of
  percent) rather than tight numeric agreement; tightening it further would need a
  substantially longer run than is practical for a routine regression test. Documented as a
  known limitation of Section 4.1's own design choice (walls combined analytically, never
  simulated), not a bug.

### 7.5 Small-sample agreement with Module 4

- For a small, low-loss dielectric sample well inside the classical regime, confirm
  `FDTDModel.evaluate(sample)` agrees with `PerturbationModel.evaluate(sample)` on both
  $f_{\text{calc}}$ and $Q_{\text{calc}}$ to a tolerance consistent with the grid resolution —
  the primary cross-validation that assembly, stepping, and extraction are all correct
  together. **The sample itself must be resolved by several grid cells, independent of how
  well `cells_per_wavelength` resolves the cavity mode** — caught directly: a 1.5mm-radius
  sample at a ~3mm cell size (barely 1 cell across) gave ~196% Q error purely from
  rasterization/discretization of the sample's own geometry, unrelated to the cavity-mode
  resolution or the record-length fix above; a 2.5mm-radius sample at a ~2.1mm cell size
  (several cells across) brought the same comparison down to single-digit-percent $f_{calc}$
  error and tens-of-percent $Q_{calc}$ error. `cells_per_wavelength` alone does not guarantee
  this — a future refinement could size the grid from the smaller of (wavelength/N, sample
  extent/M), not implemented in this first pass.

### 7.6 Passivity regression (structurally guaranteed, checked anyway)

- Sweep passive lossy samples, confirm $Q_{\text{loaded}}\le Q_{\text{wall}}$ always. Per 0.5
  this cannot fail by construction; the test is a cheap guard against a coefficient-sign typo
  in the $C_a/C_b$ arrays, not against the conceptual sign issue that bit Module 4.

### 7.7 Interface compatibility (independent of physics correctness)

- Confirm `FDTDModel` satisfies the `PerturbationModel` protocol: `evaluate(sample)` returns a
  `PerturbationResult`; `field_provider` and `Rs_walls` are present and correctly typed; a
  `Measurement` (Module 5) constructed with an `FDTDModel` and its closed-form seed path
  (which calls `field_provider`/`Rs_walls`) runs without error — even on a coarse, physically
  crude grid, since this test checks the *contract*, not the accuracy.

---

## 8. Package layout

```
src/cavity_perturbation/fdtd/
    __init__.py         # public re-exports: FDTDModel, exceptions
    grid/
        __init__.py
        yee.py           # Yee spatial staggering, component->coordinate maps
        rasterize.py      # contains()-based material mask per component grid
    materials.py          # single-frequency conductivity match, coefficient arrays
    stability.py           # CFL dt computation + enforcement
    source.py               # mode-shaped soft-source excitation, probe placement
    stepper.py               # leapfrog E/H update loop
    extract.py                # scipy.fft / scipy.signal ringdown -> (f_r, Q)
    model.py                   # FDTDModel: PerturbationModel-shaped sibling, evaluate()
tests/test_fdtd/
    test_yee.py
    test_rasterize.py
    test_stability.py
    test_extract.py           # synthetic-signal tests, no solver
    test_source.py
    test_stepper.py
    test_empty_cavity.py       # 7.4
    test_small_sample.py       # 7.5
    test_passivity.py          # 7.6
    test_interface.py          # 7.7
```

Only `model.py` orchestrates more than one of the others; every other file is single-purpose,
per the atomization requirement. `grid/` is a self-contained sub-package (0.2) testable with
no solver present.

## 9. Step-by-step implementation order

1. `grid/yee.py` — pure staggering arithmetic, tested first (7.1).
2. `grid/rasterize.py` — `contains()`-based masks, volume-convergence test (7.1).
3. `stability.py` — CFL, tested standalone (7.2 CFL part).
4. `extract.py` — signal processing, tested entirely on synthetic signals (7.3) before any
   solver exists. Building this early de-risks the whole module: if extraction is unreliable,
   nothing downstream can be trusted, so prove it in isolation first.
5. `materials.py` — conductivity match and coefficient arrays.
6. `source.py` — excitation profile and probe placement.
7. `stepper.py` — the update loop; first validated by 7.2 long-run boundedness, then 7.4
   empty-cavity convergence.
8. `model.py` — assemble `FDTDModel`, wire wall-loss combination (6.2), interface (7.7).
9. Full Section 7 suite, ending with 7.5 small-sample agreement against Module 4.
