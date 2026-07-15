# GUI Module — Architecture Plan

Scope: a new, separate package, `cavity_perturbation_gui`, giving interactive access to the
four forward solvers (analytical, perturbational, variational/Ritz, FDTD) plus the inverse
fit, via PySide2 + PyQtGraph. No physics, no equations, no code in this doc — layout,
interfaces, and build order only. Unlike the module docs under `docs/module*.md`, this one is
deliberately light: the four solver docs remain the only source of truth for any formula: this
doc's entire job is to define, precisely, the *shape* of the boundary between this package and
`cavity_perturbation`.

---

## 0. Design decisions up front

- **0.1 No physics lives here.** Every number this package displays is computed by
  `cavity_perturbation`; this package's own code is limited to translating UI state into calls
  against that package's public interface, and translating results back into plots. If a
  computation looks like it needs new physics rather than a new *view* of existing physics, it
  belongs in `cavity_perturbation`, not here.
- **0.2 PyQtGraph only, everywhere, including the 3D geometry view.** The original sketch split
  matplotlib (3D sidebar) from PyQtGraph (2D curves/planes). Dropped in favor of PyQtGraph's own
  `pyqtgraph.opengl.GLViewWidget` for the 3D view — one plotting library, one set of idioms,
  one theme, across the whole app, rather than two. Honest cost/benefit: this doesn't avoid a
  new dependency outright — `pyqtgraph.opengl` needs `PyOpenGL` installed — but it avoids adding
  *matplotlib* on top of that, and keeps every plot in the app (2D and 3D) behind one API.
- **0.3 FDTD field snapshot: single capture, end of excitation.** Per Section 2.1: one
  six-component field snapshot taken at the instant the excitation pulse ends (not a
  scrubbable time history — see Section 11).
- **0.4 A fifth tab: Inversion.** Wraps `InverseSolver`/`Measurement`/`FitResult`, consuming
  either pasted-in `(f_meas, Q_meas)` values or a result captured directly from one of the four
  forward tabs via a "use as measurement" action (Section 5.6).
- **0.5 Every solver-package change below is purely additive.** New dataclasses, new
  `evaluate_with_diagnostics` methods alongside the existing `evaluate`, one widened type
  annotation. No existing method signature, return type, or test changes — `tests/test_ritz.py`,
  `tests/test_fdtd/*`, `tests/test_inverse.py`, and every script under `scripts/` keep working
  unmodified. This matters more here than usual: the whole point of this package is to never
  need to touch `cavity_perturbation`'s existing surface, only extend it.

---

## 1. Objectives of the interface boundary

This is the part worth being explicit about, since it's the actual point of keeping this a
separate package. The boundary between `cavity_perturbation_gui` and `cavity_perturbation`
must guarantee:

1. **One-way dependency.** `cavity_perturbation_gui` imports `cavity_perturbation`;
   `cavity_perturbation` never imports, references, or knows about the GUI package. Checkable
   mechanically (a grep for `cavity_perturbation_gui` under `src/cavity_perturbation/` should
   always come back empty).
2. **One entry point per physics concern.** Every call into `cavity_perturbation` from GUI code
   goes through `adapters/` (Section 4) — a widget never constructs a `CavityMode`, `Sample`,
   or solver object itself, and never reaches into a solver's internals (`FDTDStepper`,
   `YeeGrid`, raw eigenvectors) except through an object one of the adapters explicitly
   returned.
3. **Stable, narrow result types.** Each adapter call returns one plain, immutable result
   object (a `PerturbationResult`, a diagnostics dataclass, a `FitResult`, or an exception) —
   never a live solver object the UI could call back into off-thread.
4. **Independent testability.** `adapters/` is ordinary Python with no Qt import — testable
   with plain pytest, no display server needed. `widgets/` is tested with `pytest-qt` against
   *mocked* adapters, so widget tests never run real physics and physics tests never import Qt.
5. **No solver-side awareness of "why."** The diagnostics additions in Section 2 expose data
   (arrays, coefficients, snapshots) — they do not know they're for plotting. Keeps
   `cavity_perturbation` a physics library first.

---

## 2. Solver-package additions (closing the two gaps)

### 2.1 FDTD: excitation, ringdown, spectrum, and one field snapshot

New file `fdtd/diagnostics.py`:

```
FDTDDiagnostics:
    excitation_times, excitation_waveform      # the injected pulse, during the source loop
    probe_times, probe_series                  # the recorded ringdown (already computed today)
    spectrum_freqs, spectrum_power              # from extract_fft, see 2.2
    field_snapshot: dict[str, Array]            # Ex..Hz, captured once
    snapshot_grid: YeeGrid                      # coordinates for slicing the snapshot into a plane
```

`FDTDModel` gains `evaluate_with_diagnostics(sample) -> (PerturbationResult, FDTDDiagnostics)`.
Internally, `evaluate()` and `evaluate_with_diagnostics()` both call a shared private runner
with a `capture: bool` flag — the pulse loop already computes `pulse_val` per step (just needs
accumulating when `capture=True`), and the snapshot is a plain copy of `stepper.E`/`stepper.H`
taken right after the excitation loop ends (0.3's "end of excitation" choice — the natural
place, since it's exactly where today's code already transitions from "inject" to "record").
`evaluate()`'s own signature and return type are untouched.

### 2.2 `RingdownResult` gains optional spectrum fields

`extract_fft` already builds `freqs`/`power` internally and discards them. Add two
`Array | None = None` fields to `RingdownResult` (`spectrum_freqs`, `spectrum_power`),
populated by `extract_fft`. `extract_envelope`'s time-domain route doesn't compute a spectrum
at all, so those fields stay `None` there — an existing, correct distinction, not a gap.
No caller that only reads `f_r`/`Q`/`method` is affected.

### 2.3 Ritz: expose the mixing coefficients

New `RitzDiagnostics(basis_modes: list[CavityMode], coefficients: Array)` — the eigenvector
`eigenvectors[:, k_star]` `evaluate()` already computes and currently discards, alongside the
basis it was computed against. `RitzCorrectedModel` gains
`evaluate_with_diagnostics(sample) -> (PerturbationResult, RitzDiagnostics)`, same
shared-private-runner pattern as 2.1. This is what makes the Variational tab's field plot a
genuine reconstruction, $E(r)=\sum_i c_i E_i(r)$ over `basis_modes`, rather than reusing the
unperturbed single-mode field the Analytical/Perturbational tabs show — the reconstruction
arithmetic itself belongs in this package's `adapters/field_sampling.py` (Section 4), not in
`ritz.py`, per 0.1.

### 2.4 One typing widening, for the Inversion tab

`inverse.py`'s `Measurement.model` is annotated `PerturbationModel`. `RitzCorrectedModel` and
`FDTDModel` are already structurally identical (`evaluate(sample) -> PerturbationResult`,
`field_provider`, `Rs_walls`) — `InverseSolver` works with either today at runtime, duck-typed,
but `mypy --strict` would reject passing one to `Measurement(model=...)`. Add a
`PerturbationModelLike` `Protocol` (in `perturbation.py`, next to `PerturbationResult`) capturing
exactly that shape, and widen `Measurement.model: PerturbationModelLike`. This is what lets the
Inversion tab fit against whichever forward model the user is exploring, not only Module 4's.

### 2.5 What does not change

`PerturbationModel.evaluate`, `RitzCorrectedModel.evaluate`, `FDTDModel.evaluate`,
`PerturbationResult`, `InverseSolver`, every existing test. The four `evaluate_with_diagnostics`
additions and the `RingdownResult`/`Measurement` widenings are the entire footprint in
`cavity_perturbation`.

---

## 3. Package layout

```
src/cavity_perturbation_gui/
    adapters/
        cavity_adapter.py        # sidebar params -> CavityMode
        sample_adapter.py        # sidebar params -> Sample / Material / SampleRegion
        analytical_runner.py
        perturbation_runner.py
        ritz_runner.py
        fdtd_runner.py
        inversion_runner.py      # -> InverseSolver / Measurement / FitResult
        field_sampling.py        # plane-of-points -> E/H values; Ritz reconstruction (2.3)
        geometry_description.py  # cavity/sample -> plain geometric primitives for the 3D view
    workers/
        solve_worker.py          # QObject + moveToThread around one runner call
    widgets/
        main_window.py           # tabs + sidebar + log bar
        sidebar.py
        geometry_view3d.py       # GLViewWidget (0.2)
        curve_plot.py            # PyQtGraph: resonance curves, FDTD time-domain + FFT
        field_plane_view.py      # PyQtGraph image view, field cross-sections
        log_panel.py
        tabs/
            analytical_tab.py
            perturbation_tab.py
            variational_tab.py
            fdtd_tab.py
            geometry_tab.py       # added post-plan: dedicated full-size 3D view, see 5's note
            inversion_tab.py
    logging_bridge.py
    app.py / __main__.py
tests/test_gui/                  # pytest-qt; adapters mocked for widget tests
```

---

## 4. The adapter layer

One runner per tab, uniform shape: takes the shared cavity/sample parameters plus that tab's
own extra settings (e.g. FDTD's `cells_per_wavelength`, Ritz's basis size / convergence
tolerance), returns either a result-plus-diagnostics pair or raises — workers (Section 6)
catch and forward exceptions as log entries, never let one crash the app.

`field_sampling.py` is shared across all four forward tabs: given a plane specification (two
axes + a fixed third coordinate, or "through the sample center"), it builds the point grid,
masks points outside `cavity.contains()` to `NaN`, and evaluates whichever field is available —
closed-form `E(r)`/`H(r)` for Analytical/Perturbational, the Ritz reconstruction from 2.3's
coefficients for Variational, or an indexed slice of `FDTDDiagnostics.field_snapshot` for FDTD.
Each tab's plot is labeled with what it's actually showing (Section 5 note on Perturbational).

`geometry_description.py` turns a `CavityMode` + `SampleRegion` into plain primitives (box
corners, cylinder radius/length, annulus inner/outer radius, sphere/cylinder/slab
center+dimensions+axis) — `geometry_view3d.py` only ever draws primitives, never touches
`cavity_perturbation` types directly.

---

## 5. Widgets

- **Sidebar**: cavity type + dimensions + mode indices + background eps_r/mu_r + Rs source;
  sample shape + position + orientation + material — one shared parameter model every tab reads.
  `geometry_view3d.py` redraws live as parameters change.
- **Added post-plan: a dedicated "3D Geometry" tab** (`geometry_tab.py`), on explicit user
  request after v1 landed — the sidebar-docked `geometry_view3d.py` instance is small (shares
  the left column with the sidebar) and only redraws per solver run, showing no sample at all
  while the Analytical tab (which never takes one) is the last one run. The new tab owns its
  *own* `GeometryView3D` instance, full-size, redrawing directly from the sidebar's current
  parameters on every `Sidebar.changed` signal — no Run action, no background thread, since
  building the geometry primitives is cheap closed-form Module 1/3 construction with no field
  solve involved. Always shows cavity + sample together, independent of which (if any) forward
  tab has been run.
- **Four forward tabs**: a Run button (disabled mid-solve), the resonance-curve plot
  (`curve_plot.py`, Lorentzian from `f_calc`/`Q_calc`, same construction as
  `scripts/_common.py`'s existing plotting logic), and the field-plane view (two perpendicular
  planes side by side). Perturbational's field plot is explicitly labeled "unperturbed cavity
  field — the perturbation model corrects this with a single scalar inside the sample, not a
  resolved internal field" (Section 2's honesty point carried through to the UI).
- **FDTD tab** additionally gets two more `curve_plot.py` instances: excitation waveform
  (time domain) and its spectrum (from `FDTDDiagnostics`, reusing 2.2's arrays rather than
  re-computing an FFT in the GUI).
- **Inversion tab**: a measurement list (add rows manually, or via a "use this result" button
  exposed on each forward tab that appends its last `PerturbationResult` as a synthetic
  measurement bound to that tab's own model instance — using 2.4's widened typing to allow a
  Ritz- or FDTD-backed measurement, not only Perturbational); `fit_mu` toggle; runs
  `InverseSolver.fit()`; displays `FitResult` (eps, mu, `condition_number`, covariance) plus a
  residual/consistency readout.
- **Log bar**: bottom-docked, spans the whole window, always visible regardless of active tab.

---

## 6. Threading

Every Run action goes through `solve_worker.py`, never the GUI thread. Signals:
`finished(result, diagnostics)`, `failed(exception)`, `log(str)`. No mid-run cancellation in
v1 (Section 11) — the button set just disables for the duration of that tab's run.

---

## 7. Logging bridge

Standard `logging`, not stdout capture: a `logging.Handler` subclass posts formatted records to
`log_panel.py` via a Qt signal (safe across threads); `logging.captureWarnings(True)` picks up
`ritz.py`'s existing degenerate-mode-mixing `warnings.warn` unmodified. Runners additionally log
their own start/parameters/finish through the same logger, at `INFO`, so a run's provenance is
visible in the log bar without instrumenting every widget.

---

## 8. Testing plan

Lighter than the physics modules' Section 6/7 validation suites, since there's no new physics
to validate here — only that the interface is honored:

- `adapters/`: plain pytest, real `cavity_perturbation` calls, no Qt — confirms each runner's
  translation from parameters to solver objects is correct and that results round-trip through
  the diagnostics dataclasses correctly.
- `widgets/`: `pytest-qt`, adapters mocked — confirms button-click-to-adapter-call wiring,
  that a raised exception surfaces in the log bar rather than crashing, and that the field-plane
  view relabels itself correctly per tab (the Perturbational-vs-Ritz distinction in Section 5).
- One `test_gui/test_no_reverse_import.py`: greps `src/cavity_perturbation/` for any reference
  to `cavity_perturbation_gui`, enforcing Section 1's one-way dependency mechanically rather
  than by convention alone.
- `cavity_perturbation`'s own existing suite (`tests/`) is the regression guard for Section 2's
  additions — run unmodified; new tests there only for the four new
  `evaluate_with_diagnostics` methods, `RingdownResult`'s new fields, and the `Protocol` widening.

---

## 9. Dependencies

New optional extra in `pyproject.toml`, separate from the existing `viz` extra (kept for
`scripts/visualize_cavity.py`, which stays matplotlib-only and untouched):

```
gui = ["PySide6", "pyqtgraph", "PyOpenGL", "pytest-qt"]
```

**Implementation correction — PySide2 → PySide6.** This section originally specified PySide2.
Verified directly (not assumed): PySide2 (Qt5, EOL upstream) has no distribution at all for this
project's Python version — `pip index versions PySide2` returns no match. PySide6 does
(actively maintained, current Qt6). Substituted with no architectural impact: `pyqtgraph`
supports both PySide2 and PySide6 transparently through its own Qt-binding shim
(`pyqtgraph.Qt`), so nothing about Section 0.2's "PyQtGraph only, everywhere" decision, the
widget layer's design, or any other section changes — only the specific binding library named
here. `pytest-qt` (Section 8's widget-testing tool) is added to the same extra rather than a
separate dev-dependencies group, since this project has no such group yet.

`cavity_perturbation`'s own core dependencies (`numpy`, `scipy`) are unaffected — the GUI extra
never leaks into the core package's own `dependencies` list.

---

## 10. Build order

1. Section 2's solver-package additions first, each landing alongside its own small pytest
   addition — this package can't be built against interfaces that don't exist yet, and closing
   the gaps first means the GUI side never has to work around a missing return value.
2. `adapters/` (no Qt yet) — testable in isolation, including `field_sampling.py`'s Ritz
   reconstruction, before any widget exists to display it.
3. `workers/` + `logging_bridge.py` — the threading/logging skeleton, testable with a trivial
   fake runner.
4. `main_window.py` + `sidebar.py` + `geometry_view3d.py`, wired to real adapters — gets a
   runnable (if single-tab) app early.
5. The four forward tabs, one at a time (Analytical first — no sample/solve complexity —
   through to FDTD last, since it has the most moving parts: two extra plots and the snapshot
   slicing).
6. Inversion tab last, since it depends on the forward tabs' "use as measurement" action
   existing first.
7. `test_gui/test_no_reverse_import.py` from day one, not as an afterthought — cheapest possible
   guard against the one-way-dependency rule quietly eroding as the package grows.

---

## 11. Explicitly deferred / out of scope for v1

- Mid-run cancellation (needs a cooperative-cancel hook threaded through `FDTDStepper`'s loop —
  a solver-side addition, not a GUI-side one, and not required for a first version).
- Scrubbable multi-snapshot FDTD field history (0.3's single end-of-excitation snapshot is the
  v1 scope; capturing every step is memory-expensive and not needed to satisfy the original
  request).
- Any new physics, meshing, or geometry beyond the three canonical cavity types and
  `Sphere`/`Cylinder`/`Slab` samples already in `cavity_perturbation` — this package visualizes
  what exists, it doesn't motivate building irregular-geometry support.
- Result export (CSV/plot image saving) — natural follow-on, not part of this plan.
