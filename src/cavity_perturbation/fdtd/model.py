"""FDTD Module -- model.py: FDTDModel, a PerturbationModel-shaped sibling
predicting (f, Q) by time-domain simulation + ringdown analysis, per
docs/fdtd_module_plan.md Section 0.1 and Section 6.

The only file in this sub-package that orchestrates more than one of the
others (Section 8) -- every other file here is single-purpose.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..cavity import CavityMode
from ..fields import FieldProvider, AnalyticalField
from ..perturbation import PerturbationModel, PerturbationResult, omega_tilde_to_result
from ..sample import Sample
from .diagnostics import FDTDDiagnostics
from .extract import extract_fft
from .grid.rasterize import rasterize_all
from .grid.yee import E_COMPONENTS, H_COMPONENTS, YeeGrid
from .materials import assemble_e_coefficients
from .source import build_modal_source, choose_probe_point, gaussian_modulated_pulse, gaussian_pulse_sigma_t
from .stability import stable_time_step
from .stepper import FDTDStepper

Array = np.ndarray


class FDTDCancelled(RuntimeError):
    """Raised from `_run` when `cancel_check` reports a user-requested stop
    mid-simulation (docs/gui_module_plan.md Section 6's cancellation
    addition) -- routed through `SolveWorker`'s existing exception-to-
    `.failed`-signal path exactly like any other run failure, no separate
    plumbing needed."""


_DEFAULT_CELLS_PER_WAVELENGTH = 20.0
_DEFAULT_MIN_CELLS_PER_AXIS = 8
_DEFAULT_EXCITATION_BANDWIDTH_FRACTION = 0.05
_DEFAULT_N_PULSE_SIGMA = 5.0
_DEFAULT_RECORD_PERIODS = 10.0
_DEFAULT_CFL_SAFETY_FACTOR = 0.99
_LOSSLESS_RECORD_PERIODS = 3000.0  # oscillation periods to record when neither wall nor sample loss is present


class FDTDModel:
    """Predicts (f_calc, Q_calc) for a `Sample` by running a leapfrog FDTD
    simulation of `cavity_mode`'s own mode, exciting it with a mode-shaped
    Gaussian pulse (source.py), recording the source-free ringdown at a
    fixed probe point, and extracting (f_r, Q) from that ringdown
    (extract.py) -- then combining with wall loss (Section 6.2) exactly as
    `PerturbationModel` does.
    """

    def __init__(
        self,
        cavity_mode: CavityMode,
        Rs_walls: float | None = None,
        cells_per_wavelength: float = _DEFAULT_CELLS_PER_WAVELENGTH,
        min_cells_per_axis: int = _DEFAULT_MIN_CELLS_PER_AXIS,
        excitation_bandwidth_fraction: float = _DEFAULT_EXCITATION_BANDWIDTH_FRACTION,
        n_pulse_sigma: float = _DEFAULT_N_PULSE_SIGMA,
        record_periods: float = _DEFAULT_RECORD_PERIODS,
        cfl_safety_factor: float = _DEFAULT_CFL_SAFETY_FACTOR,
    ) -> None:
        self._cavity_mode = cavity_mode
        self._Rs_walls = Rs_walls
        self._cells_per_wavelength = cells_per_wavelength
        self._min_cells_per_axis = min_cells_per_axis
        self._excitation_bandwidth_fraction = excitation_bandwidth_fraction
        self._n_pulse_sigma = n_pulse_sigma
        self._record_periods = record_periods
        self._cfl_safety_factor = cfl_safety_factor

        self._field_provider: FieldProvider = AnalyticalField(cavity_mode)
        # Grid, excitation profile, and probe location depend only on the
        # unperturbed cavity mode (Section 3), never on the sample -- built
        # once here and reused by every evaluate() call.
        self._grid = self._build_grid()
        self._source_profile = build_modal_source(self._grid, self._field_provider)
        self._probe_point, self._probe_component = choose_probe_point(cavity_mode, self._field_provider)

    @property
    def field_provider(self) -> FieldProvider:
        """Section 0.1: Module 5's closed-form seed calls this directly,
        bypassing evaluate()."""
        return self._field_provider

    @property
    def Rs_walls(self) -> float | None:
        """Section 0.1: same rationale as `field_provider`."""
        return self._Rs_walls

    def _build_grid(self) -> YeeGrid:
        rmin, rmax = self._cavity_mode.bounding_box()
        extent = rmax - rmin
        eps_bg = float(np.real(self._cavity_mode.epsilon_bg))
        mu_bg = float(np.real(self._cavity_mode.mu_bg))
        wavelength = 1.0 / (self._cavity_mode.f0 * math.sqrt(eps_bg * mu_bg))
        target_cell_size = wavelength / self._cells_per_wavelength
        n0, n1, n2 = (
            max(int(math.ceil(extent[i] / target_cell_size)), self._min_cells_per_axis) for i in range(3)
        )
        shape: tuple[int, int, int] = (n0, n1, n2)
        cell_size: tuple[float, float, float] = (
            float(extent[0]) / n0,
            float(extent[1]) / n1,
            float(extent[2]) / n2,
        )
        return YeeGrid(shape=shape, cell_size=cell_size, origin=rmin)

    def _record_duration(self, sample: Sample, f0: float) -> float:
        """Section 6.3: record for several tau, from a rough Q guess --
        obtained here by directly reusing `PerturbationModel` (Module 4),
        already-validated closed-form perturbation theory, purely as a fast
        stand-in for the "coarse pre-run" Section 6.3 asks for. This is not
        circular: the rough Q only sizes how long to run the *simulation*,
        it is never returned as this model's own answer -- the independent
        FDTD extraction is still what `evaluate` ultimately reports.

        An earlier version instead used a pessimistic bound (1/tan_delta,
        "the true filling-factor-weighted Q can only exceed this") reasoning
        that underestimating Q was conservative. That reasoning was
        backwards: underestimating Q underestimates tau, giving a *shorter*
        record than the real decay needs -- for a small sample deep in a
        large cavity (small filling factor, the common case), the true Q
        can be 50x the 1/tan_delta bound, so the record ended 50x too early
        and badly biased the extracted Q (caught directly: a small,
        low-loss sample's FDTD-extracted Q came out ~97% below
        `PerturbationModel`'s prediction using that heuristic). Reusing
        `PerturbationModel`'s own filling-factor-aware estimate (which
        already combines Q_wall in exactly the way Section 6.2 does) fixes
        this directly rather than patching the heuristic further.

        If the rough estimate is non-finite (a genuinely lossless sample
        and no wall loss), the true Q is unbounded, so "several tau" has no
        finite target -- recording toward an assumed large-but-arbitrary Q
        would make an ordinary empty-cavity run (Section 7.4) impractically
        long (also caught directly, via an even earlier Q-valued fallback:
        combined with a femtosecond-scale CFL dt at GHz frequencies, it
        produced multi-million-step runs). A record of
        `_LOSSLESS_RECORD_PERIODS` oscillation periods gives good frequency
        resolution for f_r -- the only quantity a genuinely loss-free run
        can meaningfully validate; any Q extracted from it reflects the
        finite record length, not the true (unbounded) value, an inherent,
        understood limitation of finite-time ringdown extraction."""
        rough_model = PerturbationModel(self._field_provider, self._Rs_walls)
        rough_Q = rough_model.evaluate(sample).Q_calc
        if not math.isfinite(rough_Q) or rough_Q <= 0.0:
            return _LOSSLESS_RECORD_PERIODS / f0
        tau_est = rough_Q / (math.pi * f0)
        return self._record_periods * tau_est

    def evaluate(self, sample: Sample) -> PerturbationResult:
        """Predict (f_calc, Q_calc) for `sample`. Raises ValueError if
        `sample.material` isn't passive (CLAUDE.md passivity guard, same
        boundary `PerturbationModel.evaluate` checks)."""
        result, _ = self._run(sample, capture=False)
        return result

    def evaluate_with_diagnostics(
        self,
        sample: Sample,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[PerturbationResult, FDTDDiagnostics]:
        """Same physics and same result as `evaluate` -- additionally
        returns the excitation waveform, ringdown spectrum, and a single
        end-of-excitation field snapshot for the GUI's plots
        (docs/gui_module_plan.md Section 2.1). `evaluate()` itself is
        untouched and stays exactly as cheap as before; this only adds
        bookkeeping when explicitly asked for.

        `progress_callback(current_step, total_steps)`, if given, is called
        periodically (roughly every 1% of the run, not every step -- this
        loop runs thousands of iterations and per-step callback overhead
        would be noticeable) across both the excitation and record loops.

        `cancel_check()`, if given, is polled every step (cheap -- a plain
        thread-safe flag read, unlike `progress_callback`'s Qt-signal
        overhead) and raises `FDTDCancelled` the moment it returns True,
        stopping the simulation early (docs/gui_module_plan.md Section 6's
        cancellation addition)."""
        result, diagnostics = self._run(
            sample, capture=True, progress_callback=progress_callback, cancel_check=cancel_check
        )
        assert diagnostics is not None  # capture=True always returns one
        return result, diagnostics

    def _run(
        self,
        sample: Sample,
        capture: bool,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> tuple[PerturbationResult, FDTDDiagnostics | None]:
        """Shared runner behind `evaluate`/`evaluate_with_diagnostics`
        (docs/gui_module_plan.md Section 2.1) -- one code path, `capture`
        only controls whether the (small, already-computed-anyway) extra
        bookkeeping is retained."""
        if not sample.material.is_passive:
            raise ValueError(
                f"material {sample.material!r} is not passive "
                "(requires eps''>=0, mu''>=0, eps'>0, mu'>0)"
            )

        grid = self._grid
        cavity_mode = self._cavity_mode
        f0 = cavity_mode.f0
        masks = rasterize_all(grid, cavity_mode, sample.region)
        dt = stable_time_step(
            grid.cell_size, cavity_mode.epsilon_bg, cavity_mode.mu_bg, self._cfl_safety_factor
        )
        e_coeffs = assemble_e_coefficients(
            grid, dt, cavity_mode.epsilon_bg, masks, f0=f0, sample_material=sample.material
        )
        stepper = FDTDStepper(grid, dt, cavity_mode.mu_bg, e_coeffs, masks)

        bandwidth = self._excitation_bandwidth_fraction * f0
        sigma_t = gaussian_pulse_sigma_t(bandwidth)
        t0 = self._n_pulse_sigma * sigma_t
        n_pulse_steps = int(math.ceil(2.0 * t0 / dt))

        duration = self._record_duration(sample, f0)
        n_record_steps = max(int(math.ceil(duration / dt)), 16)
        total_steps = n_pulse_steps + n_record_steps
        progress_stride = max(total_steps // 100, 1)

        excitation_times = np.empty(n_pulse_steps) if capture else None
        excitation_waveform = np.empty(n_pulse_steps) if capture else None

        t = 0.0
        for n in range(n_pulse_steps):
            if cancel_check is not None and cancel_check():
                raise FDTDCancelled("FDTD run cancelled by user")
            pulse_val = float(gaussian_modulated_pulse(np.array([t]), f0, t0, sigma_t)[0])
            if capture:
                assert excitation_times is not None and excitation_waveform is not None
                excitation_times[n] = t
                excitation_waveform[n] = pulse_val
            source = {c: pulse_val * self._source_profile[c] for c in E_COMPONENTS}
            stepper.step(e_source=source)
            t += dt
            if progress_callback is not None and n % progress_stride == 0:
                progress_callback(n, total_steps)

        # 0.3's "end of excitation" choice -- the natural place, since it's
        # exactly where the code already transitions from "inject" to
        # "record" below.
        field_snapshot = None
        if capture:
            field_snapshot = {c: stepper.E[c].copy() for c in E_COMPONENTS}
            field_snapshot.update({c: stepper.H[c].copy() for c in H_COMPONENTS})

        times = np.empty(n_record_steps)
        probe_series = np.empty(n_record_steps)
        for n in range(n_record_steps):
            if cancel_check is not None and cancel_check():
                raise FDTDCancelled("FDTD run cancelled by user")
            stepper.step(e_source=None)
            t += dt
            times[n] = t
            probe_series[n] = stepper.probe_value(self._probe_point, self._probe_component)
            if progress_callback is not None and n % progress_stride == 0:
                progress_callback(n_pulse_steps + n, total_steps)

        if progress_callback is not None:
            progress_callback(total_steps, total_steps)

        ringdown = extract_fft(times, probe_series)
        f_r_fdtd, Q_fdtd = ringdown.f_r, ringdown.Q

        if self._Rs_walls is not None:
            Q_wall = cavity_mode.Q_wall(self._Rs_walls)
            Q_loaded = 1.0 / (1.0 / Q_fdtd + 1.0 / Q_wall)
        else:
            Q_loaded = Q_fdtd

        omega_tilde = 2.0 * np.pi * f_r_fdtd * (1.0 - 1j / (2.0 * Q_loaded))
        result = omega_tilde_to_result(omega_tilde)

        if not capture:
            return result, None

        assert excitation_times is not None and excitation_waveform is not None and field_snapshot is not None
        diagnostics = FDTDDiagnostics(
            excitation_times=excitation_times,
            excitation_waveform=excitation_waveform,
            probe_times=times,
            probe_series=probe_series,
            spectrum_freqs=ringdown.spectrum_freqs,
            spectrum_power=ringdown.spectrum_power,
            field_snapshot=field_snapshot,
            snapshot_grid=grid,
        )
        return result, diagnostics
