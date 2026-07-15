"""FDTD Module -- diagnostics.py: optional per-run diagnostic data for the
GUI, per docs/gui_module_plan.md Section 2.1.

Purely additive and purely data -- `FDTDDiagnostics` doesn't know it's for
plotting (docs/gui_module_plan.md Section 1.5), and nothing in `evaluate()`'s
own (unchanged) return value or any existing test depends on this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid.yee import YeeGrid

Array = np.ndarray


@dataclass(frozen=True)
class FDTDDiagnostics:
    """One run's worth of diagnostic data, captured only when
    `FDTDModel.evaluate_with_diagnostics` is used (never by plain
    `evaluate()`, which stays exactly as cheap as before)."""

    excitation_times: Array  # s, one entry per pulse-loop step
    excitation_waveform: Array  # dimensionless pulse envelope*carrier value, same shape
    probe_times: Array  # s, the recorded ringdown (already computed by evaluate() today)
    probe_series: Array  # same shape as probe_times
    spectrum_freqs: Array | None  # Hz, from extract_fft (Section 2.2) -- None if unavailable
    spectrum_power: Array | None  # same shape as spectrum_freqs
    field_snapshot: dict[str, Array]  # Ex..Hz, captured once, end of excitation (Section 0.3)
    snapshot_grid: YeeGrid  # coordinates for slicing field_snapshot into a plane
