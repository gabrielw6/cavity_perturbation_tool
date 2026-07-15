"""FDTD Module -- extract.py: ringdown -> (f_r, Q) via scipy.fft/signal.

Two independent routes (Section 6.1): the default frequency-domain FFT/
Lorentzian-FWHM route, and a time-domain envelope/phase-slope fit used as an
independent cross-check. No custom DFT or hand-rolled peak-finding numerics
(Section 0.6) -- `scipy.fft` and `scipy.signal` do the transform, window,
and analytic-signal construction; this module's own contribution is the
physics of picking the peak and mapping linewidth/decay to Q, not the
transform itself.

Built and tested (Section 7.3) entirely on synthetic signals, before any
FDTD solver exists -- if extraction is unreliable, nothing built on top of
it can be trusted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sp_fft
from scipy import signal as sp_signal

Array = np.ndarray


class ExtractionError(Exception):
    """Ringdown extraction could not find a clear, well-formed resonance."""


@dataclass(frozen=True)
class RingdownResult:
    f_r: float  # Hz
    Q: float
    method: str  # 'fft' or 'envelope'
    # docs/gui_module_plan.md Section 2.2: extract_fft already builds these
    # internally and used to discard them -- populated there for the GUI's
    # spectrum plot (FDTDDiagnostics, Section 2.1), reusing the array rather
    # than recomputing an FFT. extract_envelope's time-domain route doesn't
    # compute a spectrum at all, so these stay None there -- an existing,
    # correct distinction, not a gap. No caller that only reads f_r/Q/method
    # is affected (both constructors below only ever use keyword args).
    spectrum_freqs: Array | None = None
    spectrum_power: Array | None = None


def _interp_crossing(freqs: Array, power: Array, peak_idx: int, level: float, direction: int) -> float | None:
    """Starting at `peak_idx` (power >= level), step in `direction` until
    the next sample drops below `level`, then linearly interpolate the
    exact crossing frequency between the two bracketing bins. Returns None
    if the array ends before a crossing is found."""
    i = peak_idx
    n = len(freqs)
    while True:
        nxt = i + direction
        if nxt < 0 or nxt >= n:
            return None
        if power[nxt] < level:
            f_a, p_a = freqs[i], power[i]
            f_b, p_b = freqs[nxt], power[nxt]
            if p_a == p_b:
                return float(f_a)
            frac = (level - p_a) / (p_b - p_a)
            return float(f_a + frac * (f_b - f_a))
        i = nxt


def extract_fft(
    t: Array, signal: Array, window: str | tuple[str, float] = ("tukey", 0.2)
) -> RingdownResult:
    """Section 6.1 default route: window the probe series, take its
    one-sided power spectrum (`scipy.fft.rfft`), locate the dominant
    positive-frequency peak, and map its -3 dB (half-power) full width to
    Q = f_r / Delta_f_3dB (Lorentzian FWHM).

    Default window is a mild Tukey taper (`('tukey', 0.2)`), not a full-
    length Hann: a ringdown is already a physically decaying transient
    (typically near its peak amplitude at the record's start, near zero by
    the end), unlike the steady-state sinusoid a full-length window is
    normally meant for. Multiplying the *entire* record by a full raised-
    cosine taper (Hann) reshapes the already-decaying envelope itself,
    biasing the extracted linewidth (verified empirically: Hann
    systematically overestimated Q by 30-40% on a synthetic e^{-t/tau} test
    signal with known Q, whereas a mild Tukey taper -- which only tapers
    the extreme edges, leaving the interior untouched -- came within a few
    percent). The taper is still needed at the edges themselves to smooth
    the periodicity discontinuity `scipy.fft` implicitly assumes."""
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n < 8:
        raise ExtractionError(f"signal too short to extract a resonance ({n} samples)")
    dt = float(t[1] - t[0])

    win = sp_signal.get_window(window, n)
    spectrum = sp_fft.rfft(signal * win)
    freqs = sp_fft.rfftfreq(n, d=dt)
    power = np.abs(spectrum) ** 2

    search = freqs > 0.0  # exclude DC so any residual offset can't win
    if not np.any(search):
        raise ExtractionError("no positive-frequency content in signal")
    peak_idx = int(np.argmax(np.where(search, power, -np.inf)))
    if peak_idx == 0 or peak_idx == len(freqs) - 1:
        raise ExtractionError("resonance peak sits at the spectrum edge -- extend the record")

    f_r = float(freqs[peak_idx])
    half_power = power[peak_idx] / 2.0

    f_low = _interp_crossing(freqs, power, peak_idx, half_power, direction=-1)
    f_high = _interp_crossing(freqs, power, peak_idx, half_power, direction=+1)
    if f_low is None or f_high is None:
        raise ExtractionError("could not bracket the -3 dB points around the resonance peak")

    delta_f = f_high - f_low
    if delta_f <= 0.0:
        raise ExtractionError(f"non-positive -3dB linewidth {delta_f!r}")
    return RingdownResult(
        f_r=f_r, Q=f_r / delta_f, method="fft", spectrum_freqs=freqs, spectrum_power=power
    )


_EDGE_TRIM_FRONT = 0.02
_EDGE_TRIM_BACK = 0.30


def extract_envelope(
    t: Array,
    signal: Array,
    edge_trim_front: float = _EDGE_TRIM_FRONT,
    edge_trim_back: float = _EDGE_TRIM_BACK,
) -> RingdownResult:
    """Time-domain cross-check (Section 6.1): the analytic signal's envelope
    decay gives tau (Q = pi*f_r*tau), and its phase slope gives f_r --
    `scipy.signal.hilbert` only, no hand-rolled analytic-signal construction.

    `scipy.signal.hilbert` is FFT-based and implicitly treats the record as
    periodic; a truncated decaying sinusoid has a real discontinuity at the
    tail-to-head wraparound, which corrupts the reconstructed envelope and
    phase near *both* edges of the record (empirically, worst from roughly
    the last third onward, where the true signal has decayed close to the
    wraparound artifact's own floor -- verified directly against the known
    analytic envelope in tests/test_fdtd/test_extract.py, not assumed).
    `edge_trim_front`/`edge_trim_back` discard that margin before fitting --
    a standard mitigation for FFT-based Hilbert transforms on finite,
    non-periodic ringdowns.
    """
    t = np.asarray(t, dtype=float)
    signal = np.asarray(signal, dtype=float)
    n = signal.size
    if n < 8:
        raise ExtractionError(f"signal too short to extract a resonance ({n} samples)")

    analytic = sp_signal.hilbert(signal)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))

    lo = int(edge_trim_front * n)
    hi = n - int(edge_trim_back * n)
    if hi - lo < 4:
        raise ExtractionError("record too short after edge trimming to fit a ringdown")
    t_core, env_core, phase_core = t[lo:hi], envelope[lo:hi], phase[lo:hi]

    floor = 1e-6 * np.max(env_core)
    valid = env_core > floor
    if np.count_nonzero(valid) < 4:
        raise ExtractionError("envelope decays below the noise floor almost immediately")

    slope_env, _ = np.polyfit(t_core[valid], np.log(env_core[valid]), 1)  # log(env) = log(A) - t/tau
    if slope_env >= 0.0:
        raise ExtractionError(f"envelope is not decaying (slope={slope_env!r}) -- not a ringdown")
    tau = -1.0 / slope_env

    slope_phase, _ = np.polyfit(t_core, phase_core, 1)
    f_r = slope_phase / (2.0 * np.pi)
    if f_r <= 0.0:
        raise ExtractionError(f"non-positive extracted frequency {f_r!r}")

    return RingdownResult(f_r=f_r, Q=np.pi * f_r * tau, method="envelope")


def extract_ringdown(t: Array, signal: Array, window: str = "hann") -> RingdownResult:
    """Section 6.1: FFT-first default route. Call `extract_envelope`
    separately for the independent time-domain cross-check (the two must
    agree; a discrepancy signals too short a record, an aliased neighboring
    mode, or source-window contamination -- Section 6.1)."""
    return extract_fft(t, signal, window=window)
