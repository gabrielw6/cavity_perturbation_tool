"""docs/fdtd_module_plan.md Section 7.3 -- extraction on synthetic signals,
no FDTD solver involved. Isolates signal processing from physics: if this
fails, no Maxwell time step was needed to find the bug."""
import numpy as np
import pytest

from cavity_perturbation.fdtd.extract import (
    ExtractionError,
    extract_envelope,
    extract_fft,
    extract_ringdown,
)


def _synthetic_ringdown(f_r: float, Q: float, n: int = 65536, n_periods_decay: int = 10):
    tau = Q / (np.pi * f_r)
    duration = n_periods_decay * tau
    t = np.linspace(0.0, duration, n, endpoint=False)
    signal = np.exp(-t / tau) * np.cos(2.0 * np.pi * f_r * t)
    return t, signal, tau


@pytest.mark.parametrize("f_r,Q", [(1.0e9, 500.0), (2.5e9, 2000.0), (5.0e8, 200.0)])
def test_fft_route_recovers_fr_and_q(f_r, Q):
    t, signal, _ = _synthetic_ringdown(f_r, Q)
    result = extract_fft(t, signal)
    assert result.method == "fft"
    assert result.f_r == pytest.approx(f_r, rel=0.01)
    assert result.Q == pytest.approx(Q, rel=0.1)


@pytest.mark.parametrize("f_r,Q", [(1.0e9, 500.0), (2.5e9, 2000.0), (5.0e8, 200.0)])
def test_envelope_route_recovers_fr_and_q(f_r, Q):
    t, signal, _ = _synthetic_ringdown(f_r, Q)
    result = extract_envelope(t, signal)
    assert result.method == "envelope"
    assert result.f_r == pytest.approx(f_r, rel=1e-4)
    assert result.Q == pytest.approx(Q, rel=1e-3)


def test_fft_and_envelope_routes_agree_with_each_other():
    f_r, Q = 1.5e9, 800.0
    t, signal, _ = _synthetic_ringdown(f_r, Q)
    fft_result = extract_fft(t, signal)
    env_result = extract_envelope(t, signal)
    assert fft_result.f_r == pytest.approx(env_result.f_r, rel=0.02)
    assert fft_result.Q == pytest.approx(env_result.Q, rel=0.15)


def test_extract_ringdown_default_is_fft_route():
    t, signal, _ = _synthetic_ringdown(1e9, 500.0)
    assert extract_ringdown(t, signal).method == "fft"


def test_too_short_signal_raises():
    t = np.linspace(0.0, 1e-9, 4)
    signal = np.cos(2.0 * np.pi * 1e9 * t)
    with pytest.raises(ExtractionError):
        extract_fft(t, signal)
    with pytest.raises(ExtractionError):
        extract_envelope(t, signal)


def test_non_decaying_signal_rejected_by_envelope_route():
    t = np.linspace(0.0, 1e-6, 4096)
    signal = np.cos(2.0 * np.pi * 1e9 * t)  # constant amplitude, not a ringdown
    with pytest.raises(ExtractionError):
        extract_envelope(t, signal)


def test_growing_signal_rejected_by_envelope_route():
    t = np.linspace(0.0, 1e-6, 4096)
    signal = np.exp(t / 2e-7) * np.cos(2.0 * np.pi * 1e9 * t)  # growing, unphysical for a passive ringdown
    with pytest.raises(ExtractionError):
        extract_envelope(t, signal)
