"""FDTD Module -- source.py: mode-shaped soft-source excitation and probe
placement, per docs/fdtd_module_plan.md Section 3.
"""
from __future__ import annotations

import numpy as np

from ..cavity import CavityMode
from ..fields import FieldProvider
from .grid.yee import E_COMPONENTS, YeeGrid

Array = np.ndarray

_PROBE_CANDIDATES_PER_AXIS = 15


def _extract_real_spatial_profile(complex_by_component: dict[str, Array]) -> dict[str, Array]:
    """Ex, Ey, Ez of one Module-1 mode's E field share a single global
    complex phase (module3 doc Section 2.3's insight, generalized here from
    one point to the whole grid): E purely imaginary for a TE mode's
    transverse components, for instance, with no further r-dependence in
    that phase. Find that shared phase from whichever component/grid-point
    has the largest magnitude, rotate it out of *every* component's *whole*
    array, and keep the real part -- recovering a real spatial mode shape
    (arbitrary overall scale) usable as a real-valued FDTD source, rather
    than naively taking `.real` first and silently zeroing out a mode
    that happens to be purely imaginary."""
    all_vals = np.concatenate([arr.reshape(-1) for arr in complex_by_component.values()])
    idx = int(np.argmax(np.abs(all_vals)))
    ref = all_vals[idx]
    if np.abs(ref) < 1e-300:
        raise ValueError("mode field is (numerically) identically zero on this grid")
    phase = np.angle(ref)
    return {c: (arr * np.exp(-1j * phase)).real for c, arr in complex_by_component.items()}


def build_modal_source(grid: YeeGrid, field_provider: FieldProvider) -> dict[str, Array]:
    """Section 3 excitation profile: `field_provider.E` sampled at each E
    component's own staggered coordinates (only that Cartesian component of
    the returned vector is kept, at that component's own location -- same
    per-component staggered evaluation discipline as grid/rasterize.py),
    reduced to a real spatial shape via `_extract_real_spatial_profile`.
    Used as a soft (additive) source: `field_provider.E`'s own arbitrary
    scale sets the pulse's spatial-shape scale, not the pulse amplitude
    itself, which the caller supplies separately (`gaussian_modulated_pulse`).
    """
    complex_by_component: dict[str, Array] = {}
    for i, component in enumerate(E_COMPONENTS):
        coords = grid.component_coords(component)
        complex_by_component[component] = field_provider.E(coords)[..., i].reshape(grid.shape)
    return _extract_real_spatial_profile(complex_by_component)


def gaussian_pulse_sigma_t(bandwidth_hz: float) -> float:
    """Temporal standard deviation for a Gaussian envelope whose own
    spectral envelope has standard deviation `bandwidth_hz` -- a Gaussian's
    Fourier transform is Gaussian, sigma_t = 1/(2*pi*sigma_f) (Section 3:
    "bandwidth wide enough to cover the expected perturbed frequency but
    narrow enough not to strongly excite neighboring modes")."""
    if bandwidth_hz <= 0.0:
        raise ValueError(f"bandwidth_hz must be > 0, got {bandwidth_hz!r}")
    return 1.0 / (2.0 * np.pi * bandwidth_hz)


def gaussian_modulated_pulse(t: Array, f0: float, t0: float, sigma_t: float) -> Array:
    """Section 3 excitation waveform: a Gaussian-modulated sinusoid centered
    at `f0`, peaking at `t0`, with temporal width `sigma_t`
    (`gaussian_pulse_sigma_t`). Dimensionless -- the caller scales it by the
    spatial profile from `build_modal_source` and an overall pulse
    amplitude."""
    t = np.asarray(t, dtype=float)
    envelope = np.exp(-0.5 * ((t - t0) / sigma_t) ** 2)
    return envelope * np.cos(2.0 * np.pi * f0 * (t - t0))


def choose_probe_point(
    cavity_mode: CavityMode,
    field_provider: FieldProvider,
    n_per_axis: int = _PROBE_CANDIDATES_PER_AXIS,
) -> tuple[Array, str]:
    """Section 3 probing: a fixed point + dominant E component where that
    mode's field is large, avoiding nodes. Scans a deterministic structured
    grid of candidate points inside `cavity_mode.bounding_box()` (filtered
    by `contains()` -- no mesh, no CAD, Section 0.3) and returns the
    location/component of largest |E|."""
    rmin, rmax = cavity_mode.bounding_box()
    axes = [np.linspace(rmin[i], rmax[i], n_per_axis) for i in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    candidates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    inside = np.asarray(cavity_mode.contains(candidates))
    candidates = candidates[inside]
    if candidates.shape[0] == 0:
        raise ValueError("no candidate points landed inside the cavity -- check bounding_box()/contains()")

    e_vals = field_provider.E(candidates)
    mags = np.abs(e_vals)
    flat_idx = int(np.argmax(mags))
    point_idx, comp_idx = np.unravel_index(flat_idx, mags.shape)
    return candidates[point_idx], E_COMPONENTS[comp_idx]
