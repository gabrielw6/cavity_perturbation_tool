"""FDTD Module -- stability.py: CFL time-step computation and enforcement.

Delta t is always computed from grid spacing and background wave speed; it
is never a caller-supplied value (docs/fdtd_module_plan.md Section 5.2) --
`stable_time_step` is the only way to obtain a time step anywhere in this
sub-package, and it always applies the safety factor itself. This is a hard
invariant, checked and enforced here, not left as documentation.
"""
from __future__ import annotations

import numpy as np

_SAFETY_FACTOR = 0.99


class CFLViolationError(Exception):
    """A time step exceeds the CFL stability bound."""


def cfl_limit(cell_size: tuple[float, float, float], epsilon_bg: complex, mu_bg: complex) -> float:
    """Section 5.2: dt <= 1 / (c' * sqrt(dx^-2 + dy^-2 + dz^-2)), c' the
    background (fastest) wave speed -- a higher-eps_r sample region is
    always slower and never binds, so only the background medium enters
    here. `epsilon_bg`/`mu_bg` are absolute SI values (CavityMode's own
    convention); only their real part sets the (lossless) wave speed."""
    dx, dy, dz = cell_size
    c_prime = 1.0 / np.sqrt(float(np.real(epsilon_bg)) * float(np.real(mu_bg)))
    return 1.0 / (c_prime * np.sqrt(dx**-2 + dy**-2 + dz**-2))


def assert_stable(
    dt: float, cell_size: tuple[float, float, float], epsilon_bg: complex, mu_bg: complex
) -> None:
    """Enforcement point (Section 5.2): raises rather than silently
    accepting a dt that violates the CFL bound -- the function Section
    7.2's "deliberately over-large dt is rejected" regression test drives
    directly."""
    limit = cfl_limit(cell_size, epsilon_bg, mu_bg)
    if dt > limit:
        raise CFLViolationError(
            f"dt={dt!r} exceeds the CFL stability limit {limit!r} for cell_size={cell_size!r}"
        )


def stable_time_step(
    cell_size: tuple[float, float, float],
    epsilon_bg: complex,
    mu_bg: complex,
    safety_factor: float = _SAFETY_FACTOR,
) -> float:
    """The sole source of Delta t in this sub-package -- always the CFL
    bound times `safety_factor` (< 1), never a value chosen independently
    by a caller (stepper.py has no dt constructor parameter)."""
    if not (0.0 < safety_factor < 1.0):
        raise ValueError(f"safety_factor must be in (0, 1), got {safety_factor!r}")
    dt = safety_factor * cfl_limit(cell_size, epsilon_bg, mu_bg)
    assert_stable(dt, cell_size, epsilon_bg, mu_bg)
    return dt
