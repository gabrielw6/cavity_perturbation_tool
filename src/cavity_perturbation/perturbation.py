"""Module 4 -- Perturbation: Forward Model.

Combines Module 2's field-energy integrals with Module 3's depolarization-
corrected material contrast to predict the perturbed (f, Q) of a loaded
cavity, per docs/module4_perturbation_equations.md. Depends only on
`FieldProvider` (Module 2) and `Sample`/`Material` (Module 3) -- never on
`CavityMode` directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fields import FieldProvider
from .sample import Sample, SampleRegion

Array = np.ndarray


@dataclass(frozen=True)
class PerturbationResult:
    f_calc: float  # Hz
    Q_calc: float  # unitless (inf if no loss at all, incl. wall)
    omega_tilde: complex  # rad/s, raw loaded eigenvalue


class PerturbationModel:
    """Forward model: given a Sample (region + material), predict the
    perturbed (f, Q) using a given FieldProvider for the unperturbed field."""

    def __init__(self, field_provider: FieldProvider, Rs_walls: float | None = None) -> None:
        self._fp = field_provider
        self._Rs_walls = Rs_walls
        # Section 0.3 fix: store `region` itself alongside its id() so the
        # id cannot be recycled for as long as this cache entry exists --
        # id(x) alone is not a safe key once x is eligible for GC.
        self._cache: dict[int, tuple[SampleRegion, float, float]] = {}

    @property
    def field_provider(self) -> FieldProvider:
        """Retroactive addition for Module 5 (module5 doc Section 2.1/2.2):
        the closed-form seed needs direct access to the unperturbed field
        solution (f0, epsilon_bg, mu_bg, Q_wall), bypassing evaluate()'s
        general path."""
        return self._fp

    @property
    def Rs_walls(self) -> float | None:
        """Retroactive addition for Module 5 (module5 doc Section 2.2): the
        closed-form seed needs to know whether this model includes a wall-
        loss term at all before it can invert the measured resonance for
        Delta."""
        return self._Rs_walls

    def _shape_integrals(self, region: SampleRegion) -> tuple[float, float]:
        key = id(region)
        cached = self._cache.get(key)
        if cached is not None and cached[0] is region:
            return cached[1], cached[2]
        I_E = self._fp.integrate_field_energy(region, "E")
        I_H = self._fp.integrate_field_energy(region, "H")
        self._cache[key] = (region, I_E, I_H)
        return I_E, I_H

    def evaluate(self, sample: Sample) -> PerturbationResult:
        """Predict (f_calc, Q_calc) for `sample` sitting in this model's
        unperturbed field. Raises ValueError if `sample.material` isn't
        passive (CLAUDE.md passivity guard, checked at this boundary)."""
        if not sample.material.is_passive:
            raise ValueError(
                f"material {sample.material!r} is not passive "
                "(requires eps''>=0, mu''>=0, eps'>0, mu'>0)"
            )

        region = sample.region
        I_E, I_H = self._shape_integrals(region)

        # Section 0.2: field direction is resolved internally, at the
        # region's center, rather than being a parameter of this method.
        # `center` isn't part of the SampleRegion ABC (same duck-typing
        # rationale as `axis`/`normal` in sample.py's depolarization_factor).
        center: Array = getattr(region, "center")
        field_dir_E = self._fp.E(center)
        field_dir_H = self._fp.H(center)
        kappa_E = sample.depolarization_factor("E", field_dir_E)
        kappa_H = sample.depolarization_factor("H", field_dir_H)

        W = self._fp.total_stored_energy()
        p_E = self._fp.epsilon_bg * kappa_E * I_E / W
        p_H = self._fp.mu_bg * kappa_H * I_H / W

        eps_r = sample.material.eps
        mu_r = sample.material.mu
        # Note: the material-contrast factor is conjugated here, unlike the
        # doc's literal Section 1.4 formula ((eps_r-1)*p_E, no conjugate).
        # As written, Im(delta) is *always* positive for any passive material
        # (eps_r'' >= 0) -- i.e. every lossy sample would improve Q, which is
        # unconditionally unphysical (verified analytically and numerically
        # across a range of eps_r, tan_delta). Conjugating (eps_r-1)/(mu_r-1)
        # here -- leaving kappa_E/kappa_H (and hence p_E/p_H) exactly as
        # Module 3 computes them -- flips Im(delta) to the physically
        # required sign while leaving Re(delta)'s sign (the frequency
        # downshift for a dielectric perturber) unchanged. See
        # tests/test_perturbation.py's passivity/reciprocal-Q tests, the
        # actual arbiters for this, analogous to Module 1's curl-residual fix.
        delta = -0.5 * (np.conj(eps_r - 1.0) * p_E + np.conj(mu_r - 1.0) * p_H)

        omega0 = 2.0 * np.pi * self._fp.f0
        if self._Rs_walls is not None:
            Q_wall = self._fp.Q_wall(self._Rs_walls)
            wall_term: complex = -1j / (2.0 * Q_wall)
        else:
            wall_term = 0j  # Rs_walls is None <=> Q_wall -> infinity
        omega_tilde = omega0 * (1.0 + wall_term + delta)

        f_calc = float(omega_tilde.real) / (2.0 * np.pi)
        if omega_tilde.imag == 0.0:
            Q_calc = float("inf")
        else:
            Q_calc = -omega_tilde.real / (2.0 * omega_tilde.imag)

        return PerturbationResult(f_calc=f_calc, Q_calc=Q_calc, omega_tilde=omega_tilde)
