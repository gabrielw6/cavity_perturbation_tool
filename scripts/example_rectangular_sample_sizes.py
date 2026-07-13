#!/usr/bin/env python
"""Example: same rectangular cavity, same sample position (E-field max),
three different sphere sizes -- forward-simulate (f, Q) at each, then
recover the sample's permittivity via Module 5's InverseSolver (mu_r held
fixed at 1) and report the fit's formal uncertainty. A bigger sample
perturbs the resonance more for the same material, so the same measurement
precision (sigma_f, sigma_Q) constrains the recovered eps far more tightly
as the sample grows -- exactly the covariance/condition-number diagnostic
(module5 doc Section 4) at work.

Push the radius far enough (e.g. a large fraction of the cavity dimension)
and recovery still looks numerically exact -- that is *not* evidence the
measurement would be accurate at that size in reality. The forward model
and InverseSolver share the same first-order (small-sample) formula, so a
round trip is a self-consistency check of that formula, not a check of
whether the formula itself is still a valid approximation once the sample
is no longer small. See the printed warning below for when that regime is
crossed.

    python scripts/example_rectangular_sample_sizes.py
"""
from __future__ import annotations

import numpy as np

from cavity_perturbation.cavity import RectangularCavity
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.perturbation import PerturbationModel
from cavity_perturbation.sample import Material, Sphere

from _common import (
    eps_sigma,
    fmt_complex,
    perturbation_validity_warning,
    relative_error,
    round_trip,
    rs_from_conductivity,
)

A = B = C = 0.03
TRUE_MATERIAL = Material.from_loss_tangent(eps_r=4.5, tan_delta_e=0.01, mu_r=1.0)  # mu_r=1 -> fit_mu=False
SAMPLE_RADII = (2e-4, 1e-3, 8e-3)  # m


def main() -> None:
    cav = RectangularCavity(A, B, C)
    field = AnalyticalField(cav)
    Rs = rs_from_conductivity(cav.f0)
    model = PerturbationModel(field, Rs_walls=Rs)
    center = np.array([A / 2, B / 2, C / 2])  # TE_011 E-field max

    print(f"Rectangular cavity a=b=c={A} m, TE_011, f0={cav.f0 / 1e9:.4f} GHz, sample at {center.tolist()}\n")

    cavity_volume = A * B * C
    for radius in SAMPLE_RADII:
        region = Sphere(center=center, radius=radius)
        forward, fit = round_trip(model, region, TRUE_MATERIAL)
        sigma_re, sigma_im = eps_sigma(fit)
        volume_fraction = region.volume() / cavity_volume

        print(
            f"=== radius = {radius:.2e} m  (volume = {region.volume():.3e} m^3, "
            f"{volume_fraction:.1%} of the cavity) ==="
        )
        print(f"f_calc = {forward.f_calc / 1e9:.6f} GHz   Q_calc = {forward.Q_calc:.6e}")
        print(f"recovered eps_r = {fmt_complex(fit.eps)}  +/- ({sigma_re:.2g}, {sigma_im:.2g})j")
        print(
            f"recovery error = {relative_error(fit.eps, TRUE_MATERIAL.eps):.3e}   "
            f"condition_number = {fit.condition_number:.3e}   success = {fit.success}"
        )
        warning = perturbation_validity_warning(cav.f0, forward)
        if warning:
            print(warning)
        print()


if __name__ == "__main__":
    main()
