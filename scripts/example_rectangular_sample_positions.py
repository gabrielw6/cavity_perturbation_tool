#!/usr/bin/env python
"""Example: same rectangular cavity, same sphere sample, three different
sample positions (E-field max, an intermediate point, and a weak-field spot
near a wall) -- forward-simulate (f, Q) at each, then recover the sample's
permittivity via Module 5's InverseSolver (mu_r held fixed at 1) and report
the fit's formal uncertainty. Sensitivity to the sample scales with the
local |E|^2, so the same measurement precision (sigma_f, sigma_Q) yields a
tighter or looser recovered-eps error bar depending on where the sample
sits -- this is exactly what the covariance/condition-number diagnostic
(module5 doc Section 4) is for.

    python scripts/example_rectangular_sample_positions.py
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
SAMPLE_RADIUS = 1e-3  # m
TRUE_MATERIAL = Material.from_loss_tangent(eps_r=4.5, tan_delta_e=0.01, mu_r=1.0)  # mu_r=1 -> fit_mu=False

POSITIONS = {
    "E-field max (center)": [A / 2, B / 2, C / 2],
    "intermediate": [A / 2, B / 2, 3 * C / 4],
    "weak field, near wall": [A / 2, 0.002, 0.002],
}


def main() -> None:
    cav = RectangularCavity(A, B, C)
    field = AnalyticalField(cav)
    Rs = rs_from_conductivity(cav.f0)
    model = PerturbationModel(field, Rs_walls=Rs)

    print(f"Rectangular cavity a=b=c={A} m, TE_011, f0={cav.f0 / 1e9:.4f} GHz\n")

    for name, pos in POSITIONS.items():
        position = np.array(pos)
        e_mag2 = float(np.sum(np.abs(field.E(position)) ** 2))
        region = Sphere(center=position, radius=SAMPLE_RADIUS)

        forward, fit = round_trip(model, region, TRUE_MATERIAL)
        sigma_re, sigma_im = eps_sigma(fit)

        print(f"=== {name}: position={pos}, |E|^2={e_mag2:.4e} ===")
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
