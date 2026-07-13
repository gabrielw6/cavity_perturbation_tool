#!/usr/bin/env python
"""Example: for each Module 1 cavity type (rectangular, cylindrical,
coaxial), drop a small dielectric sphere at the E-field maximum, predict the
loaded (f, Q) with Module 4, then run Module 5's InverseSolver on that
simulated measurement alone to recover the sample's permittivity -- a full
Module 1->5 round trip for every geometry the package supports.

    python scripts/example_roundtrip_all_cavities.py
"""
from __future__ import annotations

from cavity_perturbation.cavity import CavityMode, CoaxialCavity, CylindricalCavity, RectangularCavity
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.perturbation import PerturbationModel
from cavity_perturbation.sample import Material, Sphere

from _common import (
    eps_sigma,
    field_max_position,
    fmt_complex,
    perturbation_validity_warning,
    relative_error,
    round_trip,
    rs_from_conductivity,
)

SAMPLE_RADIUS = 1e-3  # m
TRUE_MATERIAL = Material.from_loss_tangent(eps_r=4.5, tan_delta_e=0.01, mu_r=1.0)  # mu_r=1 -> fit_mu=False


def run(name: str, cav: CavityMode) -> None:
    field = AnalyticalField(cav)
    Rs = rs_from_conductivity(cav.f0)
    model = PerturbationModel(field, Rs_walls=Rs)

    position = field_max_position(cav, field, margin=2.0 * SAMPLE_RADIUS)
    region = Sphere(center=position, radius=SAMPLE_RADIUS)

    forward, fit = round_trip(model, region, TRUE_MATERIAL)
    sigma_re, sigma_im = eps_sigma(fit)

    print(f"=== {name} ===")
    print(f"f0 = {cav.f0 / 1e9:.4f} GHz   Q_wall = {cav.Q_wall(Rs):.4e}   sample at {position.tolist()}")
    df = forward.f_calc - cav.f0
    print(f"f_calc = {forward.f_calc / 1e9:.6f} GHz   (df/f0 = {df / cav.f0:+.4e})")
    print(f"Q_calc = {forward.Q_calc:.6e}")
    print(f"true material:   eps_r = {fmt_complex(TRUE_MATERIAL.eps)}")
    print(f"recovered:       eps_r = {fmt_complex(fit.eps)}  +/- ({sigma_re:.2g}, {sigma_im:.2g})j")
    print(
        f"recovery error:  {relative_error(fit.eps, TRUE_MATERIAL.eps):.3e}   "
        f"condition_number = {fit.condition_number:.3e}   success = {fit.success}"
    )
    warning = perturbation_validity_warning(cav.f0, forward)
    if warning:
        print(warning)
    print()


def main() -> None:
    run("Rectangular (TE_011)", RectangularCavity(0.03, 0.03, 0.03))
    run("Cylindrical (TM_010)", CylindricalCavity(0.02, 0.03))
    run("Coaxial (TEM, q=1)", CoaxialCavity(0.01, 0.023, 0.5))


if __name__ == "__main__":
    main()
