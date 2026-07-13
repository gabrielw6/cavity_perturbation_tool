#!/usr/bin/env python
"""Example: verify the Rayleigh-Ritz module (`ritz.py`) against Module 4's
`PerturbationModel`, across exactly the configurations
docs/ritz_module_plan.md Section 7 says are meaningful to compare -- and
none of the ones its own inline corrections say are *not* guaranteed (see
CLAUDE.md's ritz_module_plan.md entry and the ritz-module-findings memory).

Order matters, same as the doc's own test plan: confirm Ritz is internally
trustworthy (basis-size self-convergence) before trusting any comparison
against Module 4.

    python scripts/example_verify_ritz_vs_perturbation.py
"""
from __future__ import annotations

import numpy as np

from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.inverse import point_dipole_filling_factors
from cavity_perturbation.perturbation import PerturbationModel
from cavity_perturbation.ritz import RitzCorrectedModel, nearest_basis_modes
from cavity_perturbation.sample import Cylinder, Material, Sample, Sphere

A, B, C = 0.03, 0.025, 0.04  # non-cubic -- avoids accidental exact mode-frequency degeneracy
MODE = ModeIndex("TE", (0, 1, 1))
POSITION = [A / 2, 0.8 * B / 2, 1.3 * C / 2]  # off-axis, so the sample genuinely couples basis modes
MATERIAL = Material.from_loss_tangent(eps_r=4.5, tan_delta_e=0.01)  # mu_r=1 -> RitzCorrectedModel's only supported case


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def verify_basis_size_self_convergence() -> None:
    """Step 1 (docs/ritz_module_plan.md Section 7, "order matters"):
    confirm Ritz's own prediction stabilizes as the basis grows, before
    trusting any comparison to PerturbationModel below."""
    section("1. Basis-size self-convergence (Ritz vs. itself)")
    region = Sphere(center=POSITION, radius=1e-3)
    sample = Sample(region=region, material=MATERIAL)

    prev_f = None
    for n_basis in (1, 3, 5, 9):
        basis = nearest_basis_modes(RectangularCavity, (A, B, C), MODE, n_basis=n_basis)
        rmodel = RitzCorrectedModel(basis, Rs_walls=None)
        r = rmodel.evaluate(sample)
        shift = "" if prev_f is None else f"  (df/f from previous N = {(r.f_calc - prev_f) / r.f_calc:+.2e})"
        print(f"N={n_basis:2d}  f_calc={r.f_calc / 1e9:.9f} GHz  Q_calc={r.Q_calc:11.4e}{shift}")
        prev_f = r.f_calc


def verify_n1_reduction_to_point_dipole_formula() -> None:
    """Bare N=1 Ritz (no mode mixing) should reduce EXACTLY to Module 4's
    *uncorrected* (kappa=1) point-dipole formula -- verified directly
    against that raw formula, not against PerturbationModel's Sphere answer
    (which applies a depolarization correction Section 2.3 says Ritz must
    NOT apply). This isolates matrix assembly + the conjugate fix from
    anything eigensolve- or mode-tracking-related."""
    section("2. N=1 Ritz vs. the raw (kappa=1) point-dipole formula")
    cav = RectangularCavity(A, B, C, MODE)
    field = AnalyticalField(cav)
    pmodel = PerturbationModel(field, Rs_walls=None)

    region = Sphere(center=POSITION, radius=1e-4)
    sample = Sample(region=region, material=MATERIAL)

    basis = nearest_basis_modes(RectangularCavity, (A, B, C), MODE, n_basis=1)
    rmodel = RitzCorrectedModel(basis, Rs_walls=None)
    r_ritz = rmodel.evaluate(sample)
    delta_ritz = r_ritz.omega_tilde / (2.0 * np.pi * cav.f0) - 1.0

    p_E, _p_H = point_dipole_filling_factors(pmodel, region)
    delta_manual = -0.5 * np.conj(MATERIAL.eps - 1.0) * p_E

    print(f"delta from N=1 RitzCorrectedModel : {delta_ritz!r}")
    print(f"delta from manual kappa=1 formula : {delta_manual!r}")
    print(f"relative difference: {abs(delta_ritz - delta_manual) / abs(delta_manual):.3e}")


def verify_generic_shape_agreement() -> None:
    """For a 'generic'-shaped region, kappa_E=kappa_H=1 for BOTH models
    (Module 3's point-dipole fallback), so this is the strongest available
    cross-check: basis selection, matrix assembly, the eigensolve, and mode
    tracking must all be correct together for the two completely different
    computational routes to agree this closely."""
    section("3. Small-sample agreement, 'generic' shape (kappa_E=1 for both)")
    cav = RectangularCavity(A, B, C, MODE)
    field = AnalyticalField(cav)
    pmodel = PerturbationModel(field, Rs_walls=None)
    basis = nearest_basis_modes(RectangularCavity, (A, B, C), MODE, n_basis=5)
    rmodel = RitzCorrectedModel(basis, Rs_walls=None)

    region = Cylinder(center=POSITION, axis=[0, 0, 1], radius=1e-4, height=1e-4)
    print(f"region.shape_kind = {region.shape_kind!r}")
    sample = Sample(region=region, material=MATERIAL)

    r_p = pmodel.evaluate(sample)
    r_r = rmodel.evaluate(sample)
    print(f"PerturbationModel : f_calc={r_p.f_calc / 1e9:.9f} GHz   Q_calc={r_p.Q_calc:.6e}")
    print(f"RitzCorrectedModel: f_calc={r_r.f_calc / 1e9:.9f} GHz   Q_calc={r_r.Q_calc:.6e}")
    rel_f = abs(r_r.f_calc - r_p.f_calc) / abs(r_p.f_calc)
    rel_invQ = abs(1.0 / r_r.Q_calc - 1.0 / r_p.Q_calc) / abs(1.0 / r_p.Q_calc)
    print(f"relative difference: f_calc={rel_f:.3e}   1/Q_calc={rel_invQ:.3e}")


def verify_sample_size_correction_sweep() -> None:
    """The 'sample-size correction' study itself (docs/ritz_module_plan.md
    Section 7.3): for a Sphere (kappa_E != 1, so the two models are NOT
    expected to agree exactly -- see the doc's Section 7.4 correction),
    sweep the sample size upward and report where PerturbationModel's
    single-mode + depolarization-factor answer and RitzCorrectedModel's
    multi-mode answer diverge past 1% -- the threshold the original
    project's sample-size-correction study set out to find."""
    section("4. Sample-size-correction sweep (Sphere, N=5)")
    cav = RectangularCavity(A, B, C, MODE)
    field = AnalyticalField(cav)
    pmodel = PerturbationModel(field, Rs_walls=None)
    basis = nearest_basis_modes(RectangularCavity, (A, B, C), MODE, n_basis=5)
    rmodel = RitzCorrectedModel(basis, Rs_walls=None)
    cavity_volume = A * B * C

    print(f"{'radius [m]':>12}  {'V_s/V_cavity':>13}  {'rel. diff (f_calc)':>19}")
    crossed_at: float | None = None
    for radius in np.geomspace(5e-5, 6e-3, 10):
        region = Sphere(center=POSITION, radius=radius)
        sample = Sample(region=region, material=MATERIAL)
        r_p = pmodel.evaluate(sample)
        r_r = rmodel.evaluate(sample)
        volume_fraction = region.volume() / cavity_volume
        rel_diff = abs(r_r.f_calc - r_p.f_calc) / abs(r_p.f_calc)
        print(f"{radius:12.2e}  {volume_fraction:13.4%}  {rel_diff:19.3e}")
        if crossed_at is None and rel_diff > 0.01:
            crossed_at = radius

    if crossed_at is not None:
        print(f"\n-> crosses 1% divergence near radius = {crossed_at:.2e} m "
              f"(V_s/V_cavity = {Sphere(center=POSITION, radius=crossed_at).volume() / cavity_volume:.3%})")
    else:
        print("\n-> stayed under 1% divergence across the swept range")
    print(
        "Note: this growing gap reflects PerturbationModel's classical sphere-depolarization\n"
        "correction (kappa_E, size-independent) versus RitzCorrectedModel's multi-mode field\n"
        "mixing (which grows with sample size) -- it is not expected to shrink back down as the\n"
        "Ritz basis grows further (see docs/ritz_module_plan.md Section 7.4's correction)."
    )


def main() -> None:
    verify_basis_size_self_convergence()
    verify_n1_reduction_to_point_dipole_formula()
    verify_generic_shape_agreement()
    verify_sample_size_correction_sweep()


if __name__ == "__main__":
    main()
