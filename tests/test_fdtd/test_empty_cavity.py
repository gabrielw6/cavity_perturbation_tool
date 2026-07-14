"""docs/fdtd_module_plan.md Section 7.4 -- empty-cavity physics check, the
"strong" one: no sample, isolating whether ringdown extraction and wall-loss
modeling are correct on their own, touching neither the perturbation formula
nor a real sample.

FDTDModel.evaluate always takes a Sample, so "no sample" is approximated by
a vanishingly small, background-matched (eps_r=1, lossless) sphere -- its
rasterized `sample_interior` mask is empty or a single negligible cell,
physically indistinguishable from no sample at all.
"""
import pytest

from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fdtd.model import FDTDModel
from cavity_perturbation.sample import Material, Sample, Sphere

_NEGLIGIBLE_SAMPLE = Sample(
    region=Sphere(center=[0.001, 0.001, 0.001], radius=1e-9),
    material=Material.from_loss_tangent(eps_r=1.0, tan_delta_e=0.0),
)


def test_empty_cavity_f0_converges_as_grid_refines():
    # Rectangular: no staircasing (the grid's own outer boundary IS the PEC
    # wall), so the remaining error is standard Yee-grid numerical
    # dispersion, which shrinks with resolution.
    #
    # Decisive regression test for the near-wall tangential-E PEC pin
    # (grid/rasterize.py's tangential_wall_pin, stepper.py's use of it):
    # without it, CavityMode.contains()'s inclusive bounds let a
    # wall-coincident tangential-E point evolve freely instead of staying
    # pinned at zero, driving f_calc ~10% high and giving a spuriously
    # finite Q_calc (no wall/sample loss configured here, so a genuinely
    # correct run has no decay mechanism at all -- the residual finite
    # Q_calc below is entirely a finite-record extraction-noise floor, per
    # 6.3's own documented limitation, not the wall-pin bug). Verified
    # directly: this test's error bounds are well below what the buggy
    # version produced at either resolution (~4.2% and ~0.63%).
    cav = RectangularCavity(0.03, 0.04, 0.05, ModeIndex("TE", (0, 1, 1)))

    coarse = FDTDModel(cav, cells_per_wavelength=6, min_cells_per_axis=6)
    fine = FDTDModel(cav, cells_per_wavelength=16, min_cells_per_axis=6)

    f_coarse = coarse.evaluate(_NEGLIGIBLE_SAMPLE).f_calc
    f_fine = fine.evaluate(_NEGLIGIBLE_SAMPLE).f_calc

    err_coarse = abs(f_coarse - cav.f0) / cav.f0
    err_fine = abs(f_fine - cav.f0) / cav.f0

    assert err_coarse < 0.01
    assert err_fine < 0.005
    assert err_fine < err_coarse


def test_empty_cavity_wall_loss_q_matches_analytic_q_wall_within_first_pass_tolerance():
    # Section 4.1: wall loss is never time-stepped, only combined back in
    # analytically (Section 6.2) -- the FDTD run's own extraction noise
    # floor must sit well above Q_wall for that combination to be accurate,
    # which for a first-pass (short-record, non-PML, staircasing-included)
    # implementation is a real, documented accuracy limit, not exactness
    # (docs/fdtd_module_plan.md Section 6.2's note on this) -- verified
    # directly: excluding Q_wall from record-length sizing made this WORSE
    # (~90% low) since the extraction noise floor then sat below Q_wall
    # entirely; including it (current model.py) recovers the right order of
    # magnitude but not tight numeric agreement.
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    Rs_walls = 0.3  # deliberately lossy walls -> a modest, fast-to-resolve Q_wall
    Q_wall_true = cav.Q_wall(Rs_walls)

    model = FDTDModel(cav, Rs_walls=Rs_walls, cells_per_wavelength=10, min_cells_per_axis=8)
    result = model.evaluate(_NEGLIGIBLE_SAMPLE)

    assert result.Q_calc == pytest.approx(Q_wall_true, rel=0.5)
    assert result.Q_calc < Q_wall_true  # Section 0.5: never improves on Q_wall
