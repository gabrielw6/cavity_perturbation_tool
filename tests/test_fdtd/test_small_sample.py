"""docs/fdtd_module_plan.md Section 7.5 -- small-sample agreement with
Module 4, the primary cross-validation that assembly, stepping, and
extraction are all correct together.

The sample must be resolved by several grid cells (not just small relative
to the cavity) for a meaningful comparison -- a sample only 1-2 cells wide
is dominated by rasterization error, not physics, independent of how well
`cells_per_wavelength` resolves the cavity mode itself (found directly: a
1.5mm-radius sample at a ~3mm cell size gave ~196% Q error purely from being
under-resolved; a 2.5mm-radius sample at a ~2.1mm cell size -- several cells
across -- brought that down to a few percent)."""
import pytest

from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fdtd.model import FDTDModel
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.perturbation import PerturbationModel
from cavity_perturbation.sample import Material, Sample, Sphere


def test_fdtd_agrees_with_perturbation_model_for_small_well_resolved_sample():
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    region = Sphere(center=[0.015, 0.01, 0.0125], radius=0.0025)
    material = Material.from_loss_tangent(eps_r=2.0, tan_delta_e=0.005)
    sample = Sample(region=region, material=material)

    reference = PerturbationModel(AnalyticalField(cav)).evaluate(sample)

    fdtd = FDTDModel(cav, cells_per_wavelength=14, min_cells_per_axis=8, record_periods=6.0)
    result = fdtd.evaluate(sample)

    assert result.f_calc == pytest.approx(reference.f_calc, rel=0.05)
    assert result.Q_calc == pytest.approx(reference.Q_calc, rel=0.3)
