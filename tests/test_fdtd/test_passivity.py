"""docs/fdtd_module_plan.md Section 7.6 -- passivity regression. Per Section
0.5 this cannot fail by construction (non-negative sigma in a real-valued
update can't add energy) -- the test is a cheap guard against a
coefficient-sign typo, not the conceptual sign issue that bit Module 4."""
from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fdtd.model import FDTDModel
from cavity_perturbation.sample import Sphere

from ..conftest import assert_passive_material_never_improves_q

_RS_WALLS = 0.02  # Ohm, deliberately lossy walls so Q_wall is finite


def _coarse_model(cav, Rs_walls=_RS_WALLS):
    return FDTDModel(
        cav,
        Rs_walls=Rs_walls,
        cells_per_wavelength=8,
        min_cells_per_axis=8,
        record_periods=3.0,
    )


def test_fdtd_passive_material_never_improves_q():
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    model = _coarse_model(cav)
    region = Sphere(center=[0.015, 0.01, 0.0125], radius=0.0015)
    Q_wall = cav.Q_wall(_RS_WALLS)
    assert_passive_material_never_improves_q(
        model, region, eps_r=3.0, Q_wall=Q_wall, tan_deltas=(0.0, 1e-2, 0.1)
    )
