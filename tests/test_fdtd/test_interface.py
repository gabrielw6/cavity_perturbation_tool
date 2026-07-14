"""docs/fdtd_module_plan.md Section 7.7 -- interface compatibility,
independent of physics correctness: FDTDModel must be a drop-in
PerturbationModel sibling wherever Measurement.model holds one (Section 0.1).
Runs on a deliberately coarse/crude grid since this checks the *contract*,
not the accuracy."""
from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fdtd.model import FDTDModel
from cavity_perturbation.fields import FieldProvider
from cavity_perturbation.inverse import InverseSolver, Measurement
from cavity_perturbation.perturbation import PerturbationResult
from cavity_perturbation.sample import Material, Sample, Sphere


def _crude_model(cav, Rs_walls=None):
    return FDTDModel(
        cav,
        Rs_walls=Rs_walls,
        cells_per_wavelength=6,
        min_cells_per_axis=6,
        record_periods=2.0,
    )


def test_evaluate_returns_perturbation_result():
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    model = _crude_model(cav)
    region = Sphere(center=[0.015, 0.01, 0.0125], radius=0.0015)
    material = Material.from_loss_tangent(eps_r=3.0, tan_delta_e=0.01)
    result = model.evaluate(Sample(region=region, material=material))
    assert isinstance(result, PerturbationResult)
    assert result.f_calc > 0.0
    assert result.Q_calc > 0.0


def test_field_provider_and_rs_walls_present_and_correctly_typed():
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    model = _crude_model(cav, Rs_walls=0.02)
    assert isinstance(model.field_provider, FieldProvider)
    assert model.Rs_walls == 0.02

    model_no_walls = _crude_model(cav, Rs_walls=None)
    assert model_no_walls.Rs_walls is None


def test_measurement_and_closed_form_seed_path_runs_without_error():
    cav = RectangularCavity(0.03, 0.02, 0.025, ModeIndex("TE", (0, 1, 1)))
    model = _crude_model(cav, Rs_walls=0.02)
    region = Sphere(center=[0.015, 0.01, 0.0125], radius=0.0015)
    material = Material.from_loss_tangent(eps_r=3.0, tan_delta_e=0.01)
    result = model.evaluate(Sample(region=region, material=material))

    measurement = Measurement(
        model=model, region=region, f_meas=result.f_calc, Q_meas=result.Q_calc
    )
    solver = InverseSolver([measurement])
    # This exercises _closed_form_seed, which calls model.field_provider and
    # model.Rs_walls directly (Section 0.1) -- the check this test is for.
    fit = solver.fit()
    assert fit.eps.real > 0.0
