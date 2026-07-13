"""Shared test utilities for cavity_perturbation's test suite."""
from cavity_perturbation.sample import Material, Sample


def assert_passive_material_never_improves_q(model, region, eps_r, Q_wall, tan_deltas=(0.0, 1e-3, 1e-2, 0.1)):
    """A lossy passive material can only degrade Q, never improve it --
    shared between test_perturbation.py (Module 4) and test_ritz.py (the
    Rayleigh-Ritz module), since both `PerturbationModel` and
    `RitzCorrectedModel` need the same conjugate fix to satisfy this
    (docs/ritz_module_plan.md Section 2.3 cites this exact check as the
    fastest way to catch a conjugate regression)."""
    for tan_delta in tan_deltas:
        material = Material.from_loss_tangent(eps_r, tan_delta)
        sample = Sample(region=region, material=material)
        result = model.evaluate(sample)
        assert result.Q_calc <= Q_wall * (1.0 + 1e-9)
