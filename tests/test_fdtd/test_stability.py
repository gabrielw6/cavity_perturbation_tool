"""docs/fdtd_module_plan.md Section 7.2 -- CFL enforcement, standalone."""
import numpy as np
import pytest
from scipy import constants

from cavity_perturbation.fdtd.stability import (
    CFLViolationError,
    assert_stable,
    cfl_limit,
    stable_time_step,
)

_EPS0, _MU0 = constants.epsilon_0, constants.mu_0
_C = constants.c


@pytest.mark.parametrize(
    "cell_size",
    [
        (1e-3, 1e-3, 1e-3),
        (1e-3, 2e-3, 4e-3),  # anisotropic
        (5e-4, 5e-4, 2e-2),  # highly elongated
        (2e-2, 5e-4, 5e-4),
    ],
)
def test_stable_time_step_satisfies_cfl_bound(cell_size):
    dt = stable_time_step(cell_size, _EPS0, _MU0)
    limit = cfl_limit(cell_size, _EPS0, _MU0)
    assert 0.0 < dt <= limit


def test_cfl_limit_vacuum_matches_known_formula():
    dx = dy = dz = 1e-3
    limit = cfl_limit((dx, dy, dz), _EPS0, _MU0)
    expected = 1.0 / (_C * np.sqrt(3.0) / dx)
    assert limit == pytest.approx(expected, rel=1e-9)


def test_finer_grid_gives_smaller_stable_dt():
    dt_coarse = stable_time_step((2e-3, 2e-3, 2e-3), _EPS0, _MU0)
    dt_fine = stable_time_step((1e-3, 1e-3, 1e-3), _EPS0, _MU0)
    assert dt_fine < dt_coarse


def test_assert_stable_accepts_dt_within_bound():
    cell_size = (1e-3, 1e-3, 1e-3)
    dt = 0.5 * cfl_limit(cell_size, _EPS0, _MU0)
    assert_stable(dt, cell_size, _EPS0, _MU0)  # no raise


def test_assert_stable_rejects_deliberately_over_large_dt():
    cell_size = (1e-3, 1e-3, 1e-3)
    limit = cfl_limit(cell_size, _EPS0, _MU0)
    with pytest.raises(CFLViolationError):
        assert_stable(2.0 * limit, cell_size, _EPS0, _MU0)


def test_stable_time_step_rejects_invalid_safety_factor():
    cell_size = (1e-3, 1e-3, 1e-3)
    with pytest.raises(ValueError):
        stable_time_step(cell_size, _EPS0, _MU0, safety_factor=1.0)
    with pytest.raises(ValueError):
        stable_time_step(cell_size, _EPS0, _MU0, safety_factor=0.0)
