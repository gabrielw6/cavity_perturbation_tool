"""Validation suite for Module 3 (docs/module3_sample_equations.md Section 5)."""
import numpy as np
import pytest
from scipy import integrate

from cavity_perturbation import sample
from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.sample import (
    Cylinder,
    Material,
    Sample,
    SampleRegion,
    Slab,
    Sphere,
    gauss_legendre,
    orthonormal_frame,
)


# ===========================================================================
# Section 1: Material
# ===========================================================================

@pytest.mark.parametrize("eps_r,tan_d", [(4.5, 0.02), (10.0, 0.0), (2.1, 1e-4)])
def test_material_loss_tangent_round_trip(eps_r, tan_d):
    mat = Material.from_loss_tangent(eps_r, tan_d)
    assert mat.loss_tangent_e == pytest.approx(tan_d, abs=1e-12)


def test_material_loss_tangent_m_symmetric():
    mat = Material.from_loss_tangent(4.5, 0.02, mu_r=2.0, tan_delta_m=0.01)
    assert mat.loss_tangent_m == pytest.approx(0.01, abs=1e-12)


def test_material_re_eps_nonpositive_raises():
    mat = Material(eps=-1 - 0.1j, mu=1 - 0j)
    with pytest.raises(ValueError):
        _ = mat.loss_tangent_e


def test_material_is_passive():
    assert Material.from_loss_tangent(4.5, 0.02).is_passive
    assert not Material(eps=4.5 + 0.1j, mu=1 - 0j).is_passive  # eps'' < 0
    assert not Material(eps=-4.5 - 0.1j, mu=1 - 0j).is_passive  # eps' < 0
    assert not Material(eps=4.5 - 0.1j, mu=-1 - 0j).is_passive  # mu' < 0


# ===========================================================================
# Section 3.1/3.2: shared primitives
# ===========================================================================

@pytest.mark.parametrize("f,exact", [
    (lambda x: x**2, lambda a, b: (b**3 - a**3) / 3),
    (lambda x: x**4, lambda a, b: (b**5 - a**5) / 5),
])
def test_gauss_legendre_polynomials(f, exact):
    a, b = -1.3, 2.7
    x, w = gauss_legendre(10, a, b)
    assert np.sum(w * f(x)) == pytest.approx(exact(a, b), rel=1e-10)


def test_gauss_legendre_nonpolynomial_matches_quad():
    a, b = 0.0, 3.0
    x, w = gauss_legendre(20, a, b)
    approx = np.sum(w * np.sin(x))
    exact, _ = integrate.quad(np.sin, a, b)
    assert approx == pytest.approx(exact, rel=1e-10)


@pytest.mark.parametrize("n_hat", [[0, 0, 1], [1, 0, 0], [0.6, 0.8, 0]])
def test_orthonormal_frame_orthonormal_and_right_handed(n_hat):
    e1, e2, n = orthonormal_frame(np.array(n_hat, dtype=float))
    for v in (e1, e2, n):
        assert np.linalg.norm(v) == pytest.approx(1.0)
    assert np.dot(e1, e2) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(e1, n) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(e2, n) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(np.cross(n, e1), e2)


def test_orthonormal_frame_degenerate_branch_triggers():
    """n_hat = x_hat is near-parallel to the default reference u=x_hat --
    confirm the y-branch fallback still produces a valid frame."""
    e1, e2, n = orthonormal_frame(np.array([1.0, 0.0, 0.0]))
    assert np.dot(e1, n) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(e1) == pytest.approx(1.0)


# ===========================================================================
# Section 3.6 / Section 5: geometry validation, shared across shapes
# ===========================================================================

REGIONS = {
    "sphere": lambda: Sphere(center=[0.0, 0.0, 0.0], radius=0.02),
    "cylinder_generic": lambda: Cylinder(center=[0.1, -0.2, 0.3], axis=[0.0, 0.0, 1.0], radius=0.01, height=0.02),
    "cylinder_rod": lambda: Cylinder(center=[0, 0, 0], axis=[0.0, 1.0, 0.0], radius=0.001, height=0.02),
    "cylinder_disk": lambda: Cylinder(center=[0, 0, 0], axis=[0.0, 0.0, 1.0], radius=0.02, height=0.001),
    "slab_generic": lambda: Slab(center=[0, 0, 0], normal=[0, 0, 1], thickness=0.01, extent=(0.02, 0.02)),
    "slab_disk": lambda: Slab(center=[0, 0, 0], normal=[1, 0, 0], thickness=0.0005, extent=(0.02, 0.03)),
}


@pytest.mark.parametrize("name", REGIONS)
@pytest.mark.parametrize("n", [8, 64, 512])
def test_volume_consistency(name, n):
    region: SampleRegion = REGIONS[name]()
    pts, w = region.quadrature_points(n)
    assert np.sum(w) == pytest.approx(region.volume(), rel=1e-6)


@pytest.mark.parametrize("name", REGIONS)
def test_contains_quadrature_agreement(name):
    region: SampleRegion = REGIONS[name]()
    pts, w = region.quadrature_points(100)
    assert np.all(region.contains(pts))


def test_point_count_splitting_rule_no_collapse_for_small_n():
    for name, factory in REGIONS.items():
        region = factory()
        pts, w = region.quadrature_points(8)
        assert pts.shape[0] > 0, f"{name} produced zero quadrature points for n=8"
        assert w.shape[0] == pts.shape[0]


@pytest.mark.parametrize("name", ["cylinder_generic", "cylinder_rod", "cylinder_disk", "slab_generic", "slab_disk"])
def test_frame_transform_round_trip(name):
    region = REGIONS[name]()
    n_hat = region.axis if hasattr(region, "axis") else region.normal
    e1, e2, n = orthonormal_frame(n_hat)

    local = np.array([0.3, -0.7, 1.1])
    cavity_pt = sample._to_cavity(local.reshape(1, 3), region.center, e1, e2, n)
    back = sample._to_local(cavity_pt, region.center, e1, e2, n)
    back = np.array([back[0][0], back[1][0], back[2][0]])
    assert back == pytest.approx(local, abs=1e-10)


def test_cylinder_shape_kind_thresholds():
    assert REGIONS["cylinder_rod"]().shape_kind == "thin_rod"
    assert REGIONS["cylinder_disk"]().shape_kind == "thin_disk"
    assert REGIONS["cylinder_generic"]().shape_kind == "generic"


def test_slab_shape_kind_thresholds():
    assert REGIONS["slab_disk"]().shape_kind == "thin_disk"
    assert REGIONS["slab_generic"]().shape_kind == "generic"


def test_sphere_shape_kind():
    assert REGIONS["sphere"]().shape_kind == "sphere"


# ===========================================================================
# Section 2: depolarization factor
# ===========================================================================

def test_depolarization_sphere():
    eps_r = 4.5 - 0.3j
    region = Sphere(center=[0, 0, 0], radius=0.01)
    s = Sample(region=region, material=Material(eps=eps_r, mu=1 - 0j))
    ratio = s.depolarization_factor("E", field_direction=np.array([0, 0, 1.0]))
    assert ratio == pytest.approx(3.0 / (eps_r + 2.0))


def test_depolarization_thin_rod_axial_and_transverse():
    eps_r = 6.0 - 0.1j
    axis = np.array([0.0, 1.0, 0.0])
    region = Cylinder(center=[0, 0, 0], axis=axis, radius=0.001, height=0.02)
    s = Sample(region=region, material=Material(eps=eps_r, mu=1 - 0j))

    axial = s.depolarization_factor("E", field_direction=axis)
    assert axial == pytest.approx(1.0)  # N=0 -> no correction

    transverse = s.depolarization_factor("E", field_direction=np.array([1.0, 0.0, 0.0]))
    assert transverse == pytest.approx(2.0 / (eps_r + 1.0))


def test_depolarization_thin_disk_normal_and_tangential():
    eps_r = 3.2 - 0.05j
    normal = np.array([1.0, 0.0, 0.0])
    region = Slab(center=[0, 0, 0], normal=normal, thickness=0.0005, extent=(0.02, 0.03))
    s = Sample(region=region, material=Material(eps=eps_r, mu=1 - 0j))

    along_normal = s.depolarization_factor("E", field_direction=normal)
    assert along_normal == pytest.approx(1.0 / eps_r)

    tangential = s.depolarization_factor("E", field_direction=np.array([0.0, 1.0, 0.0]))
    assert tangential == pytest.approx(1.0)  # N=0 -> no correction


def test_depolarization_generic_shape_returns_unity():
    eps_r = 9.0 - 0.2j
    region = REGIONS["cylinder_generic"]()
    s = Sample(region=region, material=Material(eps=eps_r, mu=1 - 0j))
    ratio = s.depolarization_factor("E", field_direction=np.array([1.0, 0.0, 0.0]))
    assert ratio == pytest.approx(1.0)


def test_depolarization_oblique_field_falls_back_to_generic():
    """A 45-degree field must resolve to the generic fallback, not a
    scalar-N formula (Section 2.3)."""
    eps_r = 6.0 - 0.1j
    axis = np.array([0.0, 0.0, 1.0])
    region = Cylinder(center=[0, 0, 0], axis=axis, radius=0.001, height=0.02)
    s = Sample(region=region, material=Material(eps=eps_r, mu=1 - 0j))
    oblique_field = np.array([1.0, 0.0, 1.0])  # 45 degrees from axis
    ratio = s.depolarization_factor("E", field_direction=oblique_field)
    assert ratio == pytest.approx(1.0)


@pytest.mark.parametrize("theta_deg,expected", [(0.0, "aligned"), (9.0, "aligned"), (45.0, "oblique"),
                                                  (82.0, "perpendicular"), (90.0, "perpendicular")])
def test_angle_classification_boundaries(theta_deg, expected):
    axis = np.array([0.0, 0.0, 1.0])
    theta = np.radians(theta_deg)
    field = np.array([np.sin(theta), 0.0, np.cos(theta)])
    assert sample._classify_alignment(axis, field) == expected


def test_depolarization_n_sum_rule():
    """Section 2.5: the N values used must satisfy sum(N_i) = 1 for the
    underlying ellipsoid limit -- a standing check on the table itself."""
    assert sample._N_SPHERE * 3 == pytest.approx(1.0)
    assert sample._N_TABLE[("thin_rod", "aligned")] + 2 * sample._N_TABLE[("thin_rod", "perpendicular")] == pytest.approx(1.0)
    assert sample._N_TABLE[("thin_disk", "aligned")] + 2 * sample._N_TABLE[("thin_disk", "perpendicular")] == pytest.approx(1.0)


def test_depolarization_factor_invalid_field_type_raises():
    region = Sphere(center=[0, 0, 0], radius=0.01)
    s = Sample(region=region, material=Material(eps=4.5 - 0j, mu=1 - 0j))
    with pytest.raises(ValueError):
        s.depolarization_factor("B", field_direction=np.array([0, 0, 1.0]))


def test_zero_field_direction_falls_back_to_generic_not_a_crash():
    """A (numerically) zero field at the sample's center is a legitimate
    degeneracy -- e.g. a rod sitting exactly on a TM_010 cavity's axis,
    where H0 vanishes identically by symmetry -- and must fall back to the
    generic correction, not raise."""
    axis = np.array([0.0, 0.0, 1.0])
    region = Cylinder(center=[0, 0, 0], axis=axis, radius=0.001, height=0.02)
    s = Sample(region=region, material=Material(eps=6.0 - 0j, mu=1 - 0j))
    assert sample._classify_alignment(axis, np.zeros(3)) == "oblique"
    ratio = s.depolarization_factor("H", field_direction=np.zeros(3))
    assert ratio == pytest.approx(1.0)


def test_purely_imaginary_field_direction_classified_correctly():
    """A field whose Cartesian components share a common j-prefactor (the
    normal case for TE/TM fields with a real amplitude convention) must
    still resolve its real spatial direction, not get truncated to zero by
    a naive .real cast."""
    axis = np.array([0.0, 0.0, 1.0])
    imaginary_axial_field = 1j * axis  # purely imaginary, but spatially "aligned" with axis
    assert sample._classify_alignment(axis, imaginary_axial_field) == "aligned"

    imaginary_transverse_field = 1j * np.array([1.0, 0.0, 0.0])
    assert sample._classify_alignment(axis, imaginary_transverse_field) == "perpendicular"


# ===========================================================================
# Cross-check with Module 2: real SampleRegion satisfies FieldProvider contract
# ===========================================================================

def test_real_sample_region_satisfies_field_provider_contract():
    a, b, c = 0.03, 0.03, 0.03
    cav = RectangularCavity(a, b, c, ModeIndex("TE", (0, 1, 1)))
    field = AnalyticalField(cav)
    region = Sphere(center=[a / 2, b / 2, c / 2], radius=0.005)
    I_E = field.integrate_field_energy(region, "E")
    assert I_E > 0
