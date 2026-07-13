#!/usr/bin/env python
"""Simulate a cavity-perturbation measurement: build a Module 1 cavity, drop
a standard sample (sphere / thin rod / thin disk -- the canonical shapes
Module 3's depolarization-factor table covers) into it, and predict the
loaded (f, Q) via Module 4's forward model. Built entirely on the public
CavityMode / FieldProvider / Sample interfaces -- no cavity-type-specific
code lives in this script beyond building the requested geometry.

Usage:
    python scripts/simulate_perturbation.py rectangular --a 0.03 --b 0.03 --c 0.03 \\
        --mode-kind TE --mode-indices 0 1 1 \\
        --shape sphere --sample-radius 1e-3 \\
        --sample-eps-r 4.5 --sample-tan-delta-e 0.01

    python scripts/simulate_perturbation.py cylindrical --radius 0.02 --length 0.03 \\
        --mode-kind TM --mode-indices 0 1 0 \\
        --shape rod --sample-radius 2e-4 --sample-orientation perpendicular \\
        --sample-eps-r 9.0 --sample-tan-delta-e 0.002

    python scripts/simulate_perturbation.py coaxial --r-inner 0.01 --r-outer 0.023 \\
        --length 0.5 --shape disk --sample-extent 2e-3 2e-3 \\
        --sample-eps-r 2.1 --sample-tan-delta-e 0.0005 --no-wall-loss

Add --plot to also draw the wall-only vs. loaded resonance (Lorentzian)
curves (requires matplotlib, pip install -e ".[viz]").

Add --compare-ritz to also predict (f, Q) via the Rayleigh-Ritz multi-mode
model (docs/ritz_module_plan.md) and report its difference from Module 4's
single-mode + depolarization-factor prediction -- growing disagreement as
the sample grows is the "sample-size correction" effect that module exists
to quantify. Only supports non-magnetic samples (--sample-mu-r 1, the
default); with --plot, the Ritz curve is added to the figure too.
"""
from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy import constants

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from cavity_perturbation.cavity import (
    CavityMode,
    CoaxialCavity,
    CylindricalCavity,
    ModeIndex,
    RectangularCavity,
)
from cavity_perturbation.fields import AnalyticalField, FieldProvider
from cavity_perturbation.perturbation import PerturbationModel, PerturbationResult
from cavity_perturbation.ritz import RitzCorrectedModel, nearest_basis_modes
from cavity_perturbation.sample import (
    Cylinder,
    Material,
    Sample,
    SampleRegion,
    Slab,
    Sphere,
    orthonormal_frame,
    real_field_direction,
)

COPPER_CONDUCTIVITY = 5.8e7  # S/m
_OUTSIDE_FRACTION_TOLERANCE = 0.01  # max fraction of a sample allowed outside the cavity's domain


def _rs_from_conductivity(f0: float, sigma: float) -> float:
    return float(np.sqrt(np.pi * f0 * constants.mu_0 / sigma))


# --- Cavity construction (mirrors scripts/visualize_cavity.py) ------------

def cavity_type_and_args(
    args: argparse.Namespace,
) -> tuple[Callable[..., CavityMode], tuple[float, ...], ModeIndex]:
    """(constructor, positional geometry args, mode) -- shared by
    build_cavity (one instance) and --compare-ritz's basis construction
    (several instances of the same geometry, different modes)."""
    if args.cavity == "rectangular":
        return RectangularCavity, (args.a, args.b, args.c), ModeIndex(args.mode_kind, tuple(args.mode_indices))
    if args.cavity == "cylindrical":
        return CylindricalCavity, (args.radius, args.length), ModeIndex(args.mode_kind, tuple(args.mode_indices))
    if args.cavity == "coaxial":
        return CoaxialCavity, (args.r_inner, args.r_outer, args.length), ModeIndex("TEM", tuple(args.mode_indices))
    raise ValueError(f"unknown cavity type {args.cavity!r}")


def build_cavity(args: argparse.Namespace) -> CavityMode:
    eps = args.bg_eps_r * constants.epsilon_0
    mu = args.bg_mu_r * constants.mu_0
    cavity_type, cavity_args, mode = cavity_type_and_args(args)
    return cavity_type(*cavity_args, mode, eps=eps, mu=mu)


def resolve_rs(cav: CavityMode, args: argparse.Namespace) -> float:
    if args.rs is not None:
        return args.rs
    return _rs_from_conductivity(cav.f0, args.conductivity)


# --- Sample construction ----------------------------------------------------

def _sample_margin(args: argparse.Namespace) -> float:
    """Characteristic half-size of the requested sample, used to keep the
    auto-picked position clear of cavity boundaries (see resolve_position)."""
    if args.shape == "sphere" or args.shape == "rod":
        return args.sample_radius or 0.0
    if args.shape == "disk":
        return 0.5 * max(args.sample_extent) if args.sample_extent else 0.0
    return 0.0


def resolve_position(cav: CavityMode, field: FieldProvider, args: argparse.Namespace) -> np.ndarray:
    """Default sample position: the E-field maximum within the cavity's
    actual valid domain. The bounding box's own geometric center is *not*
    a safe generic default -- e.g. for CoaxialCavity it falls at rho=0,
    inside the excluded inner conductor, right on the field's 1/rho
    singularity. Searching for the field max is both robust (guaranteed
    inside `contains()`) and physically apt (real measurements place the
    sample at a field extremum for maximum sensitivity).

    Candidates within `margin` of a boundary are excluded before ranking --
    without this, a mode whose field doesn't depend on one axis (e.g.
    TE_0np, flat along x) ties across that whole axis, and a naive argmax
    breaks the tie by picking the first grid point, which is a boundary."""
    if args.sample_position is not None:
        return np.array(args.sample_position, dtype=float)

    margin = _sample_margin(args)
    rmin, rmax = cav.bounding_box()
    n = 25
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(rmin, rmax)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    candidates = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    safe = cav.contains(candidates)
    if margin > 0:
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = margin
            safe = safe & cav.contains(candidates + offset) & cav.contains(candidates - offset)
    eligible = candidates[safe] if np.any(safe) else candidates[cav.contains(candidates)]
    if eligible.shape[0] == 0:
        raise ValueError(
            "could not find any point inside the cavity to place the sample -- "
            "specify --sample-position explicitly"
        )
    with np.errstate(all="ignore"):
        e_mag2 = np.sum(np.abs(field.E(eligible)) ** 2, axis=-1)
    e_mag2 = np.where(np.isfinite(e_mag2), e_mag2, 0.0)
    return eligible[np.argmax(e_mag2)]


def resolve_axis(field: FieldProvider, position: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Cylinder axis / Slab normal, per --sample-orientation. 'aligned' means
    parallel to the reference field at the sample's center (Module 3's N=0
    rod-axial / N=1 disk-normal case); 'perpendicular' means orthogonal to it
    (N=1/2 rod-transverse / N=0 disk-tangential case) -- these are exactly
    the canonical depolarization-table entries (module3 doc Section 2.2)."""
    if args.sample_axis is not None:
        v = np.array(args.sample_axis, dtype=float)
        return v / np.linalg.norm(v)

    ref_field = field.E(position) if args.sample_orient_field == "E" else field.H(position)
    ref_dir = real_field_direction(ref_field)
    if ref_dir is None:
        raise ValueError(
            f"the {args.sample_orient_field} field is (numerically) zero at --sample-position "
            f"{position.tolist()} -- pick a different position, orient relative to the other "
            "field (--sample-orient-field), or give an explicit --sample-axis"
        )
    if args.sample_orientation == "aligned":
        return ref_dir
    e1, _e2, _n = orthonormal_frame(ref_dir)
    return e1


def build_sample_region(cav: CavityMode, field: FieldProvider, args: argparse.Namespace) -> SampleRegion:
    position = resolve_position(cav, field, args)

    if args.shape == "sphere":
        if args.sample_radius is None:
            raise ValueError("--sample-radius is required for --shape sphere")
        return Sphere(center=position, radius=args.sample_radius)

    axis = resolve_axis(field, position, args)

    if args.shape == "rod":
        if args.sample_radius is None:
            raise ValueError("--sample-radius is required for --shape rod")
        length = args.sample_length if args.sample_length is not None else 16.0 * args.sample_radius
        return Cylinder(center=position, axis=axis, radius=args.sample_radius, height=length)

    if args.shape == "disk":
        if args.sample_extent is None:
            raise ValueError("--sample-extent EX EY is required for --shape disk")
        thickness = (
            args.sample_thickness
            if args.sample_thickness is not None
            else 0.05 * min(args.sample_extent)
        )
        return Slab(center=position, normal=axis, thickness=thickness, extent=tuple(args.sample_extent))

    raise ValueError(f"unknown --shape {args.shape!r}")


# --- Reporting ---------------------------------------------------------------

def _fmt_complex(z: complex) -> str:
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.6g}{sign}{abs(z.imag):.3g}j"


def print_report(
    cav: CavityMode,
    region: SampleRegion,
    material: Material,
    Rs: float | None,
    combined: PerturbationResult,
    sample_only: PerturbationResult,
    kappa_E: complex,
    kappa_H: complex,
    outside_fraction: float,
) -> None:
    f0 = cav.f0
    Q_wall = cav.Q_wall(Rs) if Rs is not None else float("inf")

    print("=== Cavity (Module 1) ===")
    print(f"f0     = {f0:.6e} Hz  ({f0 / 1e9:.4f} GHz)")
    print(f"Q_wall = {Q_wall:.4e}" + (f"  (Rs = {Rs:.4e} Ohm)" if Rs is not None else "  (wall loss disabled)"))

    print("\n=== Sample (Module 3) ===")
    print(f"shape        = {region.__class__.__name__}  (shape_kind='{region.shape_kind}')")
    print(f"center       = {np.asarray(getattr(region, 'center')).tolist()}")
    print(f"volume       = {region.volume():.4e} m^3")
    print(
        f"material     = eps_r={_fmt_complex(material.eps)} (tan_d_e={material.loss_tangent_e:.4g}), "
        f"mu_r={_fmt_complex(material.mu)} (tan_d_m={material.loss_tangent_m:.4g})"
    )
    print(f"kappa_E      = {_fmt_complex(kappa_E)}   kappa_H = {_fmt_complex(kappa_H)}")
    if outside_fraction > 0:
        print(
            f"WARNING: ~{outside_fraction:.1%} of the sample's quadrature points fall "
            "outside the cavity volume -- move or shrink the sample.",
            file=sys.stderr,
        )

    print("\n=== Perturbation result (Module 4) ===")
    df = combined.f_calc - f0
    print(f"f_calc              = {combined.f_calc:.6e} Hz   (df = {df:+.4e} Hz, df/f0 = {df / f0:+.4e})")
    print(f"Q_calc (loaded)     = {combined.Q_calc:.6e}")
    print(f"Q_sample_only       = {sample_only.Q_calc:.6e}   (wall loss excluded)")
    if Rs is not None:
        print(
            f"1/Q_calc            = {1.0 / combined.Q_calc:.6e}  ~=  1/Q_wall + 1/Q_sample_only "
            f"= {1.0 / Q_wall + 1.0 / sample_only.Q_calc:.6e}  (first-order additivity)"
        )


def print_ritz_comparison(
    f0: float,
    combined: PerturbationResult,
    ritz_combined: PerturbationResult,
    sample_only: PerturbationResult,
    ritz_sample_only: PerturbationResult,
    basis_size: int,
) -> None:
    """Compares Module 4's single-mode + depolarization-factor prediction
    against the Rayleigh-Ritz multi-mode prediction for the *same* cavity
    and sample. Growing disagreement as the sample grows relative to the
    cavity is the "sample-size correction" effect docs/ritz_module_plan.md
    Section 7.3 studies -- not evidence either individual answer is wrong."""
    print("\n=== Rayleigh-Ritz comparison (docs/ritz_module_plan.md) ===")
    print(f"basis size (N)       = {basis_size}")

    df_calc = ritz_combined.f_calc - combined.f_calc
    print(
        f"f_calc (Ritz)        = {ritz_combined.f_calc:.6e} Hz   "
        f"(Module 4: {combined.f_calc:.6e} Hz, diff = {df_calc:+.4e} Hz, rel = {df_calc / f0:+.4e})"
    )
    print(f"Q_calc (Ritz)        = {ritz_combined.Q_calc:.6e}   (Module 4: {combined.Q_calc:.6e})")

    inv_q_module4 = 1.0 / sample_only.Q_calc
    inv_q_ritz = 1.0 / ritz_sample_only.Q_calc
    print(
        f"1/Q_sample_only      = {inv_q_ritz:.4e} (Ritz)  vs  {inv_q_module4:.4e} (Module 4)   "
        f"diff = {inv_q_ritz - inv_q_module4:+.4e}"
    )


def plot_resonance_curves(
    f0: float,
    Q_wall: float,
    combined: PerturbationResult,
    save: str | None,
    ritz_combined: PerturbationResult | None = None,
) -> None:
    """Q_wall=inf (no wall loss at all) or combined.Q_calc=inf (no loss
    anywhere in this configuration) are both genuine, not edge cases to
    paper over: a lossless resonator really does have infinite Q, and a
    Lorentzian has no meaningful finite width to draw in that case -- drawn
    as a vertical line at the resonant frequency instead of a fabricated
    finite-width curve."""
    import matplotlib.pyplot as plt

    def lorentzian(f: np.ndarray, f_res: float, Q: float) -> np.ndarray:
        x = 2.0 * Q * (f - f_res) / f_res
        return 1.0 / (1.0 + x**2)

    # Range must cover all peaks' widths *and* the shifts between their
    # centers -- sizing it from one linewidth alone (ignoring the frequency
    # pull) can clip another peak off-screen when the shift exceeds it.
    results = (Q_wall, combined.Q_calc) + ((ritz_combined.Q_calc,) if ritz_combined is not None else ())
    finite_qs = [q for q in results if np.isfinite(q)]
    width = 4.0 * f0 / min(finite_qs) if finite_qs else 1e-3 * f0
    centers = (f0, combined.f_calc) + ((ritz_combined.f_calc,) if ritz_combined is not None else ())
    f_lo = min(centers) - width
    f_hi = max(centers) + width
    f = np.linspace(f_lo, f_hi, 2000)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    def plot_one(f_res: float, Q: float, label: str, color: str) -> None:
        if np.isfinite(Q):
            ax.plot(f / 1e9, lorentzian(f, f_res, Q), color=color, label=f"{label} (Q={Q:.3g})")
        else:
            ax.axvline(f_res / 1e9, color=color, linestyle="--", label=f"{label} (Q=inf)")

    plot_one(f0, Q_wall, "original cavity, no sample", "tab:blue")
    plot_one(combined.f_calc, combined.Q_calc, "loaded, with sample (Module 4)", "tab:orange")
    if ritz_combined is not None:
        plot_one(ritz_combined.f_calc, ritz_combined.Q_calc, "loaded, with sample (Ritz)", "tab:green")

    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("normalized response")
    ax.set_title("Original cavity vs. sample-loaded resonance")
    ax.legend()
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150)
        print(f"Saved plot to {save}")
    else:
        plt.show()


# --- CLI ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="cavity", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode-indices", type=int, nargs="+", required=True,
                         help="mode index tuple, e.g. --mode-indices 0 1 1")
    common.add_argument("--bg-eps-r", type=float, default=1.0, help="cavity fill relative permittivity (default: 1.0, vacuum)")
    common.add_argument("--bg-mu-r", type=float, default=1.0, help="cavity fill relative permeability (default: 1.0, vacuum)")

    rs_group = common.add_mutually_exclusive_group()
    rs_group.add_argument("--rs", type=float, default=None, help="wall surface resistance [Ohm] (overrides --conductivity)")
    rs_group.add_argument("--conductivity", type=float, default=COPPER_CONDUCTIVITY,
                           help="wall conductivity [S/m] used to derive Rs if --rs not given (default: copper)")
    common.add_argument("--no-wall-loss", action="store_true", help="ignore wall loss entirely (Q_wall -> infinity)")

    common.add_argument("--shape", choices=["sphere", "rod", "disk"], required=True,
                         help="standard sample shape (sphere / thin rod / thin disk)")
    common.add_argument("--sample-radius", type=float, default=None, help="sphere or rod radius [m]")
    common.add_argument("--sample-length", type=float, default=None,
                         help="rod length [m] (default: 16x radius, comfortably 'thin_rod')")
    common.add_argument("--sample-extent", type=float, nargs=2, default=None, metavar=("EX", "EY"),
                         help="disk lateral extent [m] (required for --shape disk)")
    common.add_argument("--sample-thickness", type=float, default=None,
                         help="disk thickness [m] (default: 0.05x min(extent), comfortably 'thin_disk')")
    common.add_argument("--sample-position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                         help="sample center [m] (default: cavity bounding-box center)")
    common.add_argument("--sample-axis", type=float, nargs=3, default=None, metavar=("NX", "NY", "NZ"),
                         help="explicit rod axis / disk normal (overrides --sample-orientation)")
    common.add_argument("--sample-orientation", choices=["aligned", "perpendicular"], default="aligned",
                         help="rod axis / disk normal relative to the local field (default: aligned)")
    common.add_argument("--sample-orient-field", choices=["E", "H"], default="E",
                         help="which field to orient the rod/disk relative to (default: E)")

    common.add_argument("--sample-eps-r", type=float, default=2.5, help="sample relative permittivity (default: 2.5)")
    common.add_argument("--sample-tan-delta-e", type=float, default=1e-3, help="sample electric loss tangent (default: 0.001)")
    common.add_argument("--sample-mu-r", type=float, default=1.0, help="sample relative permeability (default: 1.0)")
    common.add_argument("--sample-tan-delta-m", type=float, default=0.0, help="sample magnetic loss tangent (default: 0.0)")

    common.add_argument("--plot", action="store_true", help="plot wall-only vs. loaded resonance curves (needs matplotlib)")
    common.add_argument("--save", type=str, default=None, help="save the plot to this path instead of showing it")
    common.add_argument("--force", action="store_true",
                         help="proceed even if a large fraction of the sample lies outside the cavity's valid volume")

    common.add_argument("--compare-ritz", action="store_true",
                         help="also predict (f, Q) via the Rayleigh-Ritz multi-mode model and report the "
                              "difference from Module 4 (non-magnetic samples only, --sample-mu-r 1)")
    common.add_argument("--ritz-n-basis", type=int, default=5,
                         help="Ritz basis size: the mode of interest plus this many minus one "
                              "nearest-frequency modes (default: 5)")
    common.add_argument("--ritz-max-index", type=int, default=4,
                         help="max per-axis mode index searched when picking Ritz basis candidates (default: 4)")

    rect = subparsers.add_parser("rectangular", parents=[common])
    rect.add_argument("--a", type=float, required=True)
    rect.add_argument("--b", type=float, required=True)
    rect.add_argument("--c", type=float, required=True)
    rect.add_argument("--mode-kind", choices=["TE", "TM"], default="TE")

    cyl = subparsers.add_parser("cylindrical", parents=[common])
    cyl.add_argument("--radius", type=float, required=True)
    cyl.add_argument("--length", type=float, required=True)
    cyl.add_argument("--mode-kind", choices=["TE", "TM"], default="TM")

    coax = subparsers.add_parser("coaxial", parents=[common])
    coax.add_argument("--r-inner", type=float, required=True)
    coax.add_argument("--r-outer", type=float, required=True)
    coax.add_argument("--length", type=float, required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    try:
        cav = build_cavity(args)
        field = AnalyticalField(cav)

        region = build_sample_region(cav, field, args)
        material = Material.from_loss_tangent(
            args.sample_eps_r, args.sample_tan_delta_e, args.sample_mu_r, args.sample_tan_delta_m
        )
        if not material.is_passive:
            raise ValueError(
                f"sample material is not passive (eps_r={material.eps}, mu_r={material.mu}) -- "
                "check --sample-tan-delta-e/-m are >= 0"
            )
        sample = Sample(region=region, material=material)

        pts, _w = region.quadrature_points(200)
        outside_fraction = 1.0 - float(np.mean(cav.contains(pts)))
        if outside_fraction > _OUTSIDE_FRACTION_TOLERANCE and not args.force:
            raise ValueError(
                f"~{outside_fraction:.1%} of the sample lies outside the cavity's valid volume "
                "(e.g. clipping into a conductor) -- the field there is extrapolated nonsense, "
                "not physical. Shrink the sample, move --sample-position further from the "
                "boundary, or pass --force to proceed anyway (e.g. for a deliberate boundary test)."
            )

        Rs = None if args.no_wall_loss else resolve_rs(cav, args)
        model_combined = PerturbationModel(field, Rs_walls=Rs)
        model_sample_only = PerturbationModel(field, Rs_walls=None)

        combined = model_combined.evaluate(sample)
        sample_only = model_sample_only.evaluate(sample)

        center = np.asarray(getattr(region, "center"))
        kappa_E = sample.depolarization_factor("E", field.E(center))
        kappa_H = sample.depolarization_factor("H", field.H(center))

        print_report(cav, region, material, Rs, combined, sample_only, kappa_E, kappa_H, outside_fraction)

        ritz_combined = None
        if args.compare_ritz:
            # A local try/except: the primary Module 4 simulation above
            # already succeeded and its report already printed -- an
            # out-of-scope sample (mu_r != 1) for the optional Ritz
            # comparison shouldn't discard that or exit(1), just skip the
            # comparison with a clear reason.
            try:
                if abs(material.mu - 1.0) > 1e-9:
                    raise ValueError(
                        f"RitzCorrectedModel only supports non-magnetic samples (mu_r=1), "
                        f"got mu_r={material.mu!r}"
                    )
                cavity_type, cavity_args, mode = cavity_type_and_args(args)
                basis = nearest_basis_modes(
                    cavity_type, cavity_args, mode,
                    n_basis=args.ritz_n_basis, max_index=args.ritz_max_index,
                    eps_bg=cav.epsilon_bg, mu_bg=cav.mu_bg,
                )
                ritz_model_combined = RitzCorrectedModel(basis, Rs_walls=Rs)
                ritz_model_sample_only = RitzCorrectedModel(basis, Rs_walls=None)
                ritz_combined = ritz_model_combined.evaluate(sample)
                ritz_sample_only = ritz_model_sample_only.evaluate(sample)
                print_ritz_comparison(cav.f0, combined, ritz_combined, sample_only, ritz_sample_only, len(basis))
            except ValueError as exc:
                print(f"\nwarning: --compare-ritz skipped: {exc}", file=sys.stderr)
                ritz_combined = None

        if args.plot:
            # Respect --no-wall-loss for the reference curve too: with no
            # wall-loss mechanism modeled, the original cavity genuinely has
            # infinite Q (not some other Rs's value) -- plot_resonance_curves
            # renders that honestly as a vertical line, not a fabricated
            # finite-width curve.
            Q_wall_for_plot = cav.Q_wall(Rs) if Rs is not None else float("inf")
            plot_resonance_curves(cav.f0, Q_wall_for_plot, combined, args.save, ritz_combined)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
