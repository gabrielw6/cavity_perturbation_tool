#!/usr/bin/env python
"""Small CLI for inspecting a Module 1 cavity: prints f0, Q_wall, stored
energy and wall-loss power, and plots a cross-sectional |E| or |H| magnitude
map. Built only on the public CavityMode interface (E, H, f0, Q_wall,
total_stored_energy, contains, bounding_box) -- no knowledge of a, b, c or
Bessel functions leaks in here.

Usage:
    python scripts/visualize_cavity.py rectangular --a 0.03 --b 0.03 --c 0.03 \\
        --mode-kind TE --mode-indices 0 1 1 --plane xy --field E

    python scripts/visualize_cavity.py cylindrical --radius 0.02 --length 0.03 \\
        --mode-kind TM --mode-indices 0 1 0 --plane xz --field H --save q.png

    python scripts/visualize_cavity.py coaxial --r-inner 0.01 --r-outer 0.023 \\
        --length 0.5 --mode-indices 1 --field E --no-plot

Requires matplotlib for plotting (pip install -e ".[viz]"); --no-plot works
without it.
"""
from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, Any

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

COPPER_CONDUCTIVITY = 5.8e7  # S/m


def _rs_from_conductivity(f0: float, sigma: float) -> float:
    return float(np.sqrt(np.pi * f0 * constants.mu_0 / sigma))


def build_cavity(args: argparse.Namespace) -> CavityMode:
    eps = args.eps_r * constants.epsilon_0
    mu = args.mu_r * constants.mu_0

    if args.cavity == "rectangular":
        mode = ModeIndex(args.mode_kind, tuple(args.mode_indices))
        return RectangularCavity(args.a, args.b, args.c, mode, eps=eps, mu=mu)
    if args.cavity == "cylindrical":
        mode = ModeIndex(args.mode_kind, tuple(args.mode_indices))
        return CylindricalCavity(args.radius, args.length, mode, eps=eps, mu=mu)
    if args.cavity == "coaxial":
        mode = ModeIndex("TEM", tuple(args.mode_indices))
        return CoaxialCavity(args.r_inner, args.r_outer, args.length, mode, eps=eps, mu=mu)
    raise ValueError(f"unknown cavity type {args.cavity!r}")


def resolve_rs(cav: CavityMode, args: argparse.Namespace) -> float:
    if args.rs is not None:
        return args.rs
    return _rs_from_conductivity(cav.f0, args.conductivity)


def print_summary(cav: CavityMode, rs: float) -> None:
    f0 = cav.f0
    Q = cav.Q_wall(rs)
    W = cav.total_stored_energy()
    P_loss = 2.0 * np.pi * f0 * W / Q

    print(f"f0      = {f0:.6e} Hz  ({f0 / 1e9:.4f} GHz)")
    print(f"Q_wall  = {Q:.4e}   (Rs = {rs:.4e} Ohm)")
    print(f"W       = {W:.4e}   (stored energy, arbitrary field-amplitude scale)")
    print(f"P_loss  = {P_loss:.4e}   (wall-loss power, same arbitrary scale as W)")


_PLANE_AXES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
_PLANE_LABELS = {"xy": ("x [m]", "y [m]"), "xz": ("x [m]", "z [m]"), "yz": ("y [m]", "z [m]")}


def cross_section_grid(
    cav: CavityMode, plane: str, slice_value: float, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an (n, n) grid over the two in-plane axes at a fixed value of the
    third axis, using the cavity's own bounding_box -- works for any geometry
    without this script knowing its shape."""
    rmin, rmax = cav.bounding_box()
    i, j, k = _PLANE_AXES[plane]
    u = np.linspace(rmin[i], rmax[i], n)
    v = np.linspace(rmin[j], rmax[j], n)
    U, V = np.meshgrid(u, v)
    pts = np.zeros((U.size, 3))
    pts[:, i] = U.ravel()
    pts[:, j] = V.ravel()
    pts[:, k] = slice_value
    return U, V, pts


def field_magnitude(cav: CavityMode, pts: np.ndarray, field: str) -> np.ndarray:
    with np.errstate(all="ignore"):  # points outside the physical domain
        vals = cav.E(pts) if field == "E" else cav.H(pts)  # (e.g. coax rho=0) can divide by zero
    mag = np.sqrt(np.sum(np.abs(vals) ** 2, axis=-1))
    mask = cav.contains(pts)
    return np.where(mask, mag, np.nan)


def plot_cross_section(
    cav: CavityMode, plane: str, slice_value: float, n: int, field: str, ax: "Axes"
) -> Any:
    U, V, pts = cross_section_grid(cav, plane, slice_value, n)
    mag = field_magnitude(cav, pts, field).reshape(U.shape)
    im = ax.pcolormesh(U, V, mag, shading="auto", cmap="viridis")
    # "xy" cross-sections should look geometrically round; longitudinal "xz"/"yz"
    # cuts are often much longer than they are wide (e.g. coax), where forcing
    # equal aspect makes the plot unreadably thin.
    ax.set_aspect("equal" if plane == "xy" else "auto")
    xlabel, ylabel = _PLANE_LABELS[plane]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"|{field}| on {plane} plane (slice = {slice_value:.4g} m)")
    return im


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="cavity", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mode-indices", type=int, nargs="+", required=True,
                         help="mode index tuple, e.g. --mode-indices 0 1 1")
    common.add_argument("--eps-r", type=float, default=1.0, help="relative permittivity of the fill (default: 1.0, vacuum)")
    common.add_argument("--mu-r", type=float, default=1.0, help="relative permeability of the fill (default: 1.0, vacuum)")
    rs_group = common.add_mutually_exclusive_group()
    rs_group.add_argument("--rs", type=float, default=None, help="wall surface resistance [Ohm] (overrides --conductivity)")
    rs_group.add_argument("--conductivity", type=float, default=COPPER_CONDUCTIVITY,
                           help="wall conductivity [S/m] used to derive Rs if --rs not given (default: copper)")
    common.add_argument("--field", choices=["E", "H"], default="E", help="field to plot (default: E)")
    common.add_argument("--plane", choices=["xy", "xz", "yz"], default="xy", help="cross-section plane (default: xy)")
    common.add_argument("--slice", type=float, default=None,
                         help="fixed coordinate of the plane's third axis (default: midpoint of the bounding box)")
    common.add_argument("--n", type=int, default=200, help="grid resolution per axis (default: 200)")
    common.add_argument("--save", type=str, default=None, help="save the plot to this path instead of showing it")
    common.add_argument("--no-plot", action="store_true", help="print the summary only, skip plotting (no matplotlib needed)")

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
    cav = build_cavity(args)
    rs = resolve_rs(cav, args)
    print_summary(cav, rs)

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting; install with pip install -e \".[viz]\","
              " or pass --no-plot.", file=sys.stderr)
        sys.exit(1)

    rmin, rmax = cav.bounding_box()
    k = _PLANE_AXES[args.plane][2]
    slice_value = args.slice if args.slice is not None else (rmin[k] + rmax[k]) / 2.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = plot_cross_section(cav, args.plane, slice_value, args.n, args.field, ax)
    fig.colorbar(im, ax=ax, label=f"|{args.field}| (arbitrary scale)")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved plot to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
