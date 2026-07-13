"""Rayleigh-Ritz sample-size-correction module.

`RitzCorrectedModel` is a `PerturbationModel` sibling (matches
`evaluate(sample) -> PerturbationResult` plus the `field_provider`/
`Rs_walls` accessors Module 5's closed-form seed needs), not a
`FieldProvider` -- captures multi-mode field mixing around a sample whose
volume isn't negligible relative to the cavity, which a single-mode
quasi-static treatment (Module 4) can't see. See docs/ritz_module_plan.md
for the full derivation; Section 0 there explains why the originally-sketched
`RitzField(FieldProvider)` stub (now removed from `fields.py`) was never the
right interface for this.

Scope: non-magnetic samples only (mu_r=1) -- mu_bg stays uniform throughout,
only eps(r) becomes non-uniform via the sample (Section 2).
"""
from __future__ import annotations

import itertools
import warnings
from typing import Callable, Sequence

import numpy as np
from scipy import constants
from scipy.linalg import eig

from .cavity import CavityMode, ModeIndex
from .fields import AnalyticalField, FieldProvider, integrate_field_cross_overlap
from .perturbation import PerturbationResult, omega_tilde_to_result
from .sample import Sample

Array = np.ndarray

_DEFAULT_N_BASIS = 5
_DEFAULT_MAX_INDEX = 4
_MU_R_TOL = 1e-9
_DEGENERACY_FLAG_RATIO = 0.5  # second-largest / largest basis-1 weight above this -> flag ambiguous mode tracking


def _enumerate_candidate_modes(mode_of_interest: ModeIndex, max_index: int = _DEFAULT_MAX_INDEX) -> list[ModeIndex]:
    """Candidate other-mode index tuples of the same kind-family and index
    arity as `mode_of_interest` (Section 1). Doesn't know or re-derive any
    cavity type's validity rules -- `nearest_basis_modes` filters invalid
    combinations by actually constructing them and catching the ValueError
    `cavity.py`'s own constructors already raise."""
    n_indices = len(mode_of_interest.indices)
    if n_indices == 1:
        # Coaxial-style: single axial standing-wave index, q >= 1.
        return [ModeIndex(mode_of_interest.kind, (q,)) for q in range(1, max_index + 3)]
    kinds = ("TE", "TM") if mode_of_interest.kind in ("TE", "TM") else (mode_of_interest.kind,)
    ranges = [range(0, max_index + 1)] * n_indices
    return [ModeIndex(kind, idx) for kind in kinds for idx in itertools.product(*ranges)]


def nearest_basis_modes(
    cavity_type: Callable[..., CavityMode],
    cavity_args: tuple[object, ...],
    mode_of_interest: ModeIndex,
    n_basis: int = _DEFAULT_N_BASIS,
    amplitude: complex = 1.0,
    eps_bg: complex = constants.epsilon_0,
    mu_bg: complex = constants.mu_0,
    max_index: int = _DEFAULT_MAX_INDEX,
) -> list[CavityMode]:
    """Basis = the mode of interest (always index 0 of the result) plus the
    `n_basis - 1` nearest-frequency other modes of the same canonical cavity
    (docs/ritz_module_plan.md Section 1) -- all from Module 1's exact closed
    forms, no new field solutions. Candidates are found by constructing them
    and letting `cavity_type`'s own validity checks filter invalid index
    combinations (a ValueError), rather than re-deriving each cavity type's
    rules here."""
    mode_of_interest_cavity = cavity_type(*cavity_args, mode_of_interest, amplitude=amplitude, eps=eps_bg, mu=mu_bg)
    f0_target = mode_of_interest_cavity.f0

    candidates: list[tuple[float, CavityMode]] = []
    for mode in _enumerate_candidate_modes(mode_of_interest, max_index):
        if mode == mode_of_interest:
            continue
        try:
            cav = cavity_type(*cavity_args, mode, amplitude=amplitude, eps=eps_bg, mu=mu_bg)
        except ValueError:
            continue
        candidates.append((abs(cav.f0 - f0_target), cav))

    candidates.sort(key=lambda pair: pair[0])
    nearby = [cav for _, cav in candidates[: max(n_basis - 1, 0)]]
    return [mode_of_interest_cavity, *nearby]


class RitzCorrectedModel:
    """Multi-mode Rayleigh-Ritz forward model: given a Sample, predicts the
    perturbed (f, Q) by letting `basis_modes` (Module 1 exact modes of one
    canonical cavity, `basis_modes[0]` the mode of interest) mix under the
    sample's field-non-uniformity, rather than the single-mode quasi-static
    treatment `PerturbationModel` uses. See docs/ritz_module_plan.md Sections
    2-3."""

    def __init__(
        self,
        basis_modes: Sequence[CavityMode],
        Rs_walls: float | None = None,
        n_points: int = 2000,
    ) -> None:
        if len(basis_modes) < 1:
            raise ValueError("basis_modes must contain at least the mode of interest")
        eps_bg, mu_bg = basis_modes[0].epsilon_bg, basis_modes[0].mu_bg
        for mode in basis_modes[1:]:
            if mode.epsilon_bg != eps_bg or mode.mu_bg != mu_bg:
                raise ValueError(
                    "all basis modes must share the same background eps_bg/mu_bg -- "
                    "docs/ritz_module_plan.md Section 2 assumes a uniform background medium"
                )
        self._basis_modes = list(basis_modes)
        self._Rs_walls = Rs_walls
        self._n_points = n_points
        # Section 5: field_provider must point at the mode-of-interest's own
        # FieldProvider, for Module 5's closed-form seed (bypasses evaluate()).
        self._field_provider = AnalyticalField(basis_modes[0])

    @property
    def field_provider(self) -> FieldProvider:
        return self._field_provider

    @property
    def Rs_walls(self) -> float | None:
        return self._Rs_walls

    @property
    def basis_size(self) -> int:
        return len(self._basis_modes)

    def evaluate(self, sample: Sample) -> PerturbationResult:
        """Predict (f_calc, Q_calc) for `sample` via the Ritz-mixed basis.
        Raises ValueError if `sample.material` isn't passive (same guard as
        `PerturbationModel.evaluate`) or isn't non-magnetic (mu_r=1, Section
        2's scope)."""
        if not sample.material.is_passive:
            raise ValueError(
                f"material {sample.material!r} is not passive "
                "(requires eps''>=0, mu''>=0, eps'>0, mu'>0)"
            )
        if abs(sample.material.mu - 1.0) > _MU_R_TOL:
            raise ValueError(
                f"RitzCorrectedModel only supports non-magnetic samples (mu_r=1), got "
                f"mu_r={sample.material.mu!r} -- docs/ritz_module_plan.md Section 2 scope"
            )

        N = len(self._basis_modes)
        eps_bg = self._basis_modes[0].epsilon_bg
        eps_r = sample.material.eps
        region = sample.region

        # K (Section 2.1): exactly diagonal, built entirely from each basis
        # mode's own f0/total_stored_energy -- no quadrature needed.
        K = np.zeros((N, N), dtype=complex)
        M = np.zeros((N, N), dtype=complex)
        for i, mode_i in enumerate(self._basis_modes):
            omega_i = 2.0 * np.pi * mode_i.f0
            W_i = mode_i.total_stored_energy()
            K[i, i] = omega_i**2 * W_i
            M[i, i] = W_i

        # Delta M (Sections 2.2-2.3): only the bare material contrast is
        # conjugated (not the cross-overlap integral) -- the same passivity
        # fix Module 4 needed, verified the same way (see CLAUDE.md's Delta-
        # conjugate entry). Module 3's depolarization factor is deliberately
        # NOT applied here (Section 2.3) -- letting the basis modes mix is
        # what captures the field distortion kappa_E approximates elsewhere,
        # and kappa_E isn't even well-defined for a multi-mode expansion.
        material_contrast = np.conj(eps_r - 1.0)
        for i in range(N):
            for j in range(i, N):
                overlap = integrate_field_cross_overlap(
                    region, self._basis_modes[i].E, self._basis_modes[j].E, n_points=self._n_points
                )
                delta_M_ij = material_contrast * eps_bg * overlap
                M[i, j] += delta_M_ij
                if j != i:
                    M[j, i] += np.conj(delta_M_ij)  # exact Hermitian-conjugate symmetry, not re-integrated

        # Section 2.5: general (non-Hermitian-safe) solver -- M is only
        # Hermitian for a lossless sample; eigh would silently discard the
        # loss information this whole project exists to extract.
        eigenvalues, eigenvectors = eig(K, M)

        # Section 3.2: mode tracking by weight on the mode-of-interest
        # (index 0), flagging rather than silently resolving near-degeneracy.
        # Each basis mode is an independent CavityMode instance with its own
        # arbitrary field-amplitude scale (Module 1's normalization
        # convention is only self-consistent *per instance*, e.g. TE and TM
        # modes at the same nominal amplitude=1.0 can differ in
        # total_stored_energy by many orders of magnitude) -- a raw
        # eigenvector-component weight is not invariant to that and would
        # over/under-weight components arbitrarily. Rescale by each
        # component's own sqrt(energy) first (Section 4's basis-rescaling
        # argument: A_i*sqrt(M_ii) is invariant to E_i -> c_i*E_i), so
        # "weight" reflects each mode's actual physical contribution.
        energy_scale = np.sqrt(np.abs(np.diag(M)))
        scaled = eigenvectors * energy_scale[:, None]
        norms = np.linalg.norm(scaled, axis=0)
        weights = np.abs(scaled[0, :]) / norms
        order = np.argsort(weights)[::-1]
        k_star = order[0]
        if len(order) > 1 and weights[order[1]] > _DEGENERACY_FLAG_RATIO * weights[order[0]]:
            warnings.warn(
                "RitzCorrectedModel: near-degenerate mode mixing -- the two largest "
                f"basis-1 weights are close ({weights[order[1]]:.3g} vs {weights[order[0]]:.3g}) "
                "-- the sample is inducing strong mixing with a neighboring mode, and 'the "
                "corrected mode of interest' is not unambiguous (docs/ritz_module_plan.md "
                "Section 3.2)",
                RuntimeWarning,
                stacklevel=2,
            )

        # Section 3.3: wall loss added last, as a separate first-order
        # perturbation on top of the (lossless-basis) Ritz eigenvalue, using
        # mode 1's own base omega as the small-perturbation reference --
        # exactly mirroring Module 4's combination formula.
        omega_k_star = np.sqrt(complex(eigenvalues[k_star]))
        if omega_k_star.real < 0:
            omega_k_star = -omega_k_star  # principal (positive-real-part) branch

        omega_1 = 2.0 * np.pi * self._basis_modes[0].f0
        if self._Rs_walls is not None:
            Q_wall = self._field_provider.Q_wall(self._Rs_walls)
            wall_term: complex = -1j * omega_1 / (2.0 * Q_wall)
        else:
            wall_term = 0j
        omega_tilde = omega_k_star + wall_term

        return omega_tilde_to_result(omega_tilde)


def converged_ritz_model(
    cavity_type: Callable[..., CavityMode],
    cavity_args: tuple[object, ...],
    mode_of_interest: ModeIndex,
    sample: Sample,
    Rs_walls: float | None = None,
    amplitude: complex = 1.0,
    eps_bg: complex = constants.epsilon_0,
    mu_bg: complex = constants.mu_0,
    n_start: int = 3,
    n_step: int = 2,
    tol: float = 1e-4,
    max_n: int = 15,
    n_points: int = 2000,
) -> tuple[RitzCorrectedModel, PerturbationResult]:
    """Basis-size convergence control (Section 3.5): grow the basis by
    `n_step` modes at a time, comparing `omega_tilde` at N and N+n_step,
    until the relative change falls within `tol` (or raise at `max_n`,
    rather than silently return an unconverged basis -- same "flag, don't
    guess" philosophy as Module 2's quadrature convergence). Returns the
    converged model (fixed basis, ready to use in a `Measurement`) alongside
    its result, since re-evaluating would waste the last iteration's work.
    """
    n = max(n_start, 1)
    basis = nearest_basis_modes(cavity_type, cavity_args, mode_of_interest, n, amplitude, eps_bg, mu_bg)
    model = RitzCorrectedModel(basis, Rs_walls=Rs_walls, n_points=n_points)
    prev = model.evaluate(sample)

    while n < max_n:
        n = min(n + n_step, max_n)
        basis = nearest_basis_modes(cavity_type, cavity_args, mode_of_interest, n, amplitude, eps_bg, mu_bg)
        model = RitzCorrectedModel(basis, Rs_walls=Rs_walls, n_points=n_points)
        curr = model.evaluate(sample)

        denom = curr.omega_tilde if curr.omega_tilde != 0.0 else prev.omega_tilde
        if denom == 0.0 or abs((curr.omega_tilde - prev.omega_tilde) / denom) <= tol:
            return model, curr
        prev = curr
        if len(basis) < n:
            break  # ran out of constructible candidate modes before reaching max_n

    raise RuntimeError(
        f"RitzCorrectedModel basis-size convergence did not reach rtol={tol} within max_n={max_n} basis modes"
    )
