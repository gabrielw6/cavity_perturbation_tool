# Cavity Perturbation Software — Modules 1–5 Design

Scope: `cavity`, `fields`, `sample`, `perturbation`, `inverse`. No Jacobian — Module 5
uses numerical (finite-difference) differentiation. Ritz fields are stubbed as an
interface only; full Ritz implementation is deferred (later "sample-size correction"
phase), but the contract is fixed now so it drops in without touching modules 3–5.

---

## 0. Shared conventions (fix these before writing any module)

These are cross-cutting decisions. Getting them wrong doesn't break any single module in
isolation — it breaks *integration*, silently, with a sign or factor-of-2 error that's
painful to trace later.

| Convention | Choice | Why it matters downstream |
|---|---|---|
| Time convention | $e^{+j\omega t}$ | Fixes sign of loss: lossy media have $\epsilon = \epsilon' - j\epsilon''$, $\mu = \mu' - j\mu''$, both $\epsilon'',\mu'' \ge 0$. |
| Complex resonance | $\tilde\omega = \omega(1 - j/2Q)$ | $Q > 0 \Rightarrow \mathrm{Im}(\tilde\omega) < 0$, consistent with the time convention above. |
| Frequency units at module boundaries | Hz in/out of every public API; rad/s only inside formula internals | Prevents $2\pi$ bugs at integration points — never let a caller guess which one a function expects. |
| Field normalization | **Arbitrary but self-consistent per `FieldProvider` instance** — not fixed to unit energy | The perturbation formula is a *ratio* (sample integral / total energy integral), so it's scale-invariant. Fixing normalization is unnecessary work and an extra place to introduce bugs. Enforce this with a unit test: scaling `E,H` by any constant must leave `PerturbationModel.evaluate()` unchanged. |
| Vectorized evaluation | Every `E(r)`, `H(r)` accepts `r` of shape `(3,)` or `(N,3)` and returns `(3,)` or `(N,3)` complex | Quadrature-based integration (Module 2) needs to evaluate fields at thousands of points; looping in Python per-point will be a real bottleneck. Build vectorization in from day one rather than retrofitting it. |

---

## Module 1 — `cavity`: Analytical Cavity Library

### Strategy

This module's only job is to answer "what are the exact fields and resonance of the
*empty* cavity in a given mode?" It must not know anything about samples or
perturbation — keeping it pure makes it independently testable against closed-form
textbook values (Harrington Ch. 2, 5, 6), which is your ground truth for everything
built on top.

Two responsibilities per cavity type:
1. Evaluate $E(r)$, $H(r)$ pointwise (for quadrature-based integration downstream).
2. Provide $f_0$ and $Q$ (from wall loss, given surface resistance $R_s$), which requires
   the volume energy integral and the wall surface-current integral. For rectangular
   and cylindrical modes these have closed forms — implement them analytically rather
   than falling back to quadrature, since they're the most-called and most
   accuracy-sensitive numbers in the whole pipeline (they set your absolute frequency
   scale).

### Interface

```python
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class ModeIndex:
    """Generic mode label. Meaning of the fields depends on cavity type
    (e.g. rectangular: (m,n,p); circular TE/TM: (kind, n, m, l))."""
    kind: str            # e.g. "TE", "TM"
    indices: tuple[int, ...]

class CavityMode(ABC):
    """One resonant mode of one empty (unperturbed) cavity."""

    @abstractmethod
    def E(self, r: np.ndarray) -> np.ndarray:
        """r: (3,) or (N,3) in meters, cavity-local Cartesian frame.
        Returns complex field, same leading shape as r, units V/m
        (arbitrary overall scale — see normalization convention)."""

    @abstractmethod
    def H(self, r: np.ndarray) -> np.ndarray:
        """Same contract as E, units A/m, same arbitrary scale as E
        (scale of E and H must be *mutually* consistent — see f0/Q below)."""

    @property
    @abstractmethod
    def f0(self) -> float:
        """Resonant frequency of this mode in Hz, closed-form for empty cavity."""

    @abstractmethod
    def Q_wall(self, Rs: float) -> float:
        """Unloaded Q from finite wall conductivity, given surface resistance
        Rs [Ohm]. Closed form: Q = omega0 * W / P_loss, both computed
        analytically for this mode/geometry."""

    @abstractmethod
    def stored_energy_density(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (w_e(r), w_m(r)): local electric and magnetic energy
        densities (proportional to eps0*|E|^2, mu0*|H|^2) at the *same*
        arbitrary scale as E, H above. Needed so Module 2 doesn't have to
        re-derive eps0|E|^2 from E() and guess a scale."""

    @abstractmethod
    def total_stored_energy(self) -> float:
        """W = integral over V of (w_e + w_m) dV, closed form for this mode.
        This is THE normalization anchor: everything in Module 2/4 that
        needs 'energy in the whole cavity' calls this instead of
        re-integrating numerically."""

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """(rmin, rmax) axis-aligned box containing the cavity volume, in the
        cavity-local frame. Used by Module 2's default quadrature to avoid
        sampling wasted points outside V."""
        ...

    @abstractmethod
    def contains(self, r: np.ndarray) -> np.ndarray:
        """Boolean mask, True where r is inside the cavity volume. Needed
        because bounding_box() is generally not the same as V (e.g. cylinder
        in a Cartesian box)."""
```

Concrete classes, one file each:

```python
class RectangularCavity(CavityMode):
    def __init__(self, a: float, b: float, c: float, mode: ModeIndex): ...

class CylindricalCavity(CavityMode):
    def __init__(self, radius: float, length: float, mode: ModeIndex): ...

class CoaxialCavity(CavityMode):
    def __init__(self, r_inner: float, r_outer: float, length: float,
                 mode: ModeIndex): ...
```

Each implements $E$, $H$ from the standard textbook mode expressions (Harrington's
sinusoidal / Bessel-function forms), and `f0`, `Q_wall`, `total_stored_energy` from the
matching closed-form expressions — no numerical integration inside Module 1 at all.

### Contract to Module 2

`CavityMode` is the entire surface Module 2 depends on. Module 2 never touches `a,b,c`
or Bessel functions directly — it only calls `E`, `H`, `contains`, `bounding_box`,
`total_stored_energy`. This is the seam that lets `fields.AnalyticalField` be a thin
wrapper, and later lets a Ritz or FEM field provider satisfy the *same downstream
contract* (Module 2's own `FieldProvider` interface, not this one) without Module 1
knowing it exists.

### Test plan (do this before Module 2)
- Rectangular $\mathrm{TE}_{101}$: check $f_0$ against $\frac{c}{2}\sqrt{(1/a)^2+(1/c)^2}$.
- Circular cavity $\mathrm{TM}_{010}$: check against $f_0 = 1.841\, c /(2\pi a)$ (matches your earlier probe on the Ritz trial-field example).
- Numerically integrate $|E|^2$ over the full volume via brute-force quadrature and confirm it's proportional to `total_stored_energy()` (validates internal consistency of the closed forms against the field expressions).

---

## Module 2 — `fields`: Field Provider Abstraction

### Strategy

This is the actual seam of the whole architecture. Module 4 (`perturbation`) must be
able to ask "what is $\int_{V_s} |E|^2\,dV$ and $\int_{V_s}|H|^2\,dV$, and what is the
total stored energy?" without caring whether the field came from a closed-form cavity
mode, a Rayleigh–Ritz expansion, or (eventually) an FEM solve. Get this interface right
and modules 3–5 never need to change when you add Ritz or FEM later.

Two integration strategies coexist:
- **Fast path**: some `(cavity type, region shape)` pairs have closed-form sub-volume
  integrals (e.g. rectangular cavity × axis-aligned box). Implement these as optional
  overrides for speed/accuracy, but don't require them.
- **Default path**: numerical quadrature — evaluate the field at the region's quadrature
  points (supplied by Module 3's `SampleRegion`) and take the weighted sum. This must
  always work, for any region shape, for any field provider. Build this first; treat
  fast paths as an optimization added later if profiling says so.

### Interface

```python
from abc import ABC, abstractmethod
import numpy as np

class FieldProvider(ABC):
    """Uniform access to a trial/exact field solution for one cavity mode."""

    @abstractmethod
    def E(self, r: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def H(self, r: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def total_stored_energy(self) -> float:
        """Denominator of the perturbation formula. Must be on the SAME
        scale as E()/H() returned by this same instance."""

    @property
    @abstractmethod
    def f0(self) -> float: ...

    @abstractmethod
    def Q_wall(self, Rs: float) -> float: ...

    def integrate_field_energy(self, region: "SampleRegion", field: str,
                                n_points: int = 2000) -> complex:
        """Default (quadrature) implementation of:
             integral over `region` of |E|^2 dV   (field='E')
             integral over `region` of |H|^2 dV   (field='H')
        Concrete subclasses may override with an analytic fast path for
        specific region shapes; callers should never need to know which
        path was taken.
        """
        pts, w = region.quadrature_points(n_points)
        vals = self.E(pts) if field == "E" else self.H(pts)
        integrand = np.sum(np.abs(vals) ** 2, axis=-1)  # |vector|^2 per point
        return np.sum(w * integrand)
```

```python
class AnalyticalField(FieldProvider):
    """Thin wrapper around a Module-1 CavityMode."""
    def __init__(self, mode: "CavityMode"):
        self._mode = mode

    def E(self, r): return self._mode.E(r)
    def H(self, r): return self._mode.H(r)
    def total_stored_energy(self): return self._mode.total_stored_energy()
    @property
    def f0(self): return self._mode.f0
    def Q_wall(self, Rs): return self._mode.Q_wall(Rs)

    # Optional fast path example (only implement where closed forms exist):
    def integrate_field_energy(self, region, field, n_points=2000):
        closed_form = _lookup_closed_form(self._mode, region, field)
        if closed_form is not None:
            return closed_form
        return super().integrate_field_energy(region, field, n_points)
```

```python
class RitzField(FieldProvider):
    """STUB for later phase. Fixing the contract now so Modules 3-5 never
    need to change when this is implemented.

    Internally this will hold a set of basis functions and solve the
    generalized eigenproblem (Sec 7-6 Ritz procedure) for coefficients A_i.
    From the outside it is indistinguishable from AnalyticalField."""
    def __init__(self, basis_functions, coefficients):
        raise NotImplementedError("Deferred to sample-size-correction phase")

    def E(self, r): ...
    def H(self, r): ...
    def total_stored_energy(self): ...
    @property
    def f0(self): ...
    def Q_wall(self, Rs): ...
```

### Contract to Module 4

Module 4 depends **only** on `FieldProvider` — never on `CavityMode`, never on
`RectangularCavity` etc. directly. This is what lets `PerturbationModel` be written
once and reused unchanged when the field provider later becomes `RitzField` or an FEM
wrapper.

### Test plan
- For `AnalyticalField` wrapping a known `CavityMode`, confirm
  `integrate_field_energy` over a region covering the *whole* cavity volume equals
  `2 * total_stored_energy` split appropriately between E and H parts (sanity check on
  quadrature correctness and point density).
- Confirm the scale-invariance property from Section 0 here: multiply the underlying
  mode's `E,H` by a constant, and check ratios (not raw integrals) stay fixed.

---

## Module 3 — `sample`: Geometry + Material

### Strategy

Two independent concerns bundled into one module because they're always used together:
**where** the sample is and what shape it has (needed for the integration region and
for quasi-static depolarization correction), and **what** it's made of (complex
$\epsilon,\mu$).

The depolarization correction matters even *before* you get to full Rayleigh–Ritz
sample-size correction: Harrington's quasi-static shape corrections (thin slab, sphere,
rod — his Fig. 7-3 discussion) are cheap, closed-form, and already extend classical
perturbation theory beyond the point-dipole limit. Building them into Module 3 now
means Module 4's formula is "small-sample perturbation + shape correction" from the
start, rather than a strict point-sample approximation you'd have to retrofit.

### Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

class SampleRegion(ABC):
    """Geometric region occupied by the sample, in the cavity-local frame."""

    @abstractmethod
    def contains(self, r: np.ndarray) -> np.ndarray:
        """Boolean mask, r: (N,3) -> (N,)"""

    @abstractmethod
    def volume(self) -> float: ...

    @abstractmethod
    def quadrature_points(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (points (M,3), weights (M,)) such that
           sum(weights * f(points)) approximates integral_region f dV.
           M need not equal n exactly (e.g. structured grids round up)."""

    @property
    @abstractmethod
    def shape_kind(self) -> str:
        """One of {'sphere','thin_slab_normal','thin_slab_tangential',
        'thin_rod','generic'} — used to select a depolarization formula.
        'generic' means: no closed-form correction available, fall back to
        the point-dipole (small-sample) limit and flag it (see Module 4)."""

@dataclass(frozen=True)
class Sphere(SampleRegion):
    center: np.ndarray
    radius: float
    # implements contains/volume/quadrature_points; shape_kind = 'sphere'

@dataclass(frozen=True)
class Cylinder(SampleRegion):
    center: np.ndarray
    axis: np.ndarray       # unit vector
    radius: float
    height: float
    # shape_kind = 'thin_rod' if height >> radius and axis aligned with
    # local E, else 'generic'

@dataclass(frozen=True)
class Slab(SampleRegion):
    center: np.ndarray
    normal: np.ndarray     # unit vector
    thickness: float
    extent: tuple[float, float]   # in-plane dimensions
    # shape_kind = 'thin_slab_normal' or 'thin_slab_tangential' depending
    # on caller-declared field orientation at the slab location
```

```python
@dataclass(frozen=True)
class Material:
    eps: complex   # eps' - j*eps'' ,  eps'' >= 0
    mu: complex    # mu'  - j*mu''  ,  mu''  >= 0

    @property
    def loss_tangent_e(self) -> float:
        return self.eps.imag and -self.eps.imag / self.eps.real  # note sign

    @classmethod
    def from_loss_tangent(cls, eps_r: float, tan_delta_e: float,
                           mu_r: float = 1.0, tan_delta_m: float = 0.0
                           ) -> "Material":
        return cls(eps=eps_r * (1 - 1j * tan_delta_e),
                    mu=mu_r * (1 - 1j * tan_delta_m))

@dataclass(frozen=True)
class Sample:
    region: SampleRegion
    material: Material

    def depolarization_factor(self, field: str) -> complex:
        """Returns the quasi-static correction multiplier relating the
        INTERNAL field in the sample to the UNPERTURBED external field at
        the sample's location, for field in {'E','H'}. E.g. for a sphere:
              E_internal / E0 = 3 / (eps_r + 2)
        For 'generic' shape_kind, returns 1.0 (point-dipole limit) — Module
        4 must surface this as an accuracy caveat, not silently assume it's
        exact.
        """
        ...
```

### Contract to Module 4

Module 4 asks a `Sample` for exactly three things: `region` (to hand to
`FieldProvider.integrate_field_energy`), `material` (the complex $\epsilon,\mu$ being
tested — note this is the *fit variable* in Module 5, so `Material` must be cheap to
construct many times), and `depolarization_factor` (correction multiplier). It never
touches quadrature details directly.

### Test plan
- Sphere depolarization factor reduces to Harrington's $3/(\epsilon_r+2)$ result and
  matches the worked example (spherical cavity + concentric dielectric sphere) you
  already have as ground truth.
- `quadrature_points` integrates a known function (e.g. constant 1) to recover
  `volume()` to within a specified tolerance, for every shape.

---

## Module 4 — `perturbation`: Forward Model

### Strategy

This is Harrington's cavity-material perturbation formula (Eq. 7-11 generalized to
complex $\epsilon,\mu$ for the loaded-Q case), evaluated using whatever `FieldProvider`
and `Sample` it's given — it must not know or care which concrete field provider is
plugged in. No Jacobian here per your instruction: `evaluate()` returns only the
complex resonance (equivalently $f_{calc}, Q_{calc}$), and Module 5 differentiates it
numerically.

Formula implemented, in the complex form appropriate for lossy samples:

$$\frac{\tilde\omega - \omega_0}{\omega_0} = -\,\frac{\displaystyle\int_{V_s}\big[(\epsilon-\epsilon_0)\,\kappa_E\, |E_0|^2 + (\mu-\mu_0)\,\kappa_H\,|H_0|^2\big]\,dV}{2\,W}$$

where $\kappa_E,\kappa_H$ are the depolarization factors from Module 3 (1.0 in the
point-dipole limit) and $W$ is `total_stored_energy()` from Module 2. This reduces
exactly to your background material's Eq. (7-11)/(7-13) when $\kappa=1$ and the sample
is small.

### Interface

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class PerturbationResult:
    f_calc: float          # Hz
    Q_calc: float          # unitless (inf if no loss at all, incl. wall)
    omega_tilde: complex   # rad/s, for anyone who wants the raw eigenvalue

class PerturbationModel:
    """Forward model: given a Sample (region + material), predict the
    perturbed (f, Q) using a given FieldProvider for the unperturbed field."""

    def __init__(self, field_provider: "FieldProvider", Rs_walls: float | None = None):
        """Rs_walls: surface resistance [Ohm] for wall-loss Q contribution;
        None means treat walls as loss-free (Q determined by sample only)."""
        self._fp = field_provider
        self._Rs = Rs_walls

    def evaluate(self, region: "SampleRegion", material: "Material",
                 sample: "Sample" = None) -> PerturbationResult:
        """sample bundles region+material+depolarization; passing region
        and material separately is also supported since material is what
        Module 5 varies while region stays fixed across a fit."""
        ...
```

Design note on the split signature: Module 5 calls this **many** times per fit
(hundreds of forward evaluations), varying only `material` while `region` (and hence
quadrature points, and hence `integrate_field_energy` for $E$ and $H$ *shape
integrals*, which don't depend on $\epsilon,\mu$) stays fixed. Precompute and cache
$\int_{V_s}|E_0|^2\,dV$ and $\int_{V_s}|H_0|^2\,dV$ once per `(field_provider, region)`
pair — this is the actual expensive part (quadrature over possibly thousands of
points), and it's *material-independent*. `PerturbationModel` should own this cache
internally, keyed on `id(region)` or a region hash, so Module 5 doesn't have to know to
optimize this itself:

```python
class PerturbationModel:
    def __init__(self, field_provider, Rs_walls=None):
        self._fp = field_provider
        self._Rs = Rs_walls
        self._cache: dict[int, tuple[complex, complex]] = {}  # region id -> (IE, IH)

    def _shape_integrals(self, region):
        key = id(region)
        if key not in self._cache:
            IE = self._fp.integrate_field_energy(region, "E")
            IH = self._fp.integrate_field_energy(region, "H")
            self._cache[key] = (IE, IH)
        return self._cache[key]
```

### Contract to Module 5

Module 5 depends only on `PerturbationModel.evaluate(region, material) ->
PerturbationResult`. It does not know about `FieldProvider`, `CavityMode`, or
quadrature at all. This is what makes the inverse solver forward-model-agnostic (swap
in a Ritz- or FEM-backed `field_provider` later, Module 5 is untouched).

### Test plan
- Small-sample limit sanity check: shrink a `Sphere` region toward a point and confirm
  `evaluate()` converges to the closed-form point-dipole result from your background
  material (the worked spherical-cavity example).
- Passivity guard: reject/flag materials with $\epsilon'' < 0$ or $\mu'' < 0$ at the
  `evaluate()` boundary — a sign error here silently produces $Q < 0$, which is a
  confusing failure mode two modules downstream.

---

## Module 5 — `inverse`: Nonlinear Least-Squares Fit

### Strategy

Wrap `PerturbationModel.evaluate` in a residual function and let `scipy.optimize.least_squares`
handle both the optimization and the numerical Jacobian (`jac='2-point'` or `'3-point'`)
— explicitly deferring the analytic Jacobian we discussed earlier. Fit in $(f, 1/Q)$
rather than $(f, Q)$: the loss contribution is linear in $1/Q$ (and in $\epsilon''$),
which keeps the residual surface well-scaled and avoids compressing the informative
range of $Q$, per the earlier discussion — this is a modeling choice independent of
whether the Jacobian is analytic or numeric, so keep it even while ignoring the
Jacobian machinery.

Support multiple stacked measurements (different modes, or the sample moved to
different locations) from the start, since a single $(f,Q)$ pair generally
under-determines $(\epsilon',\epsilon'',\mu',\mu'')$ simultaneously — even without
formal identifiability analysis (that's Module 6, out of scope here), the solver
interface should not have to change shape when you go from fitting $\epsilon$ alone
(1–2 unknowns, 1 measurement) to fitting $\epsilon$ and $\mu$ together (needs $\ge 2$
measurements at different field ratios).

### Interface

```python
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares

@dataclass(frozen=True)
class Measurement:
    model: "PerturbationModel"     # bound to the mode/geometry this reading came from
    region: "SampleRegion"         # sample position/shape at time of this reading
    f_meas: float                  # Hz
    Q_meas: float
    sigma_f: float = 1e-4          # relative or absolute — pick one, document it
    sigma_invQ: float = 1e-3

@dataclass
class FitResult:
    eps: complex
    mu: complex
    success: bool
    residual_norm: float
    n_measurements: int
    raw: object   # scipy OptimizeResult, kept for diagnostics

class InverseSolver:
    def __init__(self, measurements: list[Measurement],
                 fit_mu: bool = False):
        """fit_mu: if False, mu is held fixed at 1 (typical dielectric
        characterization) and only (eps', eps'') are fit — reduces the
        unknown vector from 4 to 2 and sidesteps the identifiability
        problem entirely for the common case."""
        self._meas = measurements
        self._fit_mu = fit_mu

    def _unpack(self, p: np.ndarray) -> "Material":
        if self._fit_mu:
            eps = p[0] - 1j * p[1]
            mu = p[2] - 1j * p[3]
        else:
            eps = p[0] - 1j * p[1]
            mu = 1.0 - 0j
        return Material(eps=eps, mu=mu)

    def _residuals(self, p: np.ndarray) -> np.ndarray:
        material = self._unpack(p)
        res = []
        for m in self._meas:
            r = m.model.evaluate(m.region, material)
            res.append((r.f_calc - m.f_meas) / m.sigma_f)
            res.append((1.0 / r.Q_calc - 1.0 / m.Q_meas) / m.sigma_invQ)
        return np.array(res)

    def fit(self, initial_guess: "Material | None" = None) -> FitResult:
        p0 = self._initial_guess_vector(initial_guess)
        bounds = self._bounds()
        result = least_squares(self._residuals, p0, bounds=bounds,
                                method="trf", jac="2-point")
        material = self._unpack(result.x)
        return FitResult(eps=material.eps, mu=material.mu,
                          success=result.success,
                          residual_norm=float(np.linalg.norm(result.fun)),
                          n_measurements=len(self._meas), raw=result)

    def _bounds(self):
        # eps' >= 1, eps'' >= 0 (passivity); mu similarly if fit_mu
        lo = [1.0, 0.0] + ([1.0, 0.0] if self._fit_mu else [])
        hi = [np.inf, np.inf] + ([np.inf, np.inf] if self._fit_mu else [])
        return (lo, hi)

    def _initial_guess_vector(self, guess):
        if guess is not None:
            base = [guess.eps.real, -guess.eps.imag]
            if self._fit_mu:
                base += [guess.mu.real, -guess.mu.imag]
            return np.array(base)
        # Fallback: classical small-sample closed-form estimate from the
        # FIRST measurement alone (rough, but a much better start than 1.0)
        return self._closed_form_seed()

    def _closed_form_seed(self) -> np.ndarray:
        """Cheap non-iterative estimate to seed least_squares, using the
        first measurement's model/region and the point-dipole formula
        directly (bypassing PerturbationModel's general path). Keeps
        the optimizer's first Gauss-Newton step close to converged."""
        ...
```

### Contract from Module 4

`InverseSolver` never imports `FieldProvider`, `CavityMode`, or quadrature code — it
only calls `Measurement.model.evaluate(region, material)`. This is the payoff of the
Module 4 seam: when the analytic Jacobian (or a Ritz-backed model) is added later,
`InverseSolver` changes in exactly one place — swap `jac='2-point'` for an analytic
`jac=` callable — and nothing else in this module moves.

### Test plan (most important one in the whole system)
**Synthetic-data recovery test**: pick a known $(\epsilon',\epsilon'')$, run it through
`PerturbationModel.evaluate` to generate synthetic $(f_{meas}, Q_{meas})$, optionally
add noise at the level of your expected measurement precision, then confirm
`InverseSolver.fit` recovers the known value within a tolerance consistent with that
noise. This one test exercises the full module chain (1→2→3→4→5) end to end and is
your regression guard for every future change.

---

## Cross-module integration checklist

- [ ] `AnalyticalField(CavityMode)` round-trips `f0`, `Q_wall`, `total_stored_energy`
      unchanged (Module 2 adds zero distortion over Module 1 alone).
- [ ] Scale invariance holds end-to-end: scaling a `CavityMode`'s raw field amplitude
      changes nothing in `PerturbationResult`.
- [ ] Point-dipole limit: `Sample` with `shape_kind='generic'` (depolarization = 1)
      reproduces the textbook small-sample formulas exactly.
- [ ] Passivity is enforced at the `PerturbationModel.evaluate` boundary, not just in
      `InverseSolver`'s bounds — so any future caller of Module 4 gets the same guard.
- [ ] Synthetic-recovery test passes for at least: (a) lossless-ish dielectric at
      E-max, (b) lossy dielectric at E-max, (c) two stacked measurements with `fit_mu=True`.
