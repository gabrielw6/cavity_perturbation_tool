# Cavity Perturbation Measurement Toolkit

A Python toolkit for designing and analyzing **cavity perturbation** experiments: measuring
a material's complex permittivity ($\epsilon$), permeability ($\mu$), and loss tangent by
placing a small sample inside a resonant cavity and observing how it shifts the resonant
frequency and degrades the quality factor ($Q$).

It answers two questions:

- **Forward**: given a cavity, a mode, and a *known* sample material, what resonant
  frequency and $Q$ do we expect? (`perturbation.py`)
- **Inverse**: given *measured* resonant frequency and $Q$ (with the sample in place),
  what material properties does that imply? (`inverse.py`)

> **Status**: core pipeline (cavity fields → sample integration → forward model → inverse
> fit) is implemented for rectangular, cylindrical, and coaxial cavities and for spherical,
> cylindrical, and slab-shaped samples. See **Limitations** below for what isn't there yet.
> Contributors: see `CLAUDE.md` and `docs/` for the full equations and internal design.

---

## Installation

```bash
pip install -e .
```

Requires Python 3.11+, NumPy, SciPy. No other dependencies.

---

## Coordinate conventions

Every cavity places its own local origin: a `RectangularCavity` has one corner at the
origin, with the cavity occupying $[0,a]\times[0,b]\times[0,c]$; `CylindricalCavity` and
`CoaxialCavity` are centered on the $z$-axis, spanning $z\in[0,d]$ (or $[0,L]$). Sample
positions you supply (`Sphere.center`, etc.) are in this same local frame, in meters.

---

## Quickstart: recovering a sample's permittivity

```python
import numpy as np
from cavity_perturbation.cavity import RectangularCavity, ModeIndex
from cavity_perturbation.fields import AnalyticalField
from cavity_perturbation.sample import Sphere
from cavity_perturbation.perturbation import PerturbationModel
from cavity_perturbation.inverse import Measurement, InverseSolver

# 1. Describe the empty cavity and the mode you're using (air-filled: defaults to vacuum).
cavity = RectangularCavity(a=0.09, b=0.04, c=0.12, mode=ModeIndex("TE", (0, 1, 1)))
field = AnalyticalField(cavity)

# 2. Where and how big is your sample? (Small relative to the cavity — see Limitations.)
sample_region = Sphere(center=np.array([0.045, 0.02, 0.06]), radius=0.002)

# 3. Build the forward model. Rs_walls is optional (copper ~0.02 Ohm at a few GHz);
#    omit it if you don't want wall loss included in Q_wall.
model = PerturbationModel(field, Rs_walls=0.02)

# 4. Record what you measured on the VNA with the sample in place, and how precisely.
measurement = Measurement(
    model=model,
    region=sample_region,
    f_meas=2.998e9,      # Hz
    Q_meas=4200,
    sigma_f=1e-4,        # fractional: 100 ppm frequency precision
    sigma_Q=1e-2,        # fractional: 1% Q precision
)

# 5. Fit. fit_mu=False assumes a non-magnetic sample (mu_r=1) and fits (eps', eps'') only.
result = InverseSolver([measurement], fit_mu=False).fit()

print(f"eps_r = {result.eps.real:.3f} - j{-result.eps.imag:.4f}")
print(f"loss tangent = {-result.eps.imag/result.eps.real:.4f}")
print(f"converged: {result.success}, condition number: {result.condition_number:.1e}")
```

`result.condition_number` is a diagnostic, not just a number to ignore: a large value
(rule of thumb, $>10^6$) means this measurement doesn't tightly constrain the fitted
parameters — see **Multi-mode fitting** below before trusting the numbers in that case.

---

## Forward problem: predicting $f$, $Q$ for a known material

Useful for planning an experiment before you run it — e.g. checking how much frequency
shift to expect for a given sample size.

```python
from cavity_perturbation.sample import Sample, Material

material = Material.from_loss_tangent(eps_r=10.2, tan_delta_e=0.002)
sample = Sample(region=sample_region, material=material)

prediction = model.evaluate(sample)
print(f"expected f = {prediction.f_calc/1e9:.4f} GHz, Q = {prediction.Q_calc:.0f}")
```

---

## Choosing where to put your sample

Placement isn't arbitrary — the correction applied for a finite-size sample
(`sample.py`'s depolarization factor) only has a clean closed form when the sample's
characteristic axis is **aligned with** or **perpendicular to** the local field at its
location (within about 10°). Place a rod-shaped sample along the $E$-field maximum (or
exactly transverse to it), not at an oblique angle — an oblique placement silently falls
back to a less accurate point-dipole estimate rather than failing outright, so it's worth
getting the placement right rather than relying on the fallback.

A sphere has no such restriction (its correction is orientation-independent).

---

## Multi-mode fitting (recovering both $\epsilon$ and $\mu$)

A single $(f,Q)$ measurement can't separate $\epsilon$ and $\mu$ — there are two unknowns'
worth of information (real and imaginary parts of each) and only one complex measurement.
To fit both, pass `fit_mu=True` **and** at least two `Measurement`s that see a different
mix of $E$ and $H$ at the sample (different modes, or the sample moved between an
$E$-field maximum and an $H$-field maximum):

```python
solver = InverseSolver([measurement_at_E_max, measurement_at_H_max], fit_mu=True)
result = solver.fit()
```

If the two measurements don't actually differ enough in field character, the fit will
still return *a* result, but `result.condition_number` will be large — check it rather
than assuming convergence means the answer is well-determined.

---

## Module reference

| Module | Purpose |
|---|---|
| `cavity.py` | Exact field solutions for rectangular, cylindrical, and coaxial cavity resonances: $E$, $H$, $f_0$, $Q_{\text{wall}}$, stored energy. |
| `fields.py` | Uniform interface for evaluating fields and integrating $\int_{V_s}\lvert E\rvert^2\,dV$ over a sample region, regardless of the underlying field model. |
| `sample.py` | Sample geometry (`Sphere`, `Cylinder`, `Slab`), material (`Material`), and the depolarization correction for finite sample size. |
| `perturbation.py` | Forward model: predicts $f$, $Q$ for a given cavity mode and sample. |
| `inverse.py` | Fits measured $f$, $Q$ to recover $\epsilon$, $\mu$. |

Full equations, derivations, and validation targets for each module are in `docs/`.

---

## Limitations

- **Small-sample assumption**: accuracy is guaranteed for samples small relative to the
  cavity (the classical perturbation regime). A dedicated large-sample correction
  (Rayleigh–Ritz trial fields) is planned but not yet implemented — for now, keep sample
  volume well under 1% of the cavity volume as a rule of thumb, or expect increasing error
  above that.
- **Coaxial cavities**: only the fundamental TEM mode family is supported (this is also
  the only family used in practice for this kind of measurement).
- **Oblique sample orientation**: falls back to a less accurate point-dipole estimate
  rather than the shape-specific correction (see **Choosing where to put your sample**).
- **No sensitivity maps yet**: there's no built-in tool to answer "where in the cavity is
  measurement sensitivity highest" beyond the guidance above — this is planned future work.
- Python API only; no graphical interface.
