# Module 3 — Sample: Geometry + Material — Equations & Implementation Plan

Scope: `SampleRegion` (`Sphere`, `Cylinder`, `Slab`), `Material`, `Sample`, per the interface
in `architecture_modules_1-5.md`. No code — equations and build order only.

**Read Section 0 first.** It flags a real gap in the architecture doc's `depolarization_factor`
signature that needs resolving before the rest of this module makes sense — everything after
it is written against the corrected version, not the original sketch.

---

## 0. Design correction: `depolarization_factor` needs a field direction

The architecture doc's sketch has `Sample.depolarization_factor(field: str) -> complex`, and
separately has `Cylinder.shape_kind` depend on whether "axis [is] aligned with local E" — but
alignment-with-E is a relationship between the *shape* and the *field*, and nothing in
`SampleRegion` (which is documented as purely geometric — "region occupied by the sample," no
mention of fields) has access to a field direction. As written, `shape_kind` can't actually
compute what its own docstring says it depends on.

**Fix**: split the two concerns cleanly.

- `SampleRegion.shape_kind` becomes **purely geometric** — computed from the region's own
  dimensions only (aspect ratio), never from field information. New value set:
  `{'sphere', 'thin_rod', 'thin_disk', 'generic'}` (Section 3 gives the exact aspect-ratio
  thresholds). A `Cylinder` that's long and thin is a candidate for the rod-type correction;
  whether that correction actually *applies* depends on how the sample sits relative to the
  local field, which is Section 2's job, not this property's.
- `Sample.depolarization_factor` gains a required field-direction argument:
  `depolarization_factor(field_type: str, field_direction: np.ndarray) -> complex`, where
  `field_direction` is the (not necessarily unit) local $E_0$ or $H_0$ vector at the sample's
  center, as evaluated by whichever `FieldProvider` Module 4 is using. **Module 4 is
  responsible for evaluating and passing this in** — `Sample`/`SampleRegion` never construct
  or hold a `FieldProvider` reference themselves, keeping Module 3 free of any dependency on
  Module 2. This is a one-line change to Module 4's calling convention, noted here so it's
  fixed *before* Module 4 is built against the old signature.

---

## 1. `Material`

### 1.1 Convention recap

$\epsilon=\epsilon'-j\epsilon''$, $\mu=\mu'-j\mu''$, both $\epsilon'',\mu''\ge0$ (fixed
project-wide in `CLAUDE.md`). Nothing new here — restated because every formula below depends
on getting this sign right.

### 1.2 Loss tangent

$$\tan\delta_e = \frac{\epsilon''}{\epsilon'} = \frac{-\,\mathrm{Im}(\epsilon)}{\mathrm{Re}(\epsilon)}, \qquad \tan\delta_m = \frac{\mu''}{\mu'} = \frac{-\,\mathrm{Im}(\mu)}{\mathrm{Re}(\mu)}$$

The architecture sketch only has `loss_tangent_e`, uses a fragile `a and b` short-circuit
idiom to handle the zero-loss case, and doesn't handle $\mathrm{Re}(\epsilon)\le0$. Fix:

- Add `loss_tangent_m`, symmetric with `loss_tangent_e` — there's no reason to have one
  without the other; Module 5 will need both once it fits $\mu$.
- Replace the `and`-idiom with an explicit conditional. It happens to produce the right answer
  for the zero-loss case, but reads as unintentional and doesn't extend cleanly.
- $\mathrm{Re}(\epsilon)\le0$ isn't handled by the sketch at all (silent `ZeroDivisionError` or
  a nonsensical negative loss tangent) — raise explicitly here rather than let a downstream
  module receive a bad number.

### 1.3 `from_loss_tangent` construction

$$\epsilon = \epsilon_r(1-j\tan\delta_e), \qquad \mu = \mu_r(1-j\tan\delta_m)$$

Direct from 1.2's definition solved for $\epsilon$ given $\epsilon_r=\epsilon'$ and
$\tan\delta_e$; already correctly signed in the architecture sketch, no change needed —
included here only so the forward and inverse directions (1.2 and 1.3) are documented
together and visibly consistent with each other.

### 1.4 Passivity

$$\text{is\_passive} \iff \epsilon''\ge0 \ \wedge\ \mu''\ge0 \ \wedge\ \epsilon'>0 \ \wedge\ \mu'>0$$

This is a **fundamental physical validity check** (causality/passivity for a linear medium),
and belongs on `Material` as a property — `Material` owns *what* passivity means. It is
distinct from Module 5's optimizer bounds (e.g. $\epsilon'\ge1$), which are a *fitting prior*
for typical measurement targets, not a physical law — don't conflate the two. `CLAUDE.md`'s
"passivity guard... at the `perturbation.py` evaluate boundary" means Module 4 calls
`material.is_passive` at that boundary; it does not mean Module 3 re-derives the check itself
each place it's needed.

### 1.5 Implementation steps

1. Implement `Material` with `eps`, `mu` fields exactly as sketched.
2. Implement `loss_tangent_e`, `loss_tangent_m` per 1.2, with the explicit-conditional and
   $\mathrm{Re}\le0$ fixes.
3. Implement `from_loss_tangent` per 1.3.
4. Implement `is_passive` per 1.4.
5. Unit test: round-trip `from_loss_tangent` → `loss_tangent_e` recovers the original
   $\tan\delta_e$ to floating-point precision, for several $(\epsilon_r,\tan\delta)$ pairs
   including $\tan\delta=0$.

---

## 2. Depolarization factor — the master equation

### 2.1 General formula

For a canonical (ellipsoidal-limit) shape in a uniform external field $E_0$ aligned with one
of its principal axes, with geometric depolarization factor $N$ along that axis
($0\le N\le1$):

$$\frac{E_{\text{in}}}{E_0} = \frac{1}{1+N(\epsilon_r-1)}, \qquad \frac{H_{\text{in}}}{H_0} = \frac{1}{1+N(\mu_r-1)}$$

This single formula is the entire depolarization module — every canonical shape/orientation
below is just a different value of $N$ plugged into the same expression. This mirrors the
"one recipe, many eigenfunctions" pattern already used in Module 1 (§0.1) and is why it's
worth implementing as one parametrized function rather than four separately-derived cases.

**Complex generalization**: $\epsilon_r,\mu_r$ enter as complex numbers directly — the
quasi-static derivation holds unmodified for complex (lossy) material, so no separate
treatment is needed for the loss tangent.

### 2.2 $N$-value table

Derived from the electrostatic ellipsoid boundary-value problem; every value below is cross-
checked against the general identity $\sum_i N_i = 1$ over an ellipsoid's three principal axes:

| Shape | Field orientation | $N$ | Check |
|---|---|---|---|
| Sphere | any (isotropic) | $1/3$ | $3\times\frac13=1$ |
| Infinite slab | normal to face | $1$ | $1+0+0=1$ |
| Infinite slab | tangential to face | $0$ | (two tangential directions share the remaining $0$) |
| Infinite thin rod | axial ($E\parallel$ axis) | $0$ | |
| Infinite thin rod | transverse ($E\perp$ axis) | $1/2$ | $0+\frac12+\frac12=1$ |

Sphere: $E_{\text{in}}/E_0=3/(\epsilon_r+2)$. Slab-normal: $E_{\text{in}}/E_0=1/\epsilon_r$.
Slab-tangential and rod-axial: $E_{\text{in}}/E_0=1$ (no correction — tangential $E$ is
continuous across the interface in both cases, which is *why* $N=0$; this is a boundary-
condition fact, not an approximation, distinct from the "generic, unknown" fallback below
even though both numerically return $1.0$). Rod-transverse: $E_{\text{in}}/E_0=2/(\epsilon_r+1)$.

### 2.3 Oblique-field scope decision

For a non-spherical shape, $N$ is only single-valued (and the formula above only applies as a
scalar) when the field is aligned with or perpendicular to the shape's axis/normal — at an
oblique angle, the internal field is a tensor operation on the external field, not a scalar
multiple, and `depolarization_factor` cannot return a physically meaningful single number.
Cavity perturbation samples are, in practice, deliberately oriented along a field extremum for
exactly this reason (aligned or perpendicular placement, never oblique), so this isn't a
significant scope restriction — but it must be checked, not assumed:

$$\cos\theta = \frac{|\hat n \cdot \hat F|}{\lvert\hat n\rvert\,\lvert\hat F\rvert}, \qquad \hat n = \text{shape's axis (rod) or normal (slab)}, \quad \hat F = \texttt{field\_direction}$$

With tolerance $\theta_{\text{tol}}=10°$ (named constant, not a magic number):
- $\theta<\theta_{\text{tol}}$: axial/normal case, use the corresponding $N$ from 2.2.
- $\theta>90°-\theta_{\text{tol}}$: transverse/tangential case, use the corresponding $N$.
- otherwise: fall back to `shape_kind='generic'` behavior (2.4) and flag it — the sample is
  misoriented relative to what the closed-form correction assumes.

### 2.4 `'generic'` fallback

Returns $1.0$ (no correction — equivalent to $N=0$, i.e. assumes point-dipole/small-sample
behavior). This is a **weaker approximation than any canonical-shape entry in 2.2**, even
though sphere/rod-axial/slab-tangential also sometimes evaluate to a similar or identical
number — those are *derived* results for a specific boundary-value problem; `'generic'` is an
*assumption of negligible correction*, valid only when the sample is small compared to the
field-variation scale. Module 4/5 should treat `'generic'`-shape samples as carrying strictly
larger, undocumented model error than any resolved canonical case — this connects forward to
the (not-yet-designed) sample-size-correction study, which is specifically about quantifying
how bad this fallback gets as sample size grows.

### 2.5 Implementation steps

1. Implement the master formula (2.1) as a single function of `(N, eps_or_mu_r)`.
2. Implement the $N$-lookup (2.2) as a small table keyed on `(shape_kind, orientation)`.
3. Implement the angle test (2.3) and the axial/transverse/generic branch selection.
4. Wire `Sample.depolarization_factor(field_type, field_direction)` to: determine
   `self.region.shape_kind`; if `'sphere'`, return the isotropic value directly (no angle
   test needed); otherwise run the angle test against the region's `axis`/`normal` attribute
   and dispatch to 2.2 or the `'generic'` fallback.
5. Unit test each row of the 2.2 table independently, plus the sum-rule cross-check
   ($N$ values used must satisfy $\sum N_i=1$ for the underlying ellipsoid limit) as a
   standing regression test on the table itself, not just on the formula.

---

## 3. `SampleRegion` geometry

### 3.1 Shared building block: 1-D Gauss–Legendre on an arbitrary interval

$$\{\xi_i, w_i\}_{i=1}^{n} = \texttt{numpy.polynomial.legendre.leggauss}(n) \quad\text{(nodes/weights on }[-1,1]\text{)}$$
$$x_i = \frac{b-a}{2}\xi_i + \frac{a+b}{2}, \qquad w_i' = \frac{b-a}{2}w_i \quad\text{(affine map to }[a,b]\text{)}$$

Implement once as `gauss_legendre(n, a, b) -> (points, weights)`; every region's radial and
axial quadrature directions reuse it directly (this is the same "implement the identity once,
reuse across geometries" discipline as Module 1 §0.4).

### 3.2 Local-frame → cavity-frame rigid transform

`Cylinder` and `Slab` are naturally parametrized in a local frame aligned with their `axis`/
`normal`; quadrature points must be generated in that local frame (where the shape is
axis-aligned and simple) and then rotated/translated into the cavity's Cartesian frame.

Given a unit vector $\hat n$ (the `axis` or `normal`), build an orthonormal basis
$\{\hat e_1,\hat e_2,\hat n\}$:
1. Pick a reference vector $\hat u = \hat x$, unless $|\hat n\cdot\hat x| > 0.9$ (near-parallel,
   degenerate cross product), in which case use $\hat u=\hat y$ instead.
2. $\hat e_1 = \dfrac{\hat u - (\hat u\cdot\hat n)\hat n}{\lVert \hat u - (\hat u\cdot\hat n)\hat n\rVert}$ (Gram–Schmidt).
3. $\hat e_2 = \hat n\times\hat e_1$.

The specific choice of $\hat e_1,\hat e_2$ within the plane perpendicular to $\hat n$ is
arbitrary and doesn't matter physically — both `Cylinder` and `Slab` are symmetric (or treated
as such, per Section 2.3's scope decision) under rotation about $\hat n$, so any valid
orthonormal completion is correct. Points transform as
$\mathbf r_{\text{cavity}} = \texttt{center} + \xi_1\hat e_1+\xi_2\hat e_2+\xi_3\hat n$ for
local coordinates $(\xi_1,\xi_2,\xi_3)$.

### 3.3 Sphere

- `volume()`: $\dfrac{4}{3}\pi R^3$.
- `contains(r)`: $\lVert \mathbf r-\texttt{center}\rVert \le R$.
- `quadrature_points(n)`: tensor grid in $(r,u=\cos\theta,\phi)$, using $dV = r^2\,dr\,du\,d\phi$
  (the $u=\cos\theta$ substitution absorbs $\sin\theta\,d\theta$ exactly, so no extra Jacobian
  factor beyond $r^2$):
  - $r$-direction: `gauss_legendre(n_r, 0, R)`, with weights additionally multiplied by $r_i^2$.
  - $u$-direction: `gauss_legendre(n_u, -1, 1)` directly (no extra factor).
  - $\phi$-direction: $n_\phi$ equally spaced points on $[0,2\pi)$, uniform weight
    $2\pi/n_\phi$ each (a periodic trapezoid rule, which is spectrally accurate for smooth
    periodic integrands — standard result, not an approximation to worry about further).
  - Final weight per 3-D point: product of the three 1-D weights. Convert
    $(r,u,\phi)\to(x,y,z)$ via $z=ru$, $\rho=r\sqrt{1-u^2}$, $x=\rho\cos\phi$, $y=\rho\sin\phi$,
    then add `center`.

### 3.4 Cylinder

- `volume()`: $\pi R^2 H$.
- `contains(r)`: transform to local $(\rho,\phi,z)$ via 3.2's frame; test
  $\rho\le R \wedge |z|\le H/2$.
- `quadrature_points(n)`: tensor grid in local $(\rho,\phi,z)$, $dV=\rho\,d\rho\,d\phi\,dz$:
  - $\rho$-direction: `gauss_legendre(n_\rho, 0, R)`, weights multiplied by $\rho_i$.
  - $\phi$-direction: same periodic-uniform rule as 3.3.
  - $z$-direction: `gauss_legendre(n_z, -H/2, H/2)` directly.
  - Transform local $(\rho,\phi,z)\to(x,y,z)_{\text{local axes}}$ then to cavity frame via 3.2.
- `shape_kind`: purely from aspect ratio $H/(2R)$ — suggested thresholds (named constants):
  $>5$: `'thin_rod'`; $<0.2$: `'thin_disk'`; else `'generic'`.

### 3.5 Slab

- `volume()`: $\texttt{thickness}\times\texttt{extent}[0]\times\texttt{extent}[1]$.
- `contains(r)`: transform to local $(\xi_1,\xi_2,\xi_3=\text{along }\hat n)$ via 3.2; test
  $|\xi_3|\le\texttt{thickness}/2 \wedge |\xi_1|\le\texttt{extent}[0]/2 \wedge |\xi_2|\le\texttt{extent}[1]/2$.
- `quadrature_points(n)`: tensor grid, all three directions plain `gauss_legendre` (Cartesian,
  no Jacobian curvature factor — the box is axis-aligned in local coordinates), transformed
  via 3.2.
- `shape_kind`: aspect ratio $\texttt{thickness}/\min(\texttt{extent})$ — suggested threshold
  $<0.2$: `'thin_disk'` (the slab is behaving like the canonical "thin" limit Section 2 assumes);
  else `'generic'` (thickness comparable to lateral extent — the flat-sample approximation
  itself is questionable, and this should be treated the same as any other `'generic'` case,
  not silently given the slab formula anyway).

### 3.6 Implementation steps

1. Implement `gauss_legendre` (3.1), unit test against `scipy.integrate.quad` on a few
   polynomials and one non-polynomial smooth function.
2. Implement the local-frame construction (3.2) as a standalone function
   `orthonormal_frame(n_hat) -> (e1, e2, n_hat)`; unit test it directly for degeneracy
   handling (pass in $\hat n=\hat x$ and confirm the near-parallel branch triggers correctly).
3. Implement `Sphere` (3.3) — no dependency on 3.2, so it can be validated independently first.
4. Implement `Cylinder` (3.4), reusing 3.1 and 3.2.
5. Implement `Slab` (3.5), reusing 3.1 and 3.2.
6. For every shape, confirm the point-count-splitting rule ($n_i\approx n^{1/3}$ per tensor
   direction, rounded up) produces a valid grid for small `n` (e.g. `n=8`) without any
   direction collapsing to zero points.
7. Run the Section 5 validation suite.

---

## 4. Overall build order

1. `Material` (Section 1) — no dependency on anything else in this module, build and test first.
2. Shared geometry primitives (3.1, 3.2) — geometry-agnostic, build and test second.
3. `Sphere` (3.3) — simplest region, validates the primitives.
4. `Cylinder` (3.4), then `Slab` (3.5).
5. Depolarization (Section 2) last — it depends on `shape_kind` from step 3/4 and consumes
   the corrected `field_direction`-aware signature from Section 0.

## 5. Validation targets

- **Volume consistency**: for every shape, at several resolutions `n`, confirm
  $\sum_i w_i \to \texttt{volume()}$ to quadrature tolerance — this is the Module 2 §1.4 check,
  but it needs to be validated from *this* side too (against a `SampleRegion` known-good
  reference) before Module 2 ever consumes it.
- **`contains` / quadrature agreement**: every generated quadrature point should satisfy
  `contains(point) == True` for that same region (a point outside the region contributing
  nonzero weight is a frame-transform or bounds bug).
- **Depolarization table** (2.5): each of the five rows in 2.2, checked independently, plus
  the $\sum N_i=1$ cross-check.
- **Frame-transform round-trip**: for `Cylinder`/`Slab`, transform a point local→cavity→local
  and confirm it returns to the original coordinates (catches a sign error in the rotation
  construction, which — per the Module 1 review — is exactly the class of bug most likely to
  slip through if only checked in one direction).
- **`Material` round-trip** (1.5).
- **Angle-test boundary behavior** (2.3): confirm the $\theta_{\text{tol}}=10°$ branch
  boundaries produce the expected axial/transverse/generic classification on a few
  hand-picked `field_direction` vectors, including one deliberately at $45°$ (must resolve to
  `'generic'`).
