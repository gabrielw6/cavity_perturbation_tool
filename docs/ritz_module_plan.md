# Rayleigh–Ritz Module — Plan (final, cross-checked against the implemented codebase)

Scope: what becomes of the `RitzField` stub in `architecture_modules_1-5.md`. No code —
equations, interface audit, and test plan only.

**Read Section 0 first.** Working through the actual mathematics changes the shape of this
module significantly relative to the original stub — the stub's `FieldProvider`-shaped
`RitzField` turns out not to be the right interface for what this project's Ritz use case
(the "sample-size correction" goal from the original project scope) actually needs.

This version incorporates two corrections found only by checking this plan against the
actual, tested `cavity_perturbation` codebase rather than against Module 4's equations alone:
a factor-of-2 error in $K$/background-$M$ (§2.1–2.2, caught by cross-checking against Module
1's exact `total_stored_energy()` convention), and a missing conjugate in $\Delta M$ (§2.3,
the same fix `perturbation.py` needed, verified the same way — against the passivity
requirement, not by re-deriving the underlying reciprocity argument). It also adds an
interface requirement (§5) found only by reading `inverse.py` directly: matching
`PerturbationModel.evaluate`'s signature is not sufficient for a `RitzCorrectedModel`
measurement to work with Module 5's closed-form seed.

---

## 0. Scoping correction: what should this module compute?

### 0.1 Two different things "Rayleigh–Ritz" could mean here

The original background material motivated Ritz two ways: (a) an irregular cavity geometry
with no closed-form solution, and (b) a sample whose volume isn't negligible relative to the
cavity. The stub (`RitzField(FieldProvider)`, "indistinguishable from `AnalyticalField`")
was written for case (a) — a material-independent, reusable field provider.

But case (b) — which is what this project's "sample-size correction" study actually needs —
is fundamentally different, and it's worth deriving *why* rather than asserting it. Take
Module 1's exact modes of the same canonical (rectangular/cylindrical/coaxial) cavity as
trial basis functions for a Ritz expansion. Distinct modes of a uniform, empty cavity are
mutually orthogonal under both the electric and magnetic energy inner products — a standard
consequence of the Maxwell eigenvalue problem being self-adjoint for a lossless, uniform
medium. That means: if the medium in the Ritz eigenproblem is *still uniform* (case a, for
one of the *same* three canonical shapes), the mass/stiffness matrices come out exactly
diagonal — no mode-mixing occurs, and the "Ritz-improved" field is identical to the single
best-matching Module 1 mode. There's nothing to correct, because Module 1's closed form is
already exact for a uniform canonical cavity. Case (a) is only meaningful for a genuinely
irregular geometry — which is a mesh/finite-element problem (the FEM thesis chapter's
territory), not a small basis of exact eigenfunctions.

Case (b) is where the mathematics is actually non-trivial: the *sample* makes $\epsilon(\mathbf
r)$ non-uniform, which breaks the orthogonality between different basis modes and produces
genuine mode-mixing — capturing field distortion around the sample that a single-mode
quasi-static approximation (Module 4) can't see. This is the real content of "improves
accuracy where classical formulas assume $V_s\ll V_{\text{cavity}}$."

### 0.2 Resolution: a `PerturbationModel`-shaped class, not a `FieldProvider`-shaped one

Since the sample's material enters directly into the matrix assembly (Section 2), the useful
Ritz class takes a `Sample` and produces a corrected $(f,Q)$ directly — it is not a passive
field provider at all. Call it `RitzCorrectedModel`, implementing the same output contract as
`PerturbationModel`: **`evaluate(sample) -> PerturbationResult`**. This is the right place
for "indistinguishable from the outside" to apply — at the `PerturbationModel` level, not the
`FieldProvider` level — because it means `Measurement.model` (Module 5) can hold *either*
class with zero changes to `inverse.py`.

### 0.3 What happens to the original `RitzField(FieldProvider)` stub

Retire it. It isn't a case of "not yet built" — per 0.1, it was never going to do anything
useful for this project's three canonical geometries, and building it for genuinely irregular
geometries would duplicate the FEM chapter's job. Remove the stub from `fields.py`'s planned
contents; keep a short note in this module's own file (not buried in `CLAUDE.md`'s status
list) explaining why, for anyone who later wonders where it went.

### 0.4 Scope decision, flagged for override

This plan deliberately does **not** build a general irregular-geometry Ritz solver. If the
thesis structure actually wants one here rather than treating irregular geometries as the FEM
chapter's job, that's a substantially different (and larger) module — say so before Section 6
becomes an implementation task.

---

## 1. Basis functions

Basis = the mode of interest (the one being measured) plus the $N-1$ nearest-frequency other
modes of the *same* canonical cavity (same type, same dimensions) — all already available
from Module 1, no new field solutions to derive. Nearby modes dominate the mixing correction
(coupling strength falls off with frequency separation); a simple fixed default (e.g. $N=5$)
is a reasonable starting point, with an adaptive frequency-window criterion as a documented
future refinement rather than a first-pass requirement.

---

## 2. Matrix assembly

Trial field: $E(\mathbf r)=\sum_{i=1}^N A_i E_i(\mathbf r)$, using Module 1's exact modes
$E_i,H_i$, each satisfying $\nabla\times E_i=-j\omega_i\mu_{bg}H_i$ exactly for its own
eigenfrequency $\omega_i$. **Scope**: non-magnetic samples only ($\mu_r=1$), matching the
scope already used elsewhere in this project's examples — $\mu_{bg}$ stays uniform throughout,
so only $\epsilon(\mathbf r)$ becomes non-uniform via the sample.

### 2.1 $K$ (stiffness) is exactly diagonal

$$K_{ij}=\int_V\mu_{bg}^{-1}(\nabla\times E_i)\cdot(\nabla\times E_j)^*\,dV = \omega_i\omega_j\,\mu_{bg}\int_V H_i\cdot H_j^*\,dV$$

using the curl relation above. By the orthogonality argument (0.1), $\int_VH_i\cdot H_j^*dV=0$
for $i\ne j$. For $i=j$: Module 1's `total_stored_energy()` for mode $i$ alone is $W_i=W_{e,i}+W_{m,i}=2W_{m,i}$
(equipartition), and $W_{m,i}=\frac{\mu_{bg}}{2}\int|H_i|^2dV$ (Module 1's own energy-density
formula), so $\mu_{bg}\int|H_i|^2dV=2W_{m,i}=W_i$ — **not** $2W_i$; the total_stored_energy
value *already includes* the factor that a naive reading of the magnetic-only energy formula
would suggest doubling again. (An earlier draft of this section conflated "twice the magnetic
energy alone" with "`total_stored_energy()`," which are the same thing — the doubling was
already latent in the equipartition step and shouldn't be applied twice.) So:
$$K_{ij} = \omega_i^2 W_i\,\delta_{ij}$$
No new integral needed — built entirely from Module 1's existing `f0` and
`total_stored_energy()` for each basis mode.

### 2.2 $M$ (mass): background diagonal plus sample correction

By the identical argument (electric energy this time): $\epsilon_{bg}\int_VE_i\cdot E_j^*dV=W_i\delta_{ij}$.
$$M_{ij} = \underbrace{\epsilon_{bg}\int_VE_i\cdot E_j^*\,dV}_{=W_i\delta_{ij}} + \underbrace{\overline{(\epsilon_r-1)}\,\epsilon_{bg}\int_{V_s}E_i\cdot E_j^*\,dV}_{\Delta M_{ij},\text{ see 2.3 for the conjugate}}$$

### 2.3 $\Delta M_{ij}$ — the sample correction, a required conjugate, and a deliberate non-use of Module 3

$$\Delta M_{ij} = \overline{(\epsilon_r-1)}\,\epsilon_{bg}\int_{V_s}E_i(\mathbf r)\cdot E_j(\mathbf r)^*\,dV$$

**The conjugate on $(\epsilon_r-1)$ is required, not optional — verified the same way Module
4's identical fix was verified.** Without it, the $N=1$ (no mode-mixing) reduction of this
construction is $\dfrac{\tilde\omega-\omega_1}{\omega_1}\approx-\frac12(\epsilon_r-1)p_E^{(0)}$
— exactly Module 4's *pre-fix* formula in the point-dipole limit, which fails passivity (a
low-loss dielectric sample would show $\mathrm{Im}(\Delta)>0$, improving $Q$). Conjugating
$(\epsilon_r-1)$ here reduces $N=1$ to Module 4's *corrected* formula instead (module4 doc
§1.4) and passes the same check. Note that only the bare material contrast is conjugated —
the cross-overlap integral $\int_{V_s}E_i\cdot E_j^*\,dV$ itself is untouched, exactly
mirroring how Module 4 leaves $p_E,p_H$ (and $\kappa_E,\kappa_H$) unconjugated. This does
**not** change the Hermiticity conclusion below (2.5) — $M$ is still non-Hermitian for a
lossy sample either way.

**Do not apply Module 3's depolarization factor ($\kappa_E$) here.** That correction exists
specifically to approximate the field distortion a single-mode quasi-static treatment can't
see — which is exactly what letting multiple basis modes mix is meant to capture directly, via
the solved coefficients $A_i$. Applying $\kappa_E$ on top would double-correct, and it isn't
even well-defined for a multi-mode trial expansion (it was derived for one incident mode).

### 2.4 New numerical primitive needed: cross-overlap integral

$\Delta M_{ij}$ for $i\ne j$ is a genuinely new operation — Module 2's
`integrate_field_energy(region, field)` only computes the single-field, same-index integral
$\int_{V_s}|F|^2dV$. This module needs a **cross-overlap** integral between two *different*
fields:
$$\texttt{integrate\_field\_cross\_overlap(region, field\_i, field\_j)} \to \int_{\text{region}} F_i\cdot F_j^*\,dV$$
Implement as a standalone utility (not a `FieldProvider` method — it operates on two
providers, not one), reusing Module 2/3's existing quadrature machinery directly: same
`region.quadrature_points`, same weighted-sum pattern, integrand $F_i(\mathbf r_k)\cdot
F_j(\mathbf r_k)^*$ instead of $|F(\mathbf r_k)|^2$. This is purely additive — no existing
method's signature or behavior changes.

### 2.5 Hermiticity and the correct solver

For a **lossless** sample ($\epsilon_r''=0$), $M$ is Hermitian ($\Delta M_{ji}=\Delta
M_{ij}^*$ follows directly since the scalar $(\epsilon_r-1)$ is then real). For a **lossy**
sample, the complex scalar $(\epsilon_r-1)$ breaks that symmetry — $M$ is not Hermitian in
general. Use `scipy.linalg.eig(K, M)` (the general complex generalized eigenvalue solver),
not `scipy.linalg.eigh` (which assumes Hermitian structure and would silently produce wrong —
purely real — eigenvalues for a lossy sample, discarding exactly the information this whole
project is trying to extract). Reserve `eigh` for the lossless-case cross-check in Section 7.5.

---

## 3. Solving and extracting the answer

### 3.1 The eigenproblem

$$\mathbf K\mathbf A = \tilde\omega^2\mathbf M\mathbf A$$

$N$ eigenpairs $(\tilde\omega_k^2,\mathbf A_k)$, $K$ diagonal from 2.1, $M$ from 2.2–2.3.

### 3.2 Mode tracking

Pick $k^*=\arg\max_k |A_k^{(1)}|$ (the eigenvector whose weight on the *original* mode-of-
interest basis index is largest, after normalizing each eigenvector). For a weak perturbation
this is unambiguous. **Flag, don't silently resolve, near-degeneracy**: if the two largest
candidates' $|A_k^{(1)}|$ are close, the sample is inducing strong mixing with a neighboring
mode, and "the corrected version of mode 1" is itself becoming an ill-posed question — surface
this rather than silently picking one eigenvalue.

**Implementation correction (found building `ritz.py`): "$|A_k^{(1)}|$, after normalizing each
eigenvector" is underspecified, and the literal reading is wrong.** Each basis mode is an
independent `CavityMode` instance with its own arbitrary field-amplitude scale — Module 1's
normalization is only self-consistent *per instance* (CLAUDE.md Conventions), and different
mode *types* at the same nominal `amplitude=1.0` are not guaranteed comparable: e.g. a cube's
$TE_{111}$ and $TM_{111}$ (exactly frequency-degenerate, and the natural test case for this
section) differ in `total_stored_energy()` by five orders of magnitude at `amplitude=1.0`. A
raw eigenvector-component weight, or a weight normalized only by the eigenvector's own
Euclidean norm, inherits that arbitrary per-mode scale and is not the physically meaningful
"how much of mode $k$'s field is the original mode of interest" — confirmed by a deliberately
degenerate/strongly-coupled $TE_{111}$/$TM_{111}$ basis producing weights $[1.0, 2\times
10^{-19}]$ under the naive metric (clearly wrong given $\sim$47% Cauchy-Schwarz coupling)
versus $[0.87, 0.50]$ once corrected. **Fix**: rescale each basis component by its own
$\sqrt{M_{ii}}$ before taking the norm — $A_i\sqrt{M_{ii}}$ is exactly the quantity Section 4's
congruence-transform argument shows is invariant to $E_i\to c_iE_i$, so weight
$=|A_k^{(1)}\sqrt{M_{11}}|\,/\,\lVert A_k\odot\sqrt{\mathrm{diag}(M)}\rVert$. `ritz.py` also
lowers the flagging threshold from an initial guess of 0.7 to 0.5 accordingly (the corrected
$TE_{111}$/$TM_{111}$ case above, $0.50/0.87\approx0.58$, should clearly flag; well-separated
cases tested were $\lesssim10^{-3}$, nowhere near either threshold).

### 3.3 Wall loss — added last, same principle as Module 4 §2.1

The eigenproblem above only knows about lossless PEC-wall basis modes plus the sample; wall
loss is a separate, independent first-order perturbation, added the same way Module 4 combines
it:
$$\tilde\omega_{\text{final}} = \sqrt{\tilde\omega_{k^*}^2} - j\,\frac{\omega_1}{2Q_{\text{wall}}}$$
using $\omega_1$ (mode 1's own base frequency) as the small-perturbation reference, exactly
mirroring Module 4's combination formula. Take the principal (positive-real-part) branch of
the square root.

### 3.4 Extracting $f_{\text{calc}}$, $Q_{\text{calc}}$

Identical formulas to Module 4 §2.2, applied to $\tilde\omega_{\text{final}}$ — reuse that
logic directly rather than re-deriving it.

### 3.5 Basis-size convergence control

Mirrors Module 2's quadrature-doubling logic (§1.5 there), but the "resolution" parameter is
$N$ (basis size) rather than point count: solve at $N$ and $N+k$ (a few more nearby modes),
compare $\tilde\omega_{k^*}$, keep growing until the relative change falls below tolerance.
Cap $N$ at a modest value (cost grows as $O(N^2)$ cross-overlap integrals and $O(N^3)$ for the
eigensolve) — diminishing returns from distant modes make this cap rarely binding in practice.

---

## 4. Scale invariance — derived, not assumed

Each Module 1 basis mode carries its own arbitrary (Module 0) scale. Rescaling $E_i\to c_iE_i$
independently for each $i$ transforms $K\to CKC^\dagger$, $M\to CMC^\dagger$ for
$C=\mathrm{diag}(c_i)$ — a congruence transform that **preserves the generalized eigenvalues**
exactly (standard linear algebra: $K\mathbf A=\tilde\omega^2M\mathbf A \iff
CKC^\dagger\mathbf A'=\tilde\omega^2CMC^\dagger\mathbf A'$ for $\mathbf A=C^\dagger\mathbf A'$,
same $\tilde\omega^2$). So the project-wide scale-invariance convention is automatically
satisfied by this construction — not an extra property to engineer in, a consequence of
building $K,M$ directly from the basis fields with no cross-mode normalization assumed.

---

## 5. Interface-matching audit

| Dependency | Status |
|---|---|
| `RitzCorrectedModel.evaluate(sample) -> PerturbationResult` | Matches `PerturbationModel.evaluate` exactly — `Measurement.model` (Module 5) accepts either for `InverseSolver.fit()`'s residual/optimization step with **zero changes** to `inverse.py`. |
| `RitzCorrectedModel.field_provider`, `.Rs_walls` | **Also required**, found by checking the actual `inverse.py`: Module 5's closed-form seed (`point_dipole_filling_factors`, `_delta_from_measurement`) calls `model.field_provider` and `model.Rs_walls` directly, bypassing `evaluate()` entirely — matching `evaluate()`'s signature alone is not sufficient for a `RitzCorrectedModel` measurement to seed correctly; it needs these two accessors too, pointing at the mode-of-interest's own `FieldProvider` and wall-loss setting. |
| Module 1's `CavityMode.f0`, `.total_stored_energy()`, `.E()`, `.H()` | Already exposed; reused directly, no changes to `cavity.py`. |
| `epsilon_bg`, `mu_bg` | Already added (Module 4 correction, confirmed present in the current `cavity.py`/`fields.py`); reused here, no new gap. |
| Cross-overlap integral (§2.4) | **Genuine new capability**, not currently in Module 2. Additive utility function — no existing method's contract changes. |
| Module 3's `Sample`, `Material`, `SampleRegion` | Reused for region bounds and $\epsilon_r$; **`depolarization_factor` deliberately not called** (§2.3) — worth flagging explicitly so a future implementer doesn't wire it in by habit. |
| Original `RitzField(FieldProvider)` stub | **Still present in `fields.py` as of the current codebase** — §0.3's retirement is a pending action, not yet done. Remove it as part of implementing this plan, rather than leaving two competing "the Ritz thing" symbols in the codebase. |
| Module 5 (`InverseSolver`, `Measurement`) | Untouched, given the `field_provider`/`Rs_walls` accessors above are present on `RitzCorrectedModel`. |
| Module 4 (`PerturbationModel`) | Untouched — `RitzCorrectedModel` is a sibling, not a modification. |

The net integration footprint is smaller than the original stub implied: one new sibling
class (matching two accessors and one method on `PerturbationModel`, not just one method),
one new additive utility function, and one stub to actually remove — nothing existing changes.

---

## 6. Step-by-step implementation instructions

1. Implement the cross-overlap integral utility (§2.4) as a standalone function, reusing
   Module 2's quadrature/convergence machinery structurally. Unit test it against
   `integrate_field_energy` in the degenerate case `field_i is field_j` (should agree exactly).
2. Implement basis selection (§1) — nearest-frequency mode lookup for a given canonical
   cavity type and mode-of-interest.
3. Implement $K$ assembly (§2.1) — purely from existing Module 1 quantities, no quadrature.
4. Implement $M$ assembly (§2.2–2.3), **including the conjugate on $(\epsilon_r-1)$** — this
   is the single easiest line to get wrong by analogy with a "standard FEM mass matrix";
   verify against Section 7.6 immediately after implementing, not just at the end.
5. Implement the eigensolve (§2.5/3.1) using `scipy.linalg.eig`, plus the `eigh` lossless
   cross-check as a separate, test-only code path (Section 7.5).
6. Implement mode tracking with the near-degeneracy flag (§3.2).
7. Implement wall-loss combination and $f_{\text{calc}}/Q_{\text{calc}}$ extraction (§3.3–3.4),
   reusing Module 4's extraction logic rather than re-implementing it.
8. Implement the basis-size convergence loop (§3.5).
9. Add `field_provider` and `Rs_walls` public read-only properties to `RitzCorrectedModel`
   (§5), pointing at the mode-of-interest's own `FieldProvider` and wall-loss setting — needed
   for Module 5's closed-form seed to work with a Ritz-backed `Measurement`, not just its fit.
10. Remove the `RitzField(FieldProvider)` stub from `fields.py` (§0.3/§5 — it is still present
    in the current codebase; this plan replaces it, it doesn't build it).
11. Run the full Section 7 test plan.

---

## 7. Testing plan against the analytic (Module 4) model

Order matters: confirm Ritz is internally trustworthy (7.1) before trusting any comparison
against Module 4 (7.2–7.4) — otherwise a mismatch could be Ritz non-convergence, not a genuine
small-sample-approximation error.

1. **Basis-size self-convergence** (§3.5): confirm $\tilde\omega_{k^*}$ stabilizes as $N$
   grows, for a fixed sample, before comparing to anything else.
2. **Small-sample agreement**: for a sample deep in the classical regime (e.g.
   $V_s/V_{\text{cavity}}<0.1\%$), confirm `RitzCorrectedModel` and `PerturbationModel` agree
   to within a small tolerance — the primary sanity check that basis selection, matrix
   assembly, the eigensolve, and mode tracking are all correct together, since they should
   reduce to the already-validated Module 4 answer here. **Correction (found building
   `ritz.py`)**: this only holds for a `'generic'`-shaped region ($\kappa_E=1$, Module 3's
   point-dipole fallback — see Section 2.3's deliberate non-use of $\kappa_E$). For a `Sphere`
   ($\kappa_E\ne1$), agreement does *not* tighten as the sample shrinks, because $\kappa_E$ is
   a material property independent of sample size (Section 2.3's un-conjugated-N=1 point
   already shows why: bare Ritz always reduces to the $\kappa=1$ point-dipole formula, not the
   $\kappa_E$-corrected one, regardless of $V_s$) — use a `'generic'` region for this
   particular test, not a `Sphere`.
3. **Divergence-with-size sweep**: sweep sample size upward (same shape/material), track the
   relative difference between Module 4's and `RitzCorrectedModel`'s predictions as a function
   of $V_s/V_{\text{cavity}}$. **This sweep is the original project's "sample-size correction"
   study** — the point where this curve crosses 1% is exactly the threshold that study was
   meant to find; this module is what makes it computable.
4. **$N=1$ reduction and the depolarization-factor connection**: with basis size forced to
   $N=1$ (no mixing), `RitzCorrectedModel` should **not** exactly match `PerturbationModel` for
   a shape with $\kappa_E\ne1$ (e.g. a sphere) — Module 4 applies the depolarization
   correction, bare $N=1$ Ritz doesn't (verified exactly: $N=1$ Ritz's $\Delta$ matches the
   *uncorrected* ($\kappa=1$) point-dipole formula to numerical precision). **The doc's original
   claim that "the gap between them shrinks as $N$ grows" is not confirmed** — tested for a
   `Sphere` sample against a nearest-frequency nested nine-mode rectangular basis, swept up to
   $N=40$: the gap stays flat to within a factor of ~2 of its $N=1$ value (drifting slightly
   *away* from `PerturbationModel`, not toward it), not shrinking. This is plausible rather than
   alarming — a nearest-*frequency* rectangular-mode Galerkin truncation has no particular
   obligation to converge toward a *different* approximate model's classical-electrostatics
   answer for a sphere, and the residual gap stays small (same order as $\kappa_E$'s own
   departure from 1) throughout. Test only what's actually verified: nonzero at $N=1$, bounded
   (same order of magnitude, not blowing up) at larger $N$ — not a monotonic-shrinkage claim.
5. **Lossless Hermitian cross-check**: for a lossless trial sample, confirm `scipy.linalg.eig`
   and `scipy.linalg.eigh` agree (both real, both equal) — validates the general solver isn't
   introducing spurious asymmetry, independent of the physics-level checks above.
6. **Passivity / $Q$-degradation**: reuse Module 4 §5's check (`is_passive` ⇒ $Q_{\text{calc}}
   \le Q_{\text{wall}}$) as a shared test utility against `RitzCorrectedModel` too, rather than
   duplicating it. This is not a routine reuse — §2.3 derived that *without* the conjugate on
   $(\epsilon_r-1)$, this construction fails this exact check in the $N=1$ limit, the same way
   Module 4's pre-fix formula did. Run this test immediately after implementing §2.2–2.3
   (step 4 of Section 6), not as a final check — it's the fastest way to catch a conjugate
   regression before it propagates into mode-tracking or convergence work built on top of it.
7. **Mode-tracking robustness**: construct a deliberately near-degenerate basis (two modes with
   close frequencies) and confirm §3.2's ambiguity flag actually fires rather than silently
   selecting an eigenvalue.
8. **Scale invariance**: independently rescale each basis mode's arbitrary amplitude before
   assembly (§4) and confirm $\tilde\omega_{k^*}$ is unchanged.
