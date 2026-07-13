# Module 5 — Inverse: Nonlinear Least-Squares Fit — Equations & Implementation Plan

Scope: `InverseSolver`, `Measurement`, `FitResult`, per the interface in
`architecture_modules_1-5.md`. No code — equations and build order only.

**Read Section 0 first**, as with every prior module doc. Working through the sketch turned
up one call-site bug (it's written against Module 4's superseded signature), one genuinely
ambiguous unit choice the sketch itself flagged but didn't resolve, an unimplemented but
load-bearing closed-form (the seed), and one addition worth making because it's essentially
free and closes a loop opened in the very first design conversation for this project.

---

## 0. Design corrections up front

### 0.1 `_residuals` must call Module 4's corrected signature

The sketch has `m.model.evaluate(m.region, material)`. Module 4's doc (§0.4) simplified
`PerturbationModel.evaluate` to take a single `Sample`. Fix: construct
`Sample(region=m.region, material=material)` fresh on every residual evaluation (cheap —
`Sample` is a frozen dataclass) and call `m.model.evaluate(that_sample)`. This has no
performance cost: Module 4's internal cache keys off `sample.region`, i.e. `m.region` itself,
which is the same object reused across every call for a given `Measurement` — the caching
benefit described in Module 4's doc is unaffected.

### 0.2 Resolving "relative or absolute" for `sigma_f` / `sigma_invQ`

The sketch's own comment flags this as undecided. It matters because the residual as written,
$(f_{\text{calc}}-f_{\text{meas}})/\sigma_f$, is only dimensionally sensible if $\sigma_f$ is
in Hz — but a single fixed Hz value (e.g. the sketch's default $10^{-4}$) doesn't generalize
across cavities operating anywhere from hundreds of MHz to tens of GHz, and $10^{-4}\,\text{Hz}$
absolute is nonsensically tight for any real measurement. The natural, VNA-realistic quantity
is a **relative** (fractional) uncertainty — e.g. $10^{-4}$ meaning 100 ppm frequency
precision, which *is* a sensible default and is almost certainly what the sketch's default
value was meant to express.

**Fix**: both `sigma_f` and `sigma_invQ` are fractional. The residuals become:
$$r_f = \frac{f_{\text{calc}}-f_{\text{meas}}}{\sigma_f\, f_{\text{meas}}}, \qquad r_{1/Q} = \frac{1/Q_{\text{calc}} - 1/Q_{\text{meas}}}{\sigma_Q\,(1/Q_{\text{meas}})}$$

Both residuals are now dimensionless by construction, and $\sigma_Q$ (fractional uncertainty
in $Q$) is used directly as the fractional uncertainty in $1/Q$ — these are exactly equal
by error propagation on $y=1/x$ ($dy/y=-dx/x$), so no separate conversion formula is needed.
Rename the field `sigma_invQ`→`sigma_Q` to reflect this (it's the fractional precision of the
$Q$ *measurement*, not some separately-derived $1/Q$ quantity). Suggested default:
$\sigma_Q=10^{-2}$ (1%, typical for a 3-dB-bandwidth $Q$ extraction) rather than the sketch's
$10^{-3}$, which reads as optimistic for that method — but leave both user-configurable.

### 0.3 Bounds should be constructor parameters, not hardcoded

The sketch buries `[1.0, 0.0]` / `[inf, inf]` inside a private `_bounds` method. Per
`CLAUDE.md`'s passivity-vs-prior distinction (module3 doc §1.4 vs. this module): $\epsilon'\ge1$
is a *prior* for ordinary dielectrics, not a physical law (some materials, e.g. certain
metamaterials or plasmas, can have $\epsilon'<1$) — a user characterizing an unusual sample
needs to be able to relax it. Expose `eps_bounds`, `mu_bounds` (each a `(lo_re, lo_im, hi_re,
hi_im)`-style tuple, or reuse Module 3's `Material` bounds convention if one exists) as
`__init__` parameters, defaulting to the sketch's values.

### 0.4 The closed-form seed needs an actual formula

The sketch's `_closed_form_seed` is an empty stub with a docstring describing what it should
do. Section 2 below derives it in full.

### 0.5 Covariance/uncertainty — worth adding, essentially free

The original design discussion for this project (before Jacobian machinery was explicitly
deferred) wanted parameter uncertainty and an identifiability diagnostic. Both are available
almost for free here: `scipy.optimize.least_squares` already returns `result.jac` — the
(finite-difference) Jacobian of the *residual* vector at the solution — regardless of whether
an analytic Jacobian was ever built. Section 4 uses this directly. This isn't scope creep: it
consumes an output `scipy` already computes, rather than building any new differentiation
machinery.

### 0.6 Validate measurement count against `fit_mu` at construction

If `fit_mu=True` and fewer than two measurements are supplied, the fit is guaranteed
underdetermined (Section 2.4/2.5) — raise clearly in `InverseSolver.__init__` rather than
let `least_squares` either fail cryptically or silently converge along an unconstrained
direction in parameter space.

---

## 1. Residual formulation

### 1.1 Parameter vector

$p=(\epsilon',\epsilon'')$ if `fit_mu=False`, else $p=(\epsilon',\epsilon'',\mu',\mu'')$ —
the sketch's `_unpack` (`eps = p[0]-1j*p[1]`) is already correctly signed against the
project's $\epsilon=\epsilon'-j\epsilon''$ convention; no change needed there.

### 1.2 Weighted residuals (corrected)

For each `Measurement` $m$, with `sample = Sample(region=m.region, material=Material(eps,mu))`
and `r = m.model.evaluate(sample)` (0.1):
$$r_f^{(m)} = \frac{r.f_{\text{calc}} - m.f_{\text{meas}}}{\sigma_f\, m.f_{\text{meas}}}, \qquad r_{1/Q}^{(m)} = \frac{1/r.Q_{\text{calc}} - 1/m.Q_{\text{meas}}}{\sigma_Q\,(1/m.Q_{\text{meas}})}$$
stacked into one residual vector across all measurements, in the order they appear in
`self._meas` (order doesn't matter mathematically, but keep it stable for reproducible
diagnostics).

### 1.3 $Q_{\text{calc}}=\infty$ handles itself

$1/r.Q_{\text{calc}} = 0.0$ exactly in IEEE arithmetic when `Q_calc` is `float('inf')` — no
special-casing needed in the residual formula itself. (If `m.Q_meas` were ever infinite, that
*would* need a guard — but a literally lossless measurement doesn't occur in practice; treat
this as an input-validation question on `Measurement`, not a residual-formula one.)

### 1.4 Why $1/Q$ rather than $Q$ (recap)

Restated from the architecture doc because Section 0.2's fix depends on it: the loss
contribution enters the forward model linearly through $1/Q$ (and through $\epsilon''$), so
residuals built from $1/Q$ keep the problem closer to linear and keep both the informative
range and the noise floor properly represented across the wide dynamic range $Q$ itself can
span; residuals built from raw $Q$ compress exactly the information the fit needs most in the
high-loss regime.

---

## 2. Closed-form initial guess

### 2.1 Point-dipole filling factor — material-independent piece

In the point-dipole limit ($\kappa_E=\kappa_H=1$, Module 3's `'generic'` fallback), Module 4's
filling factors (§1.3) reduce to quantities that don't depend on the trial material at all:
$$p_E^{(0)} = \frac{\epsilon_{bg}\,I_E}{W}, \qquad p_H^{(0)} = \frac{\mu_{bg}\,I_H}{W}$$
computable directly from `model.field_provider` and `measurement.region`, with no material
guess needed yet — exactly the "bypass `PerturbationModel`'s general path" the sketch's
docstring describes.

### 2.2 Back out $\Delta_{\text{meas}}$ from the raw measurement

Invert Module 4's combination formula (§2.1), $\tilde\omega_{\text{loaded}}=\omega_0\big(1-\tfrac{j}{2Q_{\text{wall}}}+\Delta\big)$, for $\Delta$ given the *measured* complex resonance:
$$\tilde\omega_{\text{meas}} = 2\pi f_{\text{meas}}\left(1-\frac{j}{2Q_{\text{meas}}}\right), \qquad \Delta_{\text{meas}} = \frac{\tilde\omega_{\text{meas}}}{\omega_0} - 1 + \frac{j}{2Q_{\text{wall}}}$$
using $\omega_0=2\pi f_0$ and $Q_{\text{wall}}$ from `model.field_provider` (or $Q_{\text{wall}}\to\infty$, dropping that term, if `Rs_walls is None`).

### 2.3 Single-measurement solve (`fit_mu=False`)

With $\mu_r\equiv1$ fixed, Module 4 §1.4's formula (as actually implemented, with the
conjugate correction — see that doc's §1.4) reduces to one linear complex equation:
$$\Delta_{\text{meas}} = -\frac12\overline{(\epsilon_r-1)}\,p_E^{(0)} \quad\Longrightarrow\quad \epsilon_r^{(0)} = 1 - 2\,\overline{\left(\frac{\Delta_{\text{meas}}}{p_E^{(0)}}\right)}$$
computed from the *first* measurement in `self._meas`. One complex division and a conjugate,
no iteration — directly gives both $\epsilon_r'$ and $\epsilon_r''$ as the seed. (An earlier
draft of this section inverted the *un*-conjugated formula; this seed must invert whatever
`perturbation.py` actually computes, not the formula as first drafted — see Module 4 §1.4 for
why the conjugate is there.)

### 2.4 Two-measurement solve (`fit_mu=True`)

A single measurement cannot separate $\epsilon_r$ and $\mu_r$ (this is exactly the
identifiability point from the original project discussion — one $(f,Q)$ pair is one complex
equation in two complex unknowns). Using the first two measurements, form the linear system
for the conjugated unknowns:
$$\begin{pmatrix}-2\Delta_{\text{meas},1}\\ -2\Delta_{\text{meas},2}\end{pmatrix} = \begin{pmatrix}p_E^{(0)}(1) & p_H^{(0)}(1)\\[2pt] p_E^{(0)}(2) & p_H^{(0)}(2)\end{pmatrix}\begin{pmatrix}\overline{\epsilon_r-1}\\ \overline{\mu_r-1}\end{pmatrix}$$
solve the $2\times2$ complex linear system directly (`numpy.linalg.solve`) for
$(\overline{\epsilon_r-1},\overline{\mu_r-1})$, then conjugate each component back to get
$(\epsilon_r-1,\mu_r-1)$. **This $2\times2$ matrix is a miniature version of the
identifiability check from the original discussion**: if the two measurements have nearly the
same $p_E^{(0)}:p_H^{(0)}$ ratio (same field character at both sample placements/modes), this
matrix is ill-conditioned (or, for exactly-degenerate ratios, exactly singular) and the seed
itself will be poor — a useful early warning, before the optimizer ever runs, that these two
measurements don't actually constrain both unknowns well (see Section 4.2 for the same
diagnostic applied to the full nonlinear fit). A singular system should fall back to the same
neutral prior as the underdetermined case (2.5), not let the exception propagate out of seed
generation.

### 2.5 Fallback when underdetermined

If `fit_mu=True` but only one measurement is available — this shouldn't happen given 0.6's
constructor guard, but keep this as a defensive fallback, not a silent default — use a neutral
prior ($\epsilon_r=2-0j$, $\mu_r=1-0j$) and flag it as a non-data-derived seed in any
diagnostic output, rather than attempting a solve that isn't mathematically well-posed.

---

## 3. Optimizer configuration

### 3.1 Bounds (defaults, per 0.3)

$\epsilon'\ge1,\ \epsilon''\ge0$ (and same for $\mu$ if `fit_mu`), all user-overridable.

### 3.2 Parameter scaling

$\epsilon'$ and $\epsilon''$ often live on very different natural scales (e.g. $\epsilon'\sim
O(1\text{–}100)$, $\epsilon''\sim O(10^{-3}\text{–}10)$ for typical low-loss dielectrics) —
pass `x_scale='jac'` to `least_squares` so TRF rescales each parameter by its own Jacobian
column automatically, rather than assuming the sketch's unscaled default behaves well across
that range.

### 3.3 The solve

$$\hat p = \arg\min_p \sum_i r_i(p)^2 \quad\text{s.t. bounds (3.1)}$$
via `least_squares(self._residuals, p0, bounds=bounds, method='trf', jac='2-point',
x_scale='jac')`, with `p0` from Section 2.

**Clip the seed into the bounds before calling `least_squares`**: `scipy.optimize.least_squares`
requires $lb\le x_0\le ub$ exactly, but a data-derived closed-form seed (or a user-supplied
initial guess) is not guaranteed to satisfy a fitting *prior* like $\epsilon'\ge1$ — clip
`p0` to `(lo, hi)` immediately before the call, rather than let an otherwise-reasonable seed
make the optimizer call infeasible.

---

## 4. Uncertainty and identifiability diagnostics

### 4.1 Covariance

At the solution, `result.jac` is the Jacobian of the *already-weighted* residual vector
(1.2's residuals are pre-divided by $\sigma$, i.e. already in units of "number of standard
deviations"), so the standard weighted-least-squares covariance estimate needs no additional
$\hat\sigma^2$ rescaling:
$$\widehat{\mathrm{Cov}}(\hat p) \approx \big(J^\top J\big)^{-1}, \qquad J=\texttt{result.jac}$$

### 4.2 Condition number as the identifiability diagnostic

$$\kappa(J^\top J) = \frac{\lambda_{\max}}{\lambda_{\min}}$$
(ratio of largest to smallest eigenvalue). This is exactly the quantitative "is this
recoverable?" measure from the original project discussion, now available without ever
building the analytic Jacobian — it's a direct read of what `scipy` already computed.
Report it in `FitResult`; a very large value (suggested flag threshold: $10^6$, tune once real
data is available) means the fit found *a* minimum but the data doesn't tightly constrain all
of $p$ independently — most likely because `fit_mu=True` was requested with measurements that
don't have sufficiently different $p_E{:}p_H$ character (the same condition Section 2.4's seed
matrix would have already hinted at).

### 4.3 Guard against a singular $J^\top J$

Use `numpy.linalg.pinv` rather than `inv` for 4.1 (won't raise on an exactly singular matrix),
but still report the condition number (4.2) as the honest signal — a pseudo-inverse produces
*a* number even when the true covariance is unbounded in some direction, so don't let a
successfully-returned covariance matrix imply the fit was well-constrained; that's what 4.2
is for.

---

## 5. Step-by-step implementation instructions

1. Implement `Measurement` and `FitResult` per the sketch, with: `sigma_invQ`→`sigma_Q`
   rename (0.2), and `FitResult` gaining `covariance: np.ndarray | None` and
   `condition_number: float` fields (0.5, Section 4).
2. Implement `Measurement.__post_init__` validation: `f_meas>0`, `0<Q_meas<inf`; raise on
   violation rather than let bad input reach the optimizer.
3. Implement `InverseSolver.__init__` with configurable bounds (0.3) and the `fit_mu`/
   measurement-count guard (0.6).
4. Implement `_unpack` exactly as sketched (1.1 — already correct).
5. Implement `_residuals` with the corrected `Sample`-construction call site (0.1) and the
   corrected relative-sigma formulas (1.2).
6. Implement the point-dipole filling-factor helper (2.1) — this is a small, standalone
   function of `(model, region)` only, reusable by both the seed and any future diagnostics.
7. Implement `_closed_form_seed` per 2.2–2.5, branching on `fit_mu` and measurement count.
8. Implement `fit()`: call `least_squares` per 3.3, then compute covariance and condition
   number per Section 4, then assemble `FitResult`.
9. Run the Section 6 validation suite.

---

## 6. Validation targets

- **Synthetic-data recovery** (already flagged in `CLAUDE.md` as the whole-pipeline regression
  guard): pick a known $(\epsilon_r,\mu_r)$, run it through `PerturbationModel.evaluate` to
  generate synthetic $(f_{\text{meas}},Q_{\text{meas}})$ for one or more measurements, fit,
  and confirm recovery within a tolerance set by the chosen $\sigma_f,\sigma_Q$.
- **Closed-form seed accuracy**: in the point-dipole limit specifically (small, `'generic'`-
  shaped sample), confirm the Section 2 seed alone — with *zero* `least_squares` iterations —
  already matches the known material to within a few percent; this isolates a seed-formula bug
  from an optimizer-configuration bug if the full fit ever fails to converge.
- **Degenerate multi-mode case**: construct two measurements with deliberately similar
  $p_E{:}p_H$ ratios (e.g. the same mode measured twice) with `fit_mu=True`, and confirm both
  the Section 2.4 seed matrix and the Section 4.2 condition number flag the resulting
  ill-conditioning — this is the direct test of the identifiability diagnostic actually working,
  not just being computed.
- **Bounds enforcement**: seed a fit with a synthetic material *outside* the default bounds
  (e.g. $\epsilon'<1$) and confirm the optimizer stays within bounds rather than the seed
  itself violating them silently.
- **$Q_{\text{calc}}=\infty$ robustness**: confirm a lossless synthetic material doesn't crash
  the residual computation or the covariance step (1.3).
- **Rename check**: confirm nothing in the codebase still refers to `sigma_invQ` after 0.2's
  rename to `sigma_Q` — a straightforward grep, worth doing explicitly since the rename touches
  a public dataclass field.
