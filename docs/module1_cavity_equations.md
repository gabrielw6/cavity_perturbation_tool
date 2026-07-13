# Module 1 — Analytical Cavity Library: Equations & Implementation Plan

Scope: `RectangularCavity`, `CylindricalCavity`, `CoaxialCavity`, all implementing the
`CavityMode` interface from the Modules 1–5 design doc (`E`, `H`, `f0`, `Q_wall`,
`stored_energy_density`, `total_stored_energy`, `contains`, `bounding_box`).

No code — this is the equation set and the order of operations to turn it into code.

---

## 0. Master recipes (shared machinery, implement once, reuse three times)

Every mode in every one of these cavities is either "transverse magnetic to $z$"
($\mathrm{TM}_z$: $H_z=0$) or "transverse electric to $z$" ($\mathrm{TE}_z$: $E_z=0$), built
from a single real scalar mode function $\Phi$ that satisfies the 3-D Helmholtz equation
and separates as (transverse eigenfunction) $\times$ (axial standing-wave envelope). Getting
this recipe right once, and expressing every geometry as "what is $\Phi$ and $k_c$ for this
mode," is what keeps the three concrete classes from becoming three independent, inconsistent
derivations.

### 0.1 Field generation recipe

Let $\Phi(\text{transverse coords}, z)$ satisfy $(\nabla_t^2 + k_c^2)\Phi = 0$ transversally,
with $z$-envelope chosen so that $k^2 = k_c^2 + k_z^2$ and $k = \omega\sqrt{\epsilon\mu}$. Let
$\nabla_t$ be the gradient restricted to the transverse coordinates (Cartesian $x,y$ for the
rectangular cavity; polar $\rho,\phi$ for the circular and coaxial cavities).

**$\mathrm{TM}_z$ family** ($H_z = 0$):
$$E_z = \Phi, \qquad \mathbf{E}_t = \frac{1}{k_c^2}\nabla_t\!\left(\frac{\partial \Phi}{\partial z}\right), \qquad \mathbf{H}_t = -\frac{j\omega\epsilon}{k_c^2}\,\hat z \times \nabla_t \Phi$$

**$\mathrm{TE}_z$ family** ($E_z = 0$):
$$H_z = \Phi, \qquad \mathbf{H}_t = \frac{1}{k_c^2}\nabla_t\!\left(\frac{\partial \Phi}{\partial z}\right), \qquad \mathbf{E}_t = \frac{j\omega\mu}{k_c^2}\,\hat z \times \nabla_t \Phi$$

These signs (opposite to an earlier draft of this section) are the ones verified against
$\nabla\times E=-j\omega\mu H$ by direct derivation and confirmed by the curl-residual
regression test (`test_curl_residual`, `test_cyl_curl_residual`, `test_numerics.py`'s
`test_tez_tmz_recipe_curl_residual_{tm,te}`) — trust this derivation and the passing tests
over any other transcription of this recipe.

This is exact — it falls directly out of Maxwell's equations for any source-free,
homogeneous region and any coordinate system, so it applies unchanged to the rectangular and
cylindrical geometries below; only $\Phi$ and $k_c$ change. (The coaxial cavity's dominant
family, Section 3, is TEM rather than TM/TE-to-$z$, and uses a separate — simpler —
transmission-line recipe.)

**Implementation implication**: build a small internal helper that, given a callable $\Phi(\cdot)$
and its analytic transverse gradient, produces the six field components mechanically via
these formulas. Each concrete cavity then only needs to supply $\Phi$, $\nabla_t\Phi$, and
$k_c$ — not re-derive six components by hand. This is the single most error-prone part of
the module if done ad hoc per geometry; doing it once here removes that risk.

### 0.2 Stored-energy master formula

At resonance, time-averaged electric and magnetic energy are equal ($W_e = W_m$), so total
stored energy is:
$$W = 2W_e = \frac{\epsilon}{2}\int_V |E|^2\,dV$$

Always compute $W$ via the $E$-field integral (never $H$) — it's one integral instead of
needing both, and it's consistent across all three geometries.

### 0.3 Wall-loss Q master formula

$$Q_{\text{wall}} = \frac{\omega_0 W}{P_{\text{loss}}}, \qquad P_{\text{loss}} = \frac{R_s}{2}\oint_S |H_{\text{tan}}|^2\, dS$$

This is geometry-independent: `Q_wall(Rs)` in every concrete class is "compute $W$ (0.2),
compute $P_{\text{loss}}$ by surface-integrating $|H_{\text{tan}}|^2$ over every conducting
wall face, divide." The only geometry-specific work is enumerating the wall faces and their
$H_{\text{tan}}$.

### 0.4 Reusable integral identities

These are the only building blocks needed to get closed-form $W$ and $P_{\text{loss}}$ for
every mode — derive them once, apply them repeatedly:

**Trigonometric** (for any integer $k \ge 0$, length $L$):
$$\int_0^L \cos^2\!\left(\frac{k\pi u}{L}\right) du = \frac{L}{2}\left(1+\delta_{k0}\right), \qquad \int_0^L \sin^2\!\left(\frac{k\pi u}{L}\right) du = \frac{L}{2}\left(1-\delta_{k0}\right)$$
where $\delta_{k0}=1$ if $k=0$ else $0$. (When $k=0$, $\sin\equiv 0$, so that integral and
every field term that carries it vanish identically — no special-casing needed beyond
having $\delta_{k0}$ multiply the right term.)

**Bessel, TM family** (zero of $J_n$ at $x=X_{np}$, i.e. $J_n(X_{np})=0$):
$$\int_0^a \rho\, J_n\!\left(\frac{X_{np}\rho}{a}\right)^2 d\rho = \frac{a^2}{2}\Big[J_{n+1}(X_{np})\Big]^2$$
(standard Bessel normalization integral; follows from the Lommel integral and the recurrence
$J_n'(x) = -J_{n+1}(x)$ evaluated at a zero of $J_n$.)

**Bessel, TE family** (zero of $J_n'$ at $x=X'_{np}$, i.e. $J_n'(X'_{np})=0$):
$$\int_0^a \rho\, J_n\!\left(\frac{X'_{np}\rho}{a}\right)^2 d\rho = \frac{a^2}{2}\left(1-\frac{n^2}{X_{np}'^2}\right)\Big[J_n(X'_{np})\Big]^2$$

Implement these four as small standalone functions in a shared numerics helper (not
duplicated per cavity class) — Section 2 and the coaxial cavity's radial integral
($\int_a^b d\rho/\rho = \ln(b/a)$, elementary) both consume them.

---

## 1. Rectangular Cavity

### 1.1 Geometry & mode indexing

Box of dimensions $a \times b \times c$ along $x,y,z$ respectively, one corner at the origin.
Modes are $\mathrm{TE}_{mnp}$ ($m,n = 0,1,2,\dots$, not both zero; $p=1,2,3,\dots$) or
$\mathrm{TM}_{mnp}$ ($m,n=1,2,3,\dots$; $p=0,1,2,\dots$).

### 1.2 Mode eigenfunctions

$$\Phi_{\mathrm{TE}} = \cos\!\left(\frac{m\pi x}{a}\right)\cos\!\left(\frac{n\pi y}{b}\right)\sin\!\left(\frac{p\pi z}{c}\right), \qquad \Phi_{\mathrm{TM}} = \sin\!\left(\frac{m\pi x}{a}\right)\sin\!\left(\frac{n\pi y}{b}\right)\cos\!\left(\frac{p\pi z}{c}\right)$$
$$k_c^2 = \left(\frac{m\pi}{a}\right)^2 + \left(\frac{n\pi}{b}\right)^2$$

### 1.3 Resonant frequency

$$f_{mnp} = \frac{1}{2\sqrt{\epsilon\mu}}\sqrt{\left(\frac{m}{a}\right)^2 + \left(\frac{n}{b}\right)^2 + \left(\frac{p}{c}\right)^2}$$

For $a<b<c$ the dominant mode is $\mathrm{TE}_{011}$ — implement this as the default when the
caller doesn't specify indices, since it's what every worked example and unit test will use.

### 1.4 Full field components

Apply the recipe (0.1) to each $\Phi$ above. Result ($H_0$, $E_0$ arbitrary real amplitudes —
see Module 0's scale-invariance convention):

**$\mathrm{TE}_{mnp}$** ($E_z = 0$):
$$H_z = H_0\cos\!\frac{m\pi x}{a}\cos\!\frac{n\pi y}{b}\sin\!\frac{p\pi z}{c}$$
$$H_x = -\frac{H_0}{k_c^2}\frac{m\pi}{a}\frac{p\pi}{c}\sin\!\frac{m\pi x}{a}\cos\!\frac{n\pi y}{b}\cos\!\frac{p\pi z}{c}, \quad H_y = -\frac{H_0}{k_c^2}\frac{n\pi}{b}\frac{p\pi}{c}\cos\!\frac{m\pi x}{a}\sin\!\frac{n\pi y}{b}\cos\!\frac{p\pi z}{c}$$
$$E_x = \frac{j\omega\mu H_0}{k_c^2}\frac{n\pi}{b}\cos\!\frac{m\pi x}{a}\sin\!\frac{n\pi y}{b}\sin\!\frac{p\pi z}{c}, \quad E_y = -\frac{j\omega\mu H_0}{k_c^2}\frac{m\pi}{a}\sin\!\frac{m\pi x}{a}\cos\!\frac{n\pi y}{b}\sin\!\frac{p\pi z}{c}$$

**$\mathrm{TM}_{mnp}$** ($H_z = 0$):
$$E_z = E_0\sin\!\frac{m\pi x}{a}\sin\!\frac{n\pi y}{b}\cos\!\frac{p\pi z}{c}$$
$$E_x = -\frac{E_0}{k_c^2}\frac{m\pi}{a}\frac{p\pi}{c}\cos\!\frac{m\pi x}{a}\sin\!\frac{n\pi y}{b}\sin\!\frac{p\pi z}{c}, \quad E_y = -\frac{E_0}{k_c^2}\frac{n\pi}{b}\frac{p\pi}{c}\sin\!\frac{m\pi x}{a}\cos\!\frac{n\pi y}{b}\sin\!\frac{p\pi z}{c}$$
$$H_x = \frac{j\omega\epsilon E_0}{k_c^2}\frac{n\pi}{b}\sin\!\frac{m\pi x}{a}\cos\!\frac{n\pi y}{b}\cos\!\frac{p\pi z}{c}, \quad H_y = -\frac{j\omega\epsilon E_0}{k_c^2}\frac{m\pi}{a}\cos\!\frac{m\pi x}{a}\sin\!\frac{n\pi y}{b}\cos\!\frac{p\pi z}{c}$$

($H_x,H_y$ corrected the same way as the cylindrical $\mathrm{TM}_{npq}$ block above — an
earlier draft had these two components swapped in sign, the same recipe-transcription error,
just missed on the first pass through this section; verified symbolically against the
corrected §0.1 recipe, not just re-asserted.)

Sanity check built into the design: for $\mathrm{TE}_{011}$ ($m=0$), $E_y$ and $H_x$ collapse
to zero identically (both have a bare $\sin(m\pi x/a)=\sin 0$ or a prefactor $m\pi/a=0$) —
only $E_x, H_y, H_z$ survive, matching the textbook dominant-mode field pattern used as the
Section 1.9 validation case.

### 1.5 Stored energy (closed form)

Apply identity (0.4, trig) to each nonzero component of $|E|^2$ and sum, using
$\delta_{m0},\delta_{n0},\delta_{p0}$ as needed (at most one of $m,n$ is ever zero for TE; $p$
can be zero only for TM). This is mechanical: five terms of the pattern
"$(\text{prefactor})^2 \times \frac{a}{2}(1\pm\delta_{m0}) \times \frac{b}{2}(1\pm\delta_{n0})
\times \frac{c}{2}(1\pm\delta_{p0})$", summed, times $\epsilon/2$. Don't hand-derive a single
final closed-form expression per mode family — implement the sum-over-components-with-identities
directly, so it's automatically correct for every $(m,n,p)$ rather than only the case worked
out by hand.

### 1.6 Wall-loss integral & $Q$

Six wall faces ($x=0,a$; $y=0,b$; $z=0,c$). On each, evaluate $H_{\text{tan}}$ (the two $H$
components tangential to that face — e.g. on the $x=0/a$ faces, tangential components are
$H_y, H_z$) and integrate $|H_{\text{tan}}|^2$ over that face using the same trig identities
from (0.4), then sum the six face contributions and multiply by $R_s/2$ per (0.3).

### 1.7 `contains` / `bounding_box`

Trivial: `bounding_box` = $([0,0,0],[a,b,c])$; `contains(r)` = elementwise
$0\le x\le a \wedge 0\le y\le b \wedge 0\le z\le c$.

### 1.8 Step-by-step implementation instructions

1. Implement the shared numerics helper from Section 0.4 (trig identities) as pure functions
   of $(k, L)$ — test them standalone against `scipy.integrate.quad` before touching any
   cavity code.
2. Implement the Section 0.1 field-generation recipe as an internal helper taking $\Phi$,
   $\nabla_t\Phi$, $k_c$, family (`'TE'`/`'TM'`), and returning a callable `(E(r), H(r))` pair.
   Test it on a trivial made-up $\Phi$ (e.g. a single cosine) by checking $\nabla\times E =
   -j\omega\mu H$ numerically (finite-difference curl) before trusting it on real modes — this
   catches sign errors that are easy to make in the recipe implementation itself and would
   otherwise silently propagate into every geometry.
3. Implement `RectangularCavity.__init__(a, b, c, mode)`, storing $(m,n,p)$ and computing
   $k_c$, $f_0$ directly from 1.2/1.3.
4. Wire $\Phi_{\mathrm{TE}}$/$\Phi_{\mathrm{TM}}$ and their analytic $\nabla_t\Phi$ into the
   Section 0.1 helper to get `E(r)`, `H(r)`. Vectorize over `r` of shape `(N,3)` — every term
   is elementwise in $x,y,z$, so this is direct NumPy broadcasting, no loops.
5. Implement `total_stored_energy` per 1.5 as a direct sum of closed-form terms (not
   quadrature) — it should return a value in well under a millisecond.
6. Implement `Q_wall(Rs)` per 1.6: six closed-form face integrals, sum, master formula (0.3).
7. Run the Section 1.9 validation suite. Do not proceed to Module 2 until every check there
   passes to within floating-point tolerance (not "close enough" — these are closed-form
   comparisons, so mismatches indicate a real bug, most likely a sign or index-order error in
   step 2 or 4).

### 1.9 Validation targets

- **Resonant frequency, dominant mode**: $a=b/2=c/2$ cubic-ish case reduces to
  $f_r = \frac{1}{b\sqrt{2\epsilon\mu}}$ for the square-base ($b=c$) special case — check
  against 1.3 directly.
- **Mode-ratio table**: for a fixed $a{:}b{:}c$ ratio (e.g. $1{:}1{:}1$, $1{:}2{:}1$), compute
  $f_{mnp}/f_{011}$ for several low-order modes and confirm they match known textbook ratios
  (square-base cavity: second resonance at $\sqrt{2}\times$ the first — a clean, checkable
  number).
- **Cubic copper cavity Q**: for a cubic cavity ($b=c$), confirm
  $Q_c \propto \frac{1}{\sqrt{f}}$ and lands in the expected several-thousand range at
  microwave frequencies for copper's surface resistance — an order-of-magnitude sanity check
  before trusting exact digits.
- **Scale invariance** (Module 0 convention): multiply $E_0$/$H_0$ by an arbitrary complex
  constant; confirm `f0`, `Q_wall` are unchanged.
- **Curl residual test** (from step 2 above), retained as a permanent regression test, not
  just a one-off check.

---

## 2. Cylindrical (Circular) Cavity

### 2.1 Geometry & mode indexing

Radius $a$, length $d$, axis along $z$, one end cap at $z=0$, the other at $z=d$. Modes are
$\mathrm{TM}_{npq}$ ($n=0,1,2,\dots$; $p=1,2,3,\dots$; $q=0,1,2,\dots$) or $\mathrm{TE}_{npq}$
($n=0,1,2,\dots$; $p=1,2,3,\dots$; $q=1,2,3,\dots$). $n$ = azimuthal index, $p$ = radial
(Bessel zero) index, $q$ = axial index.

### 2.2 Bessel eigenfunctions & zero tables

$X_{np}$ = $p$-th zero of $J_n$; $X'_{np}$ = $p$-th zero of $J_n'$. Don't hard-code a table —
compute these at construction time via `scipy.special.jn_zeros(n, p)` (TM) and
`scipy.special.jnp_zeros(n, p)` (TE), cached per `(n,p)` since they're expensive relative to
everything else in this module and are re-used every time the mode's fields are evaluated.

### 2.3 Resonant frequency

$$k_c = \frac{X_{np}}{a}\ (\mathrm{TM}) \quad\text{or}\quad \frac{X'_{np}}{a}\ (\mathrm{TE}), \qquad f_{npq} = \frac{1}{2\pi\sqrt{\epsilon\mu}}\sqrt{k_c^2 + \left(\frac{q\pi}{d}\right)^2}$$

$\mathrm{TM}_{010}$ ($n=0,p=1,q=0$) is dominant for $d/a \lesssim 2$; implement this as the
default.

### 2.4 Full field components

Eigenfunctions:
$$\Phi_{\mathrm{TM}} = J_n\!\left(\frac{X_{np}\rho}{a}\right)\cos(n\phi)\cos\!\left(\frac{q\pi z}{d}\right), \qquad \Phi_{\mathrm{TE}} = J_n\!\left(\frac{X'_{np}\rho}{a}\right)\cos(n\phi)\sin\!\left(\frac{q\pi z}{d}\right)$$

(The $\sin(n\phi)$ branch is the degenerate rotated partner — see Section 4 for whether/when
to implement it.)

Applying the Section 0.1 recipe in cylindrical coordinates ($\nabla_t = \hat\rho\,\partial_\rho
+ \hat\phi\,\frac{1}{\rho}\partial_\phi$, $\hat z\times\hat\rho=\hat\phi$,
$\hat z\times\hat\phi=-\hat\rho$):

**$\mathrm{TM}_{npq}$** ($H_z=0$):
$$E_z = J_n\!\left(\frac{X_{np}\rho}{a}\right)\cos(n\phi)\cos\!\left(\frac{q\pi z}{d}\right)$$
$$E_\rho = -\frac{1}{k_c^2}\frac{q\pi}{d}\frac{X_{np}}{a}J_n'\!\left(\frac{X_{np}\rho}{a}\right)\cos(n\phi)\sin\!\left(\frac{q\pi z}{d}\right), \quad E_\phi = \frac{1}{k_c^2}\frac{q\pi}{d}\frac{n}{\rho}J_n\!\left(\frac{X_{np}\rho}{a}\right)\sin(n\phi)\sin\!\left(\frac{q\pi z}{d}\right)$$
$$H_\rho = -\frac{j\omega\epsilon}{k_c^2}\frac{n}{\rho}J_n\!\left(\frac{X_{np}\rho}{a}\right)\sin(n\phi)\cos\!\left(\frac{q\pi z}{d}\right), \quad H_\phi = -\frac{j\omega\epsilon}{k_c^2}\frac{X_{np}}{a}J_n'\!\left(\frac{X_{np}\rho}{a}\right)\cos(n\phi)\cos\!\left(\frac{q\pi z}{d}\right)$$

($H_\rho, H_\phi$ corrected to match §0.1's fixed $\mathrm{TM}_z$ sign — an earlier draft of
this section had these with the opposite, un-fixed sign.)

**$\mathrm{TE}_{npq}$** ($E_z=0$): identical structure with $J_n \to J_n$ (same function, zeros
$X'_{np}$ instead), $\cos(q\pi z/d)\leftrightarrow\sin(q\pi z/d)$ roles swapped, $\omega\epsilon
\to -\omega\mu$, per the general recipe — mechanically substitute into 0.1's TE-family
formula rather than re-deriving by hand.

Use `scipy.special.jvp(n, x, 1)` for $J_n'$ directly rather than the recurrence
$J_n'=J_{n-1}-\frac{n}{x}J_n$ — same numerical result, less code, one fewer place to introduce
a sign slip.

### 2.5 Stored energy (closed form)

$$W_e = \frac{\epsilon}{4}\int_V|E|^2\,dV$$
splits into a $\phi$-integral (elementary: $\int_0^{2\pi}\cos^2(n\phi)\,d\phi = \pi(1+\delta_{n0})$),
a $z$-integral (trig identity, 0.4), and a $\rho$-integral that is exactly the Bessel
normalization identity from 0.4 (TM or TE branch as appropriate) — for the $E_z$ term directly,
and for the $E_\rho,E_\phi$ terms via the same identity applied to $J_n'$ (which itself
satisfies a directly analogous normalization integral obtainable by integrating by parts, or
more simply: evaluate this one numerically per-mode at construction time via `scipy.integrate.quad`
over $\rho\in[0,a]$, since it is a 1-D integral of a fully known closed-form integrand and
doesn't need to be symbolic — the earlier concern about "no numerical integration inside
Module 1" was about avoiding 3-D volume quadrature; a cached 1-D radial quadrature for the
$J_n'$ normalization term is a reasonable, cheap exception, and is far less error-prone than
hand-deriving a second Bessel identity for the derivative term).

### 2.6 Wall-loss integral & $Q$

Three wall surfaces: the curved wall ($\rho=a$, $0\le z\le d$, $0\le\phi<2\pi$) and the two end
caps ($z=0$ and $z=d$, $0\le\rho\le a$). On the curved wall, tangential $H$ is $(H_\phi, H_z)$;
on the end caps, tangential $H$ is $(H_\rho, H_\phi)$. Same pattern as 1.6: evaluate, integrate
using the $\phi$/$z$/$\rho$ identities above (reusing the same cached radial quadrature where a
$J_n'$-normalization term appears), sum, apply the master formula (0.3).

### 2.7 `contains` / `bounding_box`

`bounding_box` = axis-aligned box just enclosing the cylinder:
$([-a,-a,0],[a,a,d])$. `contains(r)`: convert to $(\rho,\phi,z)$, test
$\rho\le a \wedge 0\le z\le d$.

### 2.8 Step-by-step implementation instructions, including axis handling

1. Implement Bessel zero lookup with caching (Section 2.2) and unit-test it directly against
   Table 5-2/5-3-style known values (e.g. $X_{01}=2.405$, $X'_{11}=1.841$) before anything else
   — every downstream formula depends on these being right.
2. Implement Cartesian-to-cylindrical coordinate conversion as a shared utility (used by `E`,
   `H`, `contains`), vectorized over `(N,3)` input.
3. **Handle $\rho=0$ explicitly.** Terms of the form $n/\rho \times J_n(k_c\rho)$ are $0/0$
   exactly on-axis for $n\ge 1$ (since $J_n(x)\sim x^n$ near $x=0$, so $J_n(k_c\rho)/\rho \to 0$
   for $n>1$ and $\to$ a finite nonzero limit for $n=1$). Don't rely on floating-point luck at
   $\rho=$ exactly 0 (quadrature points can legitimately land there, e.g. a region centered on
   the axis) — special-case $\rho < \epsilon_{\text{tol}}$ (e.g. $10^{-9}\times a$) and return
   the analytic small-argument limit ($n=0$: finite; $n=1$: finite nonzero, evaluate via the
   series $J_1(x)/x \to 1/2$ as $x\to0$; $n\ge2$: zero) rather than evaluating the raw
   expression and hoping.
4. Implement $\Phi_{\mathrm{TM}}$/$\Phi_{\mathrm{TE}}$ and wire through the Section 0.1 recipe
   exactly as for the rectangular cavity — same helper function, different $\Phi$.
5. Re-run the curl-residual test (Section 1.8, step 2) on a cylindrical mode specifically —
   the recipe helper is shared, but cylindrical differential operators ($1/\rho$ factors) are
   a distinct enough case that it's worth confirming the shared helper's coordinate-system
   handling is actually generic and not implicitly Cartesian.
6. Implement `total_stored_energy`: closed form for the $\phi$ and $z$ integrals, cached 1-D
   quadrature for the radial integral (Section 2.5).
7. Implement `Q_wall` per 2.6.
8. Run the Section 2.9 validation suite.

### 2.9 Validation targets

- **Bessel zeros**: $X_{01}=2.405$, $X_{11}=3.832$, $X'_{11}=1.841$ (used for $\mathrm{TE}_{111}$,
  the dominant mode when $d/a\gtrsim 2$) — check against `scipy.special.jn_zeros`/`jnp_zeros`
  output directly.
- **$\mathrm{TM}_{010}$ resonant frequency**: $f_r = \dfrac{X_{01}}{2\pi a\sqrt{\epsilon\mu}} =
  \dfrac{2.405}{2\pi a\sqrt{\epsilon\mu}}$, independent of $d$ (no $z$-variation for $q=0$) —
  a clean, checkable closed form.
- **Mode crossover**: confirm the implementation predicts $\mathrm{TM}_{010}$ dominant for
  $d/a<2$ and $\mathrm{TE}_{111}$ dominant for $d/a>2$, by comparing $f_{010}$ and $f_{111}$
  across a swept $d/a$ and finding the crossover near $d/a=2$.
- **Scale invariance and curl-residual** tests, same as Section 1.9.

---

## 3. Coaxial Cavity (TEM family)

### 3.1 Geometry & scope decision

Inner conductor radius $a$, outer conductor radius $b$, length $L$, shorted at both ends
($z=0,L$). **Scope decision**: implement only the TEM mode family (the standing-wave
transmission-line resonances). Coaxial lines also support higher-order hybrid $\mathrm{TE}/\mathrm{TM}$
modes (Bessel/Neumann-function eigenvalue problems in the annulus), but these have cutoff
frequencies well above the TEM family and are never the intended operating mode for a
perturbation-measurement fixture — exactly analogous to why ordinary coax cable is operated
below its first higher-order-mode cutoff. Document this explicitly in the class docstring
rather than silently omitting it, since it's a deliberate scope boundary, not an oversight.

### 3.2 TEM standing-wave fields

Standard transmission-line telegrapher's-equation standing wave with both ends shorted
($V(0)=V(L)=0$):
$$V(z) = V_m\sin\!\left(\frac{q\pi z}{L}\right), \qquad I(z) = j\,\frac{V_m}{Z_0}\cos\!\left(\frac{q\pi z}{L}\right), \qquad q=1,2,3,\dots$$
where $Z_0 = \dfrac{\eta}{2\pi}\ln(b/a)$, $\eta=\sqrt{\mu/\epsilon}$. Converting
voltage/current to fields via the standard coaxial TEM relations:
$$E_\rho(\rho,z) = \frac{V(z)}{\rho\ln(b/a)}, \qquad H_\phi(\rho,z) = \frac{I(z)}{2\pi\rho}$$
No $\phi$ or higher-order $\rho$ dependence — the transverse pattern is the static coaxial
field, modulated by the standing-wave envelope in $z$. This is not an instance of the Section
0.1 recipe (TEM has no cutoff, $k_c=0$, the recipe is singular there) — treat it as its own
simple, closed-form case.

### 3.3 Resonant frequency

$$f_q = \frac{q}{2L\sqrt{\epsilon\mu}}, \qquad q=1,2,3,\dots$$
Half-wave spacing, exactly analogous to the rectangular/circular cavities' axial mode spacing —
a good cross-check that all three geometries are dimensionally consistent with each other.

### 3.4 Stored energy

Using the master formula (0.2) directly:
$$W = \frac{\epsilon}{2}\int_a^b\!\int_0^{2\pi}\!\int_0^L |E_\rho|^2\,\rho\,d\rho\,d\phi\,dz = \frac{\epsilon}{2}\cdot\frac{2\pi}{\ln(b/a)}\cdot V_m^2\int_0^L \sin^2\!\left(\frac{q\pi z}{L}\right)dz$$

**Corrected from an earlier draft**, which stated this proportional to $\ln(b/a)$: since
$E_\rho=V(z)/(\rho\ln(b/a))$, the radial integral is
$\int_a^b\rho\cdot\frac{1}{\rho^2\ln^2(b/a)}\,d\rho=\frac{1}{\ln^2(b/a)}\int_a^b\frac{d\rho}{\rho}=\frac{\ln(b/a)}{\ln^2(b/a)}=\frac{1}{\ln(b/a)}$
— one power of $\ln(b/a)$ short of what the earlier draft carried through. Physically, this
sign of the dependence is the only one that makes sense: for fixed voltage amplitude $V_m$,
stored energy must **diverge**, not vanish, as the two conductors merge ($b\to a$,
$\ln(b/a)\to0$) — matching the standard coaxial capacitance-per-length result
$C'=2\pi\epsilon/\ln(b/a)$, which diverges the same way. Verified against brute-force
quadrature directly (not just this dimensional argument).

### 3.5 Wall-loss Q

Two conductor walls contribute (inner at $\rho=a$, outer at $\rho=b$) plus the two end caps.
For a low-loss coaxial resonator the standard **attenuation-constant route** is more direct
than a fresh surface integral and is worth using here instead of re-deriving 3.4's approach
from scratch:
$$\alpha_c = \frac{R_s}{2\eta\ln(b/a)}\left(\frac{1}{a}+\frac{1}{b}\right), \qquad Q_{\text{wall}} = \frac{\beta}{2\alpha_c} = \frac{q\pi/L}{2\alpha_c}$$
This is the standard transmission-line-resonator result (loss per unit length converted to a
cavity $Q$ via the mode's phase constant $\beta=q\pi/L$) and avoids a separate end-cap surface
integral, which for TEM fields contributes negligibly relative to the long side walls for any
practically-proportioned resonator — but flag this as an approximation in a code comment
(dropping the end-cap contribution), not a silent omission, in case a future short, fat coax
cavity design makes it non-negligible.

### 3.6 `contains` / `bounding_box`

`bounding_box` = $([-b,-b,0],[b,b,L])$. `contains(r)`: convert to $(\rho,z)$, test
$a\le\rho\le b \wedge 0\le z\le L$.

### 3.7 Step-by-step implementation instructions

1. Implement `CoaxialCavity.__init__(a, b, L, q=1)`, computing $Z_0$, $f_q$ directly from 3.2/3.3.
2. Implement `E(r)`, `H(r)` directly from 3.2 (no recipe helper needed — these are two
   closed-form expressions, vectorized over `r`).
3. Implement `total_stored_energy` directly from 3.4 (closed form, no quadrature).
4. Implement `Q_wall(Rs)` from 3.5, with the end-cap-omission comment.
5. Run the Section 3.8 validation suite.

### 3.8 Validation targets

- **Resonant frequency**: confirm $f_1$ matches $c/(2L)$ for an air-filled cavity to within
  floating-point precision (this is the simplest closed form in the whole module — if this is
  wrong, look for a units bug, e.g. $c$ vs $\omega$ mixups, before anything else).
- **Impedance sanity**: confirm $Z_0=50\,\Omega$ reproduces the standard $b/a\approx 2.3$ ratio
  for an air-filled line (a well-known number, good tripwire for a $\ln$ vs $\log_{10}$ mistake).
- **Scale invariance**: same test as Sections 1.9/2.9, applied to $V_m$.

---

## 4. Cross-cutting implementation notes

- **Bessel library**: `scipy.special` throughout (`jv`, `jvp`, `jn_zeros`, `jnp_zeros`). Don't
  hand-roll series expansions except for the explicit $\rho\to0$ limits in Section 2.8 step 3.
- **Vectorization contract**: every `E(r)`/`H(r)` accepts `r` of shape `(3,)` or `(N,3)` in the
  cavity's local Cartesian frame and internally converts to whatever native coordinates that
  geometry needs (cylindrical for 2 and 3) — conversion utilities should be shared, not
  duplicated between `CylindricalCavity` and `CoaxialCavity`.
- **Degenerate modes** ($\sin n\phi$ vs $\cos n\phi$ for $n>0$; also $\mathrm{TE}_{mnp}$
  vs. the mirror mode with $m,n$ swapped in some rectangular cases): implement only the
  $\cos n\phi$ / as-given-index branch for now. The degenerate partner has identical $f_0$
  and $Q$ and is just a $90^\circ$/$n$ rotation — it adds no information for Module 4's
  perturbation calculation unless a specific experiment deliberately excites it, so don't
  build it until something downstream actually needs it.
- **Testing order matters**: validate the trig/Bessel identities (Section 0.4) and the
  field-generation recipe (Section 0.1) as standalone, geometry-agnostic units *before*
  writing any concrete cavity class. A bug in either one otherwise shows up as three
  simultaneous, confusing failures instead of one clear one.

---

## 5. Build order / milestone checklist

1. Shared numerics: trig identities, Bessel-zero caching, Section 0.1 recipe helper, curl-residual
   test harness. Nothing geometry-specific yet.
2. `RectangularCavity` end to end, including full Section 1.9 validation suite passing.
3. `CylindricalCavity` end to end, including the $\rho=0$ handling and Section 2.9 suite.
4. `CoaxialCavity` end to end (simplest of the three — no recipe helper, no Bessel functions),
   Section 3.8 suite.
5. Only after all three pass independently: confirm each satisfies the `CavityMode` contract
   from the Modules 1–5 doc identically (same method signatures, same units, same scale-invariance
   behavior) — this is what lets Module 2's `AnalyticalField` wrap any of the three unchanged.
