# Ray-space Variational Bayes Gaussian Splatting (RVBGS)

Sharpened derivation of the ray-space extension of VBGS
([arXiv:2410.03592](https://arxiv.org/abs/2410.03592)).
This note fixes camera conventions, states the incomplete-observation model
precisely, derives closed-form depth updates under the NIW expected energy, and
proves reduction to VBGS when depth is known.

Reference implementation and numerical checks live in
`src/vbgs/vbgs/ray_space/` and `src/vbgs/tests/test_ray_space_vbgs.py`.

---

## 4.1 Motivation

VBGS assumes Euclidean observations \(\mathcal{D}=\{(\mathbf{x}_n,\mathbf{c}_n)\}_{n=1}^{N}\)
with \(\mathbf{x}_n\in\mathbb{R}^3\) known from RGB-D sensing.

For monocular or calibrated multi-view imagery, each pixel specifies only a
**projection ray**. Write the observation as

\[
\mathbf{r}_n=(u_n,v_n,T_n,K),
\]

where \(K\) is the intrinsic matrix and \(T_n\) is the camera pose. The
corresponding world point is not observed; depth along the ray is latent.

---

## 4.2 Latent ray parameterisation

**Convention (camera-to-world).** Let \(T_n^{\mathrm{c2w}}=[R_n\mid\mathbf{t}_n]\)
map camera coordinates to world coordinates (the convention used in this
codebase). Then the camera centre and (unnormalised) viewing direction are

\[
\mathbf{C}_n=\mathbf{t}_n,
\qquad
\mathbf{d}_n=R_n\,K^{-1}\tilde{\mathbf{u}}_n,
\qquad
\tilde{\mathbf{u}}_n=(u_n,v_n,1)^\top.
\]

**World-to-camera form.** If instead \(T_n^{\mathrm{w2c}}=[R_n\mid\mathbf{t}_n]\)
is used, then \(\mathbf{C}_n=-R_n^\top\mathbf{t}_n\) and
\(\mathbf{d}_n=R_n^\top K^{-1}\tilde{\mathbf{u}}_n\). The draft statement
\(\mathbf{C}_n=-\mathbf{t}_n\) is correct only when \(R_n=I\); we reject it in
general.

The unknown world point is

\[
\boxed{\mathbf{x}_n(\lambda_n)=\mathbf{C}_n+\lambda_n\mathbf{d}_n,\qquad\lambda_n\in\mathbb{R}.}
\]

In applications one may restrict \(\lambda_n>0\) (in front of the camera). The
algebra below is unchanged under truncation of \(q(\lambda_n)\) to \((0,\infty)\).

---

## 4.3 Incomplete-observation generative model

Treat RGB-D VBGS as the complete-data model and monocular/multi-view sensing as
**missing depth**. Priors on mixture parameters are unchanged:

\[
\pi\sim\mathrm{Dir}(\boldsymbol{\alpha}_0),
\qquad
(\boldsymbol{\mu}_k,\Sigma_k)\sim\mathrm{NIW}(\mathbf{m}_0,\kappa_0,W_0,\nu_0).
\]

For each observation \(n\),

\[
z_n\sim\mathrm{Categorical}(\pi),
\qquad
\lambda_n\sim p(\lambda),
\qquad
\mathbf{x}_n\mid z_n=k,\lambda_n,\Theta
\;\sim\;
\mathcal{N}\bigl(\boldsymbol{\mu}_k,\Sigma_k\bigr)
\;\text{with}\;
\mathbf{x}_n=\mathbf{x}_n(\lambda_n).
\]

Equivalently, the geometric likelihood of a ray under component \(k\) at depth
\(\lambda\) is the Gaussian density evaluated at the ray point:

\[
p(\mathbf{r}_n\mid z_n=k,\lambda_n=\lambda,\Theta)
:=
\mathcal{N}\bigl(\mathbf{C}_n+\lambda\mathbf{d}_n;\boldsymbol{\mu}_k,\Sigma_k\bigr).
\]

**Remark (determinism).** Given \(\lambda_n\), \(\mathbf{x}_n\) is deterministic.
The joint should therefore be written over \((Z,\Lambda,\Theta)\) with likelihood
\(p(\mathbf{r}\mid Z,\Lambda,\Theta)\), not as an independent factor \(p(X)\).
Colour factors (if present) factorise as in VBGS and are omitted below.

Depth prior used for tractable updates:

\[
p(\lambda_n)=\mathcal{N}(\lambda_n;m_\lambda,\tau_\lambda^{-1}),
\]

optionally truncated to \(\lambda_n>0\). The improper flat prior is the limit
\(\tau_\lambda\to 0\).

---

## 4.4 Mean-field variational family

\[
q(Z,\Lambda,\Theta)=q(Z)\,q(\Lambda)\,q(\Theta),
\]

with

\[
q(Z)=\prod_n\prod_k r_{nk}^{z_{nk}},
\qquad
q(\Lambda)=\prod_n q(\lambda_n),
\qquad
q(\Theta)=q(\pi)\prod_k q(\boldsymbol{\mu}_k,\Sigma_k).
\]

As in VBGS, \(q(\pi)\) is Dirichlet and \(q(\boldsymbol{\mu}_k,\Sigma_k)\) is NIW.
The new block is \(q(\Lambda)\).

---

## 4.5 Evidence lower bound

\[
\mathcal{L}(q)
=
\mathbb{E}_q\bigl[\log p(Z,\Lambda,\Theta,\mathbf{r})\bigr]
-
\mathbb{E}_q[\log q].
\]

Expanding the joint,

\[
\boxed{
\begin{aligned}
\mathcal{L}
&=
\mathbb{E}_q[\log p(\Theta)]
+
\mathbb{E}_q[\log p(Z\mid\pi)]
+
\mathbb{E}_q[\log p(\Lambda)]
\\
&\quad+
\mathbb{E}_q[\log p(\mathbf{r}\mid Z,\Lambda,\Theta)]
-
\mathbb{E}_q[\log q].
\end{aligned}
}
\]

Relative to VBGS, the new terms are \(\mathbb{E}[\log p(\Lambda)]\) and the
depth-averaged expected Gaussian log-likelihood. When colours are modelled,
their expected log-likelihoods appear exactly as in VBGS.

---

## 4.6 Expected Gaussian energy along a ray

Under the NIW variational posterior, the expected log-density of a point
\(\mathbf{x}\in\mathbb{R}^D\) under component \(k\) is the standard VBGS energy

\[
\ell_k(\mathbf{x})
=
\mathbf{h}_k^\top\mathbf{x}
-\tfrac12\mathbf{x}^\top\Lambda_k\mathbf{x}
+c_k,
\]

where

\[
\Lambda_k:=\mathbb{E}_{q_k}[\Sigma_k^{-1}],
\qquad
\mathbf{h}_k:=\mathbb{E}_{q_k}[\Sigma_k^{-1}\boldsymbol{\mu}_k],
\]

and \(c_k\) collects terms independent of \(\mathbf{x}\)
(\(\mathbb{E}[\boldsymbol{\mu}^\top\Sigma^{-1}\boldsymbol{\mu}]\),
\(\mathbb{E}[\log|\Sigma^{-1}|]\), and the Gaussian base measure).

Substitute \(\mathbf{x}(\lambda)=\mathbf{C}+\lambda\mathbf{d}\):

\[
\ell_k(\mathbf{C}+\lambda\mathbf{d})
=
-\tfrac12 a_k\,\lambda^2
+
b_k\,\lambda
+
\text{const}_{k,\mathbf{r}},
\]

with

\[
a_k=\mathbf{d}^\top\Lambda_k\mathbf{d},
\qquad
b_k=\mathbf{h}_k^\top\mathbf{d}-\mathbf{C}^\top\Lambda_k\mathbf{d}.
\]

Thus the expected log-likelihood is **quadratic in depth**.

---

## 4.7 Responsibility update

Standard mean-field calculus yields

\[
\boxed{
\log r_{nk}
=
\mathbb{E}_q[\log\pi_k]
+
\mathbb{E}_{q(\lambda_n)}\bigl[\ell_k(\mathbf{x}_n(\lambda_n))\bigr]
+
\text{const}_n.
}
\]

For Gaussian \(q(\lambda_n)=\mathcal{N}(m_n,v_n)\) the expectation is closed form:

\[
\mathbb{E}[\ell_k(\mathbf{x}(\lambda))]
=
\ell_k(\mathbf{C}+m_n\mathbf{d})
-\tfrac12 v_n\,a_k,
\]

i.e. the energy at the mean point minus a ray-aligned variance penalty.

---

## 4.8 Depth update (closed form)

\[
\log q^*(\lambda_n)
=
\log p(\lambda_n)
+
\sum_k r_{nk}\,\ell_k(\mathbf{C}_n+\lambda_n\mathbf{d}_n)
+
\text{const}.
\]

With Gaussian depth prior precision \(\tau_\lambda\) and mean \(m_\lambda\),

\[
\log q^*(\lambda_n)
=
-\tfrac12 A_n\lambda_n^2+B_n\lambda_n+\text{const},
\]

\[
\boxed{
\begin{aligned}
A_n
&=
\tau_\lambda+\sum_k r_{nk}\,a_{nk},
\\
B_n
&=
\tau_\lambda m_\lambda+\sum_k r_{nk}\,b_{nk},
\\
q^*(\lambda_n)
&=
\mathcal{N}\!\left(\lambda_n;\frac{B_n}{A_n},\frac{1}{A_n}\right)
\end{aligned}
}
\]

(or the same Gaussian truncated to \((0,\infty)\)). This is the tractable
one-dimensional update whose existence the draft identified as the main
algorithmic question.

**Interpretation.** Each component contributes a soft Gaussian constraint along
the ray; responsibilities weight those constraints; the depth prior regularises.

---

## 4.9 Gaussian (NIW) parameter updates

Expected sufficient statistics use the first two moments of \(\mathbf{x}_n\)
under \(q(\lambda_n)\):

\[
\boxed{
\mathbb{E}[\mathbf{x}_n]
=
\mathbf{C}_n+\mathbb{E}[\lambda_n]\,\mathbf{d}_n,
\qquad
\mathrm{Cov}(\mathbf{x}_n)
=
\mathrm{Var}(\lambda_n)\,\mathbf{d}_n\mathbf{d}_n^\top.
}
\]

Hence

\[
\mathbb{E}[\mathbf{x}_n\mathbf{x}_n^\top]
=
\mathrm{Cov}(\mathbf{x}_n)
+
\mathbb{E}[\mathbf{x}_n]\mathbb{E}[\mathbf{x}_n]^\top.
\]

With \(N_k=\sum_n r_{nk}\) and
\(\bar{\mathbf{x}}_k=N_k^{-1}\sum_n r_{nk}\mathbb{E}[\mathbf{x}_n]\),

\[
S_k
=
\sum_n r_{nk}
\Bigl(
\mathrm{Cov}(\mathbf{x}_n)
+
(\mathbb{E}[\mathbf{x}_n]-\bar{\mathbf{x}}_k)
(\mathbb{E}[\mathbf{x}_n]-\bar{\mathbf{x}}_k)^\top
\Bigr).
\]

NIW natural-parameter updates are identical to VBGS with these moments in place
of Dirac observations. Every observation contributes an extra rank-one
uncertainty along its viewing ray whenever \(\mathrm{Var}(\lambda_n)>0\).

---

## Proposition 1 (Reduction to VBGS)

**Statement.** Fix observed depths \(d_n\) and set
\(q(\lambda_n)=\delta(\lambda_n-d_n)\) for all \(n\) (equivalently:
\(\mathbb{E}[\lambda_n]=d_n\), \(\mathrm{Var}(\lambda_n)=0\)). Then every
coordinate-ascent update and the geometric part of the ELBO coincide with those
of VBGS on the completed points \(\mathbf{x}_n=\mathbf{C}_n+d_n\mathbf{d}_n\).

**Proof.** Under the Dirac (or zero-variance) depth posterior,

\[
\mathbb{E}[\mathbf{x}_n]=\mathbf{C}_n+d_n\mathbf{d}_n,
\qquad
\mathrm{Cov}(\mathbf{x}_n)=\mathbf{0}.
\]

Responsibility scores reduce to
\(\mathbb{E}_q[\log\pi_k]+\ell_k(\mathbf{x}_n)\), the VBGS scores. NIW
statistics \((N_k,\bar{\mathbf{x}}_k,S_k)\) reduce to the complete-data VBGS
statistics. Depth prior and entropy terms become constants independent of
\((q(Z),q(\Theta))\) and may be dropped from the optimisation over those
blocks. Therefore the fixed points of the remaining updates match VBGS. ∎

**Measure-theoretic note.** Prefer the zero-variance limit over literal Dirac
entropies: \(\mathrm{Var}(\lambda_n)\downarrow 0\) with fixed mean \(d_n\).

---

## Proposition 2 (Closed-form depth posterior)

**Statement.** Under the mean-field family of §4.4, NIW expected energies of
§4.6, and Gaussian (possibly truncated) depth prior of §4.3, the optimal
\(q^*(\lambda_n)\) is Gaussian (resp. truncated Gaussian) with precision \(A_n\)
and mean \(B_n/A_n\) as in §4.8.

**Proof.** The expected log joint as a function of \(\lambda_n\) is a sum of
quadratics (§4.6) plus a quadratic log-prior, hence quadratic. Completing the
square yields the stated Gaussian. Truncation to \((0,\infty)\) preserves the
exponential-family form on that interval. ∎

---

## Discussion

VBGS is the complete-data special case of a ray-space model with latent depths.
The conceptual extension is small; the algorithmic content is the tractable
\(q(\lambda)\) update (Proposition 2) and the ray-aligned second-moment
correction in the NIW M-step.

**Identifiability.** With a single view and a weak depth prior, world scale is
poorly constrained: moving component means toward the camera while shrinking
depths can preserve angular structure. An informative depth prior (e.g. from a
monocular depth network), multi-view observations that pin world Gaussians, or
known depths (Proposition 1) restore a well-posed geometry problem. The
reference tests cover the Dirac reduction, the closed-form depth update, the
ray-aligned scatter correction, and refinement under a noisy depth prior.

Remaining engineering choices—truncation, depth prior, coupling to colour
likelihoods, and integration with truncated E-steps / densification in
ImprovedVBGS—are implementation details on top of this coordinate-ascent
skeleton.
