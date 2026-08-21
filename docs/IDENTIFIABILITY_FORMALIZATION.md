# Formalization of contaminated-reference identifiability in CARE-ASD

Status: manuscript-supporting analysis. This document formalizes the scope of claims already tested
by frozen CARE-ASD evidence. It introduces no new intervention, threshold, training run, or result.

## 1. Notation and physical observation model

Let \(m(t)\) denote machine-origin sound during normal operation and \(e(t)\) denote environmental or
interfering sound. The synchronized near- and far-microphone observations are \(x_n(t)\) and
\(x_f(t)\). Over an interval in which a linear time-invariant approximation is adequate,

\[
x_n(t)=h_{nm}*m(t)+h_{ne}*e(t)+\epsilon_n(t),
\]

\[
x_f(t)=h_{fm}*m(t)+h_{fe}*e(t)+\epsilon_f(t),
\]

where the \(h_{ij}\) are acoustic transfer paths, `*` denotes convolution, and the error terms
collect sensor noise and model mismatch. Thus both microphones may contain both source classes.
“Far” identifies an acoustic view, not an observed environmental-source label.

An anomalous machine realization may be written conceptually as \(m_a(t)=m(t)+q(t)\), where \(q(t)\)
is an unseen fault-relevant increment. The controlled CARE-ASD studies use specified fault proxies;
they do not claim that every physical fault is additive or belongs to the proxy family.

The convolutional equations motivate the signal-processing problem. The algebraic result below is
deliberately narrower: it is stated for an instantaneous mixture, or for one local frequency bin in
which convolution is represented as multiplication,

\[
\mathbf{x}_{k}=\mathbf{H}\mathbf{s}_{k},\qquad
\mathbf{s}_{k}=\begin{bmatrix}m_k&e_k\end{bmatrix}^{\mathsf T},
\]

with nonsingular \(\mathbf{H}\in\mathbb{C}^{2\times2}\). We do **not** claim a theorem for arbitrary
causal convolutional filter matrices. Such an extension would have to specify whether
\(\mathbf{T}(z)\) and \(\mathbf{T}^{-1}(z)\) are causal and stable, whether finite-impulse-response
structure is preserved, and whether \(\mathbf{H}(z)\mathbf{T}^{-1}(z)\) remains physically
admissible. None of those properties follows from abstract invertibility alone.

## 2. Admissible model classes and absent assumptions

The following distinctions prevent an algebraic ambiguity from being mistaken for a physical
source theorem.

### 2.1 Unstructured latent-factor class

Let \(\mathcal{M}_{\mathrm{lat}}\) contain pairs \((\mathbf{H},\mathbf{s})\) satisfying the local model
above, with finite-power latent processes and nonsingular mixing, but with no independence,
non-Gaussianity, source-support, geometry, transfer-path, or source-family constraints. An
**admissible latent factorization** is any member of this class that reproduces the observations
exactly. In \(\mathcal{M}_{\mathrm{lat}}\), the words “component 1” and “component 2” are coordinate
labels only; physical machine/environment provenance is not encoded by the model.

### 2.2 Physically labelled class

Let \(\mathcal{M}_{\mathrm{phys}}\) use the convolutional observation equations with source labels
defined by physical provenance. A physically admissible explanation would additionally need to
respect whatever source locations, propagation laws, supports, and source-process families are
asserted. CARE-ASD does not measure enough of those quantities to characterize this class
exhaustively.

Consequently, a non-diagonal change of latent basis is admissible in
\(\mathcal{M}_{\mathrm{lat}}\), but it is **not by itself proof** that both transformed coordinates are
physically admissible machine and environmental sources in \(\mathcal{M}_{\mathrm{phys}}\). In
particular, calling \(m'=m+\alpha e\) “machine-origin” would beg the semantic question unless the
physical model explicitly permits such a source. The manuscript therefore uses the basis argument
only as an algebraic warning about an unstructured latent model, not as a physical relabelling
theorem.

### 2.3 Available information and deliberately absent structure

The CARE-ASD safety question assumes that:

1. only normal paired observations are available when fitting the processor or selector;
2. both physical source classes may propagate to both microphones through unknown paths;
3. component labels and future fault examples are unavailable during deployment;
4. no source independence, non-Gaussianity, disjoint support, known geometry, known transfer path,
   reference-only interval, or supervised source model is guaranteed; and
5. the declared unseen-fault extension class contains both a retention-benign extension and an
   extension whose fault-relevant observation is suppressed strongly enough to violate the declared
   retention rule.

These omissions define the boundary of the result. They are not assertions that stronger structure
never holds. Under its further technical conditions, classical ICA, for example, uses mutual
independence and suitable non-Gaussianity to obtain identifiability up to conventional ambiguities
([Comon, 1994](https://doi.org/10.1016/0165-1684(94)90029-9);
[Hyvärinen and Oja, 2000](https://doi.org/10.1016/S0893-6080(00)00026-5)). Convolutive BSS likewise
requires an explicit mixing model and separability conditions; examples include
[Nguyen Thi and Jutten (1995)](https://doi.org/10.1016/0165-1684(95)00052-F) and
[joint-diagonalization conditions for convolutive mixtures](https://doi.org/10.1016/j.sigpro.2008.02.003).
CARE-ASD makes no claim that BSS is impossible under those assumptions.

## 3. Safe removal and the two distinct questions

A deterministic processor produces

\[
y(t)=\Phi_{\theta}(x_n,x_f)(t),
\]

where the controller state or parameters \(\theta\) are estimated without anomaly labels. The
unchanged-near choice is \(\Phi_0\). A useful processor must be judged along two axes:

- **environmental efficacy:** environmental energy is attenuated by a stated amount; and
- **machine/fault safety:** normal-machine function and admissible fault-relevant energy are
  retained above stated lower bounds.

In a controlled decomposition, \(\theta\) is frozen from the complete mixture and then held fixed
while the processor is applied counterfactually to known component pairs. For energy \(E(\cdot)\),
example endpoints are

\[
A_e=10\log_{10}\frac{E(h_{ne}*e)}
{E\!\left(\Phi_{\theta}(h_{ne}*e,h_{fe}*e)\right)},
\]

and

\[
R_q=\frac{E\!\left(\Phi_{\theta}(h_{nm}*q,h_{fm}*q)\right)}
{E(h_{nm}*q)}.
\]

The exact CARE-ASD gate retains its frozen conventions and thresholds; these equations state the
conceptual separation, not a replacement estimand. “Safe and useful” is always relative to a
declared component family and prespecified joint attenuation and retention requirements.

This separates two questions:

1. **Can the far microphone provide useful auxiliary information to ASD?** Potentially yes.
2. **Does usefulness or cross-channel correlation certify a component as safely removable
   environmental noise?** No. Such a certificate requires additional structural assumptions or
   controlled component evidence.

CARE-ASD primarily addresses the second question. A multi-view detector can benefit from the far
channel without assigning all shared energy to nuisance noise.

## 4. Lemma 1: exact latent-factor non-uniqueness

**Lemma 1 (unstructured instantaneous/local-bin factorization).** In
\(\mathcal{M}_{\mathrm{lat}}\), let \(\mathbf{x}_k=\mathbf{H}\mathbf{s}_k\). For any nonsingular
constant matrix \(\mathbf{T}\),

\[
\mathbf{s}'_k=\mathbf{T}\mathbf{s}_k,\qquad
\mathbf{H}'=\mathbf{H}\mathbf{T}^{-1}
\]

is an observationally equivalent admissible latent factorization. The equivalence is exact for each
sample or time-frequency coefficient, not merely an equality of second-order statistics.

**Proof.** Direct substitution gives

\[
\mathbf{H}'\mathbf{s}'_k
=\mathbf{H}\mathbf{T}^{-1}\mathbf{T}\mathbf{s}_k
=\mathbf{H}\mathbf{s}_k
=\mathbf{x}_k. \quad\square
\]

For example,

\[
\mathbf{H}=\begin{bmatrix}1&1\\0.5&1.5\end{bmatrix},\qquad
\mathbf{T}=\begin{bmatrix}1&0.25\\0.25&1\end{bmatrix}
\]

yields

\[
\mathbf{H}'=\begin{bmatrix}0.8&0.8\\0.133\overline{3}&1.466\overline{6}\end{bmatrix},
\qquad m'=m+0.25e,\quad e'=0.25m+e.
\]

This non-diagonal transformation is more general than scale or permutation. It proves that an
unstructured latent factorization has no unique coordinates. It does **not** prove that \(m'\) and
\(e'\) retain physical machine/environment provenance. That semantic limitation is material, not a
technicality.

## 5. Proposition 1: normal-only fault-safety cannot be certified uniformly

**Proposition 1 (bounded decision-theoretic non-certifiability).** Consider a processor or selector
whose decision is a function only of available normal observations and which declares a nonzero
observed direction removable. Let the declared future-fault extension class contain two models with
the same complete normal observation law: in one, admissible fault-relevant increments satisfy the
retention rule after removal; in the other, an admissible machine-fault increment maps into the
suppressed direction strongly enough to violate that rule. Then normal observations alone cannot
certify the removal as fault-safe uniformly over that class.

**Constructive argument.** Fix the normal sources, physical paths, and normal observation law. The
two following admissible future extensions are therefore exactly observationally equivalent during
normal-only fitting:

- **World A:** future fault-relevant increments have no energy in the direction suppressed by the
  processor. Removing that direction may satisfy the declared retention rule.
- **World B:** a future machine fault excites a component that propagates into that suppressed
  direction strongly enough to violate the same retention rule.

The second extension is allowed precisely because the model has no fault examples or validated
fault-support restriction. The normal-only rule receives identical information and makes the same
decision in both worlds, yet the retention decision differs. It therefore cannot issue a uniform
fault-safety certificate over both extensions. \(\square\)

This construction is physically modest: the normal acoustic world need not be relabelled, and the
only difference is an unseen future machine-fault realization. It proves a conditional limitation
of certification, not that World B must occur, not that every nontrivial processor removes fault
energy, and not that dual-channel denoising is impossible. An identity/always-abstain processor can
be uniformly retention-safe but does not establish useful environmental removal.

The proposition also says nothing about the probability of a future fault direction. A probabilistic
safety statement would require a justified distribution or support model for \(q\), which the frozen
study does not supply.

Proposition 1 does **not** depend logically on Lemma 1. Even if the normal physical decomposition
were known uniquely, the proposition would still follow whenever the declared future-fault
extension class contains both retention-benign and retention-adverse extensions that are
indistinguishable from normal observations. It is therefore an application-specific no-free-lunch
statement about a uniform certificate under an unrestricted fault-support direction, not a new
ICA/BSS identifiability theorem. Lemma 1 supplies only a separate warning about what the weak
unstructured normal model does not identify.

## 6. Formal result, interpretation, and empirical evidence

The manuscript keeps three levels separate:

1. **Formal result.** Lemma 1 gives exact coordinate non-uniqueness in an unstructured
   instantaneous/local-bin latent model. Proposition 1 gives a uniform-certification limitation over
   a declared future-fault class containing retention-benign and adverse extensions along the
   suppressed direction.
2. **Signal-processing interpretation.** Cross-channel correlation and far-channel usefulness do
   not by themselves attach a physical environmental label or establish fault-safe removability.
   Additional source, path, spatial, temporal, or fault-support information could do so.
3. **Empirical evidence.** B01, B02, SAFE-REF, controlled-component experiments, and AP-CARE test
   particular interventions and controlled families under frozen rules. Their failures neither
   prove Lemma 1 nor show that every BSS or dual-channel method fails.

The ordinary ICA ambiguities of scale and permutation are not the paper's claimed contribution.
Lemma 1 demonstrates that a weaker, unstructured factor model admits non-diagonal transformations;
the reviewer objection is correct that physical semantic admissibility does not automatically
follow. Proposition 1 supplies the application-level safety statement without relying on that
semantic leap or on Lemma 1. Neither formal statement is claimed as the paper's primary novelty;
they provide scoped rationale for the decision-oriented empirical audit.

## 7. Connection to frozen CARE-ASD experiments

| Evidence layer | Question answered | What remains unproven |
|---|---|---|
| B00 | What does the locked near-only comparator achieve? | Whether another training initialization or detector class behaves similarly. |
| B01 | What happens when a broad reference-correlated residual replaces the near waveform under an otherwise locked backend? | Which physical component caused each score change; performance of more sophisticated BSS methods. |
| B02 | Can a conservative, label-free, capacity-controlled residual branch demonstrate improvement while retaining the near path? | Equivalence, universal safety, or representativeness of every possible gate. |
| Phase 8 | Does stronger residual-induced feature displacement co-vary with anomaly-score change? | Causality and physical source identity. |
| SAFE-REF | Can the tested normal-only selector identify a synthetic safe-use label under the corrected controlled protocol? | Transfer beyond its generated mixture family; pristine preregistration; all possible selectors. |
| AP-CARE G1 | Do separately known fault-proxy and environmental components pass frozen joint retention/attenuation checks on an untouched internal holdout? | Real fault modes, arbitrary paths, and every possible controller. |
| V1-V10 | Did sequential representation-preservation hypotheses pass their own frozen gates? | Independent replication or evidence for a new positive method. |

Together, these layers support the bounded empirical statement that the present normal-only
observables and tested controllers did not establish a reliable safe-removal region in the
DCASE-2026-aligned development regime. They do not validate the weak model as a complete physical
description and do not imply that far-channel information is useless.

## 8. Assumptions that could change the result

A positive identifiability or safe-removal claim could use, and would need to validate, structure
such as:

- mutual independence and suitable non-Gaussian ICA source models;
- source-specific distributions, sparsity, temporal structure, or identifiable nonstationarity;
- measured geometry, spatial priors, or calibrated transfer paths;
- reference-only or machine-only intervals;
- supervised environmental, machine, and fault examples;
- multiple sufficiently diverse mixtures or additional microphones;
- a validated restriction on the support or distribution of future faults; or
- an abstaining controller whose component-level false-safe risk and useful coverage pass an
  untouched prospective gate.

Even if one of these assumptions restores source identifiability, downstream ASD utility and
component-level safety remain separate empirical endpoints.
