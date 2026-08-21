# Formalization of contaminated-reference identifiability in CARE-ASD

Status: manuscript-supporting analysis. This document formalizes the scope of claims already tested
by frozen CARE-ASD evidence. It introduces no new intervention, threshold, training run, or result.

## 1. Notation

Let \(m(t)\) denote machine-origin sound during normal operation and \(e(t)\) denote environmental or
interfering sound. The synchronized near- and far-microphone observations are \(x_n(t)\) and
\(x_f(t)\). Acoustic transfer paths are stable linear time-invariant filters \(h_{ij}\) over the
analyzed interval, `*` denotes convolution, and \(\epsilon_n,\epsilon_f\) collect sensor noise and
model mismatch.

An anomalous machine realization may be written as \(m_a(t)=m(t)+q(t)\), where \(q(t)\) denotes an
unseen fault-relevant increment. This notation is conceptual: the controlled CARE-ASD experiments
use specified fault proxies and do not claim that every physical fault is additive.

For compactness, write

\[
\mathbf{x}(t)=
\begin{bmatrix}x_n(t)\\x_f(t)\end{bmatrix},\qquad
\mathbf{s}(t)=
\begin{bmatrix}m(t)\\e(t)\end{bmatrix},\qquad
\mathbf{x}=\mathbf{H}*\mathbf{s}+\boldsymbol{\epsilon},
\]

with

\[
x_n(t)=h_{nm}*m(t)+h_{ne}*e(t)+\epsilon_n(t),
\]

\[
x_f(t)=h_{fm}*m(t)+h_{fe}*e(t)+\epsilon_f(t).
\]

Thus both microphones may contain both machine-origin and environmental energy. “Far” means a
different acoustic view, not an observed noise label.

## 2. Assumptions and deliberately absent assumptions

The bounded non-uniqueness argument uses the following weak model class:

1. normal-only paired observations are available for fitting or controlling the processor;
2. both latent sources may propagate to both microphones through unknown paths;
3. the processor has no component labels during deployment;
4. no source independence, non-Gaussianity, disjoint support, known geometry, known transfer path,
   reference-only interval, or supervised source model is assumed; and
5. the admissible unseen-fault class is not known to be orthogonal to the reference-correlated
   machine subspace.

The argument does **not** claim non-identifiability after imposing sufficiently strong additional
structure. Independent-component assumptions, identifiable nonstationarity, calibrated geometry,
reference-only segments, more microphones, supervised component labels, or a validated fault
support model can change the problem.

## 3. Safe removal and the two distinct questions

A deterministic processor produces

\[
y(t)=\Phi_{\theta}(x_n,x_f)(t),
\]

where the controller state or parameters θ are estimated without anomaly labels. The unchanged-near
choice is Φ₀. A useful processor must be judged along two axes:

- **environmental efficacy:** environmental energy is attenuated by a stated amount; and
- **machine/fault safety:** normal-machine function and admissible fault-relevant energy are
  retained above stated lower bounds.

In a controlled decomposition, θ is frozen from the complete mixture and then held fixed while the
processor is applied counterfactually to the known component pairs. For energy \(E(\cdot)\), example
endpoints are

\[
A_e=10\log_{10}\frac{E(h_{ne}*e)}
{E\!\left(\Phi_{\theta}(h_{ne}*e,h_{fe}*e)\right)},
\]

and

\[
R_q=\frac{E\!\left(\Phi_{\theta}(h_{nm}*q,h_{fm}*q)\right)}
{E(h_{nm}*q)}.
\]

The exact CARE-ASD gate uses its frozen conventions and thresholds; these equations state the
conceptual separation, not a replacement estimand. A processor is certified “safe and useful” only
relative to a declared component family and only when its prespecified attenuation and retention
requirements hold jointly.

This separates two questions:

1. **Can the far microphone provide useful auxiliary information to ASD?** Potentially yes.
2. **Can reference-correlated energy be identified as safely removable environmental noise from
   normal-only observations?** Not without further assumptions in the model class above.

CARE-ASD addresses primarily the second question. A multi-view detector can benefit from the far
channel without interpreting all shared energy as nuisance to be subtracted.

## 4. Proposition: source-label non-uniqueness under the weak model class

**Proposition (bounded non-uniqueness).** Consider noiseless normal observations
\(\mathbf{x}=\mathbf{H}*\mathbf{s}\) in the weak model class above. If the source and transfer models
are both unknown, then the semantic decomposition of a reference-correlated component into
machine-origin and environmental parts is not unique in general. Consequently, a normal-only rule
cannot uniformly certify that removing that component is fault-safe over all observationally
equivalent models in this class.

**Argument.** Let \(\mathbf{T}\) be any invertible \(2\times2\) matrix of gains, or more generally an
invertible matrix of admissible filters. Define

\[
\mathbf{s}'=\mathbf{T}*\mathbf{s},\qquad
\mathbf{H}'=\mathbf{H}*\mathbf{T}^{-1}.
\]

Then

\[
\mathbf{H}'*\mathbf{s}'
=\mathbf{H}*\mathbf{T}^{-1}*\mathbf{T}*\mathbf{s}
=\mathbf{H}*\mathbf{s}
=\mathbf{x}.
\]

Unless additional constraints select one factorization, the observations do not determine whether
a shared latent direction belongs to the machine source, the environment, or a mixture of both.
This is not only a scale/permutation ambiguity. For example, take

\[
\mathbf{H}=\begin{bmatrix}1&1\\0.5&1.5\end{bmatrix},\qquad
\mathbf{T}=\begin{bmatrix}1&0.25\\0.25&1\end{bmatrix}.
\]

Both matrices are invertible, and

\[
\mathbf{H}'=\mathbf{H}\mathbf{T}^{-1}
=\begin{bmatrix}0.8&0.8\\0.133\overline{3}&1.466\overline{6}\end{bmatrix}.
\]

Every entry of both \(\mathbf{H}\) and \(\mathbf{H}'\) is nonzero, so near and far remain mixtures
of both latent sources. Yet

\[
m'=m+0.25e,\qquad e'=0.25m+e,
\]

changes how the same observed, cross-channel-correlated energy is assigned to the semantic source
labels. A normal-only selector sees exactly the same \(\mathbf{x}\) under the two factorizations and therefore
makes the same removal decision. A component treated as environmental under one admissible model
can include machine-origin energy under another. Because normal observations do not constrain an
unseen fault increment to avoid that direction, the decision cannot be certified as uniformly
fault-safe across both explanations. □

## 5. What the proposition does and does not establish

The proposition establishes **mathematical non-uniqueness within a weak model class**. It does not
establish that every physical two-microphone mixture is non-identifiable, that no algorithm can use
a far microphone, or that cancellation is universally unsafe.

Four levels must remain distinct:

1. **Mathematical non-uniqueness:** more than one admissible source/path factorization yields the
   same normal observations.
2. **Empirical non-identifiability:** the tested observable risk statistics did not reliably track
   the known safe-use labels in controlled experiments.
3. **Downstream degradation or non-improvement:** B01 and B02 outcomes are conditional results for
   the locked detector and training realization.
4. **Failure to certify safe removal:** the frozen controller gates did not establish simultaneous
   fault retention and environmental attenuation over their declared controlled families.

None of levels 2–4 proves the universal statement in level 1 for a richer, structurally constrained
model. Conversely, a favorable ASD score alone would not prove source identifiability or component
safety.

## 6. Connection to the frozen CARE-ASD experiments

| Evidence layer | Question answered | What remains unproven |
|---|---|---|
| B00 | What does the locked near-only comparator achieve? | Whether another training initialization or detector class behaves similarly. |
| B01 | What happens when a broad, reference-correlated residual replaces the near waveform under an otherwise locked backend? | Which physical component caused each score change. |
| B02 | Can a conservative, label-free, capacity-controlled residual branch demonstrate improvement while retaining the near path? | Equivalence, universal safety, or optimality of the gate. |
| Phase 8 | Does stronger residual-induced feature displacement co-vary with anomaly-score change? | Causality and physical source identity. |
| SAFE-REF | Can the tested normal-only selector identify a synthetic safe-use label under the corrected controlled protocol? | Transfer beyond its generated mixture family; pristine preregistration of the corrected protocol. |
| AP-CARE G1 | Do separately known fault and environmental components pass frozen joint retention/attenuation checks on an untouched internal holdout? | All real fault modes, all paths, and every possible controller. |
| V1-V10 | Did sequential representation-preservation hypotheses pass their own frozen gates? | Independent replication or evidence for a new positive method. |

Together, these layers support the bounded statement that the present normal-only observations and
tested controllers did not establish a reliable safe-removal region in the DCASE-2026-aligned
development regime. They do not support a generic claim that far-channel information is useless.

## 7. Assumptions that future work would need to establish

A positive safe-removal claim should state and validate at least one source of additional
identifiability, such as:

- measured or calibrated transfer paths and microphone geometry;
- verified source independence or structured nonstationarity;
- reference-only or machine-only intervals;
- supervised environmental and machine/fault component examples;
- an array with enough spatial constraints for the declared source model;
- a validated prior restricting where unseen fault energy can occur; or
- a conservative abstaining controller whose component-level false-safe risk and useful coverage
  pass untouched prospective gates.

Even under such assumptions, downstream ASD utility and component-level safety remain separate
empirical endpoints.
