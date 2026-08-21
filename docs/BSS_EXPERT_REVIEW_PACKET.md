# BSS/multichannel-DSP expert review packet

Purpose: obtain independent review of the formal scope and signal-processing positioning before
journal assembly. This packet records no approval. Review should be made against the manuscript and
[`IDENTIFIABILITY_FORMALIZATION.md`](IDENTIFIABILITY_FORMALIZATION.md), with Audit-A4 treated as
immutable.

## 1. Scientific question

A synchronized far microphone may help anomalous sound detection (ASD), but machine-origin and
environmental sound can reach both microphones. When training uses only normal operation, what can
justify declaring reference-correlated energy *environment-only and safely removable* when future
fault evidence is unseen?

The paper does not ask whether the far channel can be useful; it asks whether the tested
normal-only evidence supports a safe-removal decision. Its proposed answer combines a conditional
formal warning with a component-aware empirical audit.

## 2. Exact physical observation model

For normal machine signal \(m(t)\), environmental/interfering signal \(e(t)\), near and far
observations \(x_n(t),x_f(t)\), unknown acoustic transfer paths \(h_{ij}\), and model/sensor errors
\(\epsilon_n,\epsilon_f\):

\[
x_n(t)=h_{nm}*m(t)+h_{ne}*e(t)+\epsilon_n(t),
\]

\[
x_f(t)=h_{fm}*m(t)+h_{fe}*e(t)+\epsilon_f(t).
\]

An anomalous realization is represented only conceptually as \(m_a(t)=m(t)+q(t)\), where \(q\) is
an unseen fault-relevant increment. The empirical work uses declared synthetic fault proxies; it
does not assert that all real faults are additive.

The equations are a physical LTI approximation. The lemma below is *not* a theorem for arbitrary
convolutional filter matrices.

## 3. Lemma 1

**Admissible class.** At one instantaneous mixture or local frequency bin, let
\(\mathbf{x}_k=\mathbf{H}\mathbf{s}_k\), with nonsingular
\(\mathbf{H}\in\mathbb{C}^{2\times2}\). The latent processes have finite power. The class imposes no
independence, non-Gaussianity, source-support, geometry, transfer-path, or source-family constraints.
Its latent coordinates do not encode physical machine/environment provenance.

**Lemma 1 (unstructured instantaneous/local-bin factorization).** For any nonsingular constant
\(\mathbf{T}\),

\[
\mathbf{s}'_k=\mathbf{T}\mathbf{s}_k,\qquad
\mathbf{H}'=\mathbf{H}\mathbf{T}^{-1}
\]

is an exact observationally equivalent admissible latent factorization, because
\(\mathbf{H}'\mathbf{s}'_k=\mathbf{H}\mathbf{s}_k\).

**Exact non-conclusion.** The result proves non-unique coordinates only in the unstructured latent
class. It does not prove that \(m'=m+\alpha e\) is physically machine-origin, that two physically
labelled acoustic worlds explain the same data, that BSS is impossible under ICA/spatial/source
assumptions, or that an arbitrary causal convolutive transformation has a stable causal inverse.
The lemma is supporting rationale, not the primary novelty claim.

## 4. Proposition 1

**Available information.** A processor or selector is fitted from the complete normal-observation
law only and declares a nonzero observed direction removable. It has no anomaly examples or
validated restriction on future-fault support.

**Future-fault class assumption.** The declared class contains two extensions of the same normal
physical model: a retention-benign extension whose fault increments avoid the suppressed direction,
and a retention-adverse extension whose admissible machine-fault increment maps into that direction
strongly enough to violate the declared retention rule.

**Safety certificate.** A uniform fault-safety certificate asserts that the declared nontrivial
removal satisfies the retention rule for every extension in this declared class.

**Proposition 1 (bounded decision-theoretic non-certifiability).** Because the extensions have the
same complete normal-observation law, a normal-only rule makes the same decision in both; because
their retention outcomes differ, normal observations alone cannot certify that removal as
fault-safe uniformly over the declared class.

**What is not proved.** The proposition gives no probability of the adverse extension; does not
show that every nontrivial processor harms a real fault; does not prove normal physical source-label
ambiguity; does not prevent an identity/always-abstain rule from preserving fault energy; and is not
a universal BSS or convolutive-separation impossibility theorem. It may properly be viewed as a
conditional, application-specific no-free-lunch result.

## 5. Relationship between Lemma 1 and Proposition 1

Proposition 1 does **not** depend logically on Lemma 1. The proposition still follows if the normal
physical decomposition is uniquely known, provided the future-fault extension class retains both
benign and adverse suppressed-direction extensions. Lemma 1 instead warns that the deliberately
weak normal latent model does not attach physical semantics to its coordinates. The manuscript must
not use one statement to inflate the other.

## 6. Five questions for the expert

1. Is Lemma 1 mathematically correct and scoped tightly enough to an unstructured
   instantaneous/local-frequency-bin latent-factor model?
2. Is Proposition 1 logically valid under its declared information set and future-fault extension
   class?
3. Is Proposition 1 too tautological or no-free-lunch-like to carry theoretical novelty, even if it
   remains useful supporting rationale?
4. Does any wording in the manuscript or formalization still imply physical source-label
   non-identifiability, general convolutive non-identifiability, or general BSS impossibility?
5. Which important BSS, ANC, echo-cancellation, target-preservation, or source-separation
   references/assumptions are missing from the prior-art audit?

Please also comment on whether “safe-removal certification” is an appropriate term and whether the
empirical audit, rather than the formal result, is the scientifically honest primary contribution.

## 7. Reviewer response form

Reviewer name/affiliation (optional): ________________________________________________

Review date: ____________________

Overall formal assessment:

- [ ] Accept as stated
- [ ] Accept with minor correction
- [ ] Major correction required
- [ ] Remove formal claim from main contribution

Lemma 1:

- [ ] Correct at stated scope
- [ ] Wording/scope correction required
- [ ] Remove from main text

Comments:

________________________________________________________________________________

________________________________________________________________________________

Proposition 1:

- [ ] Logically valid at stated scope
- [ ] Valid but too tautological for a novelty claim
- [ ] Assumptions/conclusion require correction
- [ ] Remove from main text

Comments:

________________________________________________________________________________

________________________________________________________________________________

Literature or missing assumptions:

________________________________________________________________________________

________________________________________________________________________________

Residual overclaim locations (quote section/sentence):

________________________________________________________________________________

Recommended disposition:

- [ ] Continue to journal assembly after listed corrections
- [ ] Obtain another specialist review
- [ ] Reframe as primarily empirical/methodological
- [ ] Do not submit in current form

Signature or confirmation (optional): ______________________________________________
