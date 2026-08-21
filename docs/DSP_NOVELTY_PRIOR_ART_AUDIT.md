# DSP novelty and prior-art audit: contaminated-reference safety decisions

Status: pre-submission adversarial audit completed 21 August 2026. This document reviews the
contribution that survived the BSS Reviewer #2 pass. It does not reopen Audit-A4, introduce a new
method, or claim an exhaustive systematic review.

## Executive verdict

The broad methodological novelty claim does **not** survive. Desired-signal leakage in a reference,
the attenuation--distortion trade-off, component-wise error decomposition, downstream task
evaluation, risk--coverage analysis for abstention, blind/untouched tests, and prospective decision
rules all have substantial prior art. Several sources combine two or three of these elements.

The closest single precedent found is Ivry, Cohen, and Berdugo (Interspeech 2022): it argues that a
single aggregate residual-echo metric conflates desired-speech damage with residual nuisance and
therefore reports separate desired-speech-maintenance and echo-suppression quantities across their
trade-off. BSS Eval, P.835 speech-enhancement evaluation, distortion-weighted beamforming, and
orthogonal-projection analysis of enhancement errors establish the same underlying principle in
other forms. Selective-prediction literature makes risk--coverage reporting clearly prior art.

A narrower claim remains a **POSSIBLE DIFFERENTIATOR**: in the sources reviewed, we did not identify
a single prior framework that operationally joins all of the following for normal-only
contaminated-reference ASD: a locked downstream comparator, known nuisance and fault-proxy
components, separate attenuation and retention, selector false-safe risk with coverage, frozen
stop rules, an untouched holdout, and one joint safe-use decision. This is an ASD-specific synthesis
and implementation claim, not invention of its ingredients or a universally validated framework.
Independent expert review may still conclude that the synthesis is careful experimental practice
rather than sufficient methodological novelty for *Digital Signal Processing*.

## Audit question and source policy

The attack asks whether prior work already supplies an essentially equivalent *decision
framework*, not whether it uses CARE-ASD terminology. Searches covered BSS/ICA, convolutive BSS,
adaptive noise cancellation (ANC), contaminated references and crosstalk, target-distortion-aware
beamforming, speech enhancement, acoustic echo cancellation (AEC), downstream-task evaluation,
machine condition monitoring, ASD, and selective prediction. Publisher pages, DOI records,
official proceedings, standards/challenge pages, and official DCASE records were preferred.

The search was broad but not a PRISMA-style exhaustive review. Accordingly, the permitted novelty
language is: **“In the sources reviewed, we did not identify prior work that jointly ...”** The
manuscript must not say “no prior work” or “the first” without a separately documented exhaustive
review.

## Comparison matrix

Notation: **Y** = explicit; **P** = partial, indirect, or task-specific analogue; **N** = not part of
the reported framework; **U** = unresolved from the verified public record. “Frozen/holdout” asks
for a prospective rule and an untouched test, not merely a conventional train/test split.

| Prior work / domain | Reference contamination or leakage | Known nuisance and desired components | Attenuation and desired retention separated | Downstream utility separate | Selector risk and coverage | Frozen rule and untouched holdout | Joint safe-use decision | Novelty relation |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Widrow et al. (1975), ANC principles | P | P | P | N | N | N | N | **CLEARLY PRIOR ART** for ANC assumptions and cancellation utility |
| Al-Kindi & Dunlop (1989), reference leakage | Y | P | Y | N | P | N | P | **CLOSE PRIOR ART** for contaminated-reference processing and low-distortion control |
| Griffiths & Jim (1982), constrained beamforming | P | P | Y | N | N | N | P | **CLEARLY PRIOR ART** for preserving a desired response while suppressing interference |
| Doclo & Moonen (2005); Doclo et al. (2007), speech-distortion-weighted MWF | Y | P | Y | N | N | N | P | **CLOSE PRIOR ART** for leakage-aware attenuation--distortion trade-offs |
| Vincent, Gribonval & Févotte (2006), BSS Eval | N | Y | Y | N | N | N | N | **CLEARLY PRIOR ART** for non-substitutable target/interference/artifact components |
| Hu & Loizou (2007), P.835 evaluation | N | Y | Y | N | N | P | N | **CLEARLY PRIOR ART** for separate signal and background distortion plus overall quality |
| Ivry, Cohen & Berdugo (2022), stereo residual-echo metrics | Y | Y | Y | P | N | P | P | **CLOSE PRIOR ART**; strongest direct attack on component-safety metric novelty |
| Iwamoto et al. (2022), enhancement-error decomposition and ASR | N | Y | Y | Y | N | P | N | **CLOSE PRIOR ART** for component effects plus downstream utility |
| Maciejewski et al. (2021), separation evaluation by speaker verification | N | P | P | Y | N | P | N | **PARTIAL OVERLAP** for intrinsic fidelity versus extrinsic utility |
| ICASSP 2022 AEC Challenge | Y | P | Y | Y | N | Y | P | **CLOSE PRIOR ART** for multi-endpoint, blind downstream evaluation |
| El-Yaniv & Wiener (2010); SelectiveNet (2019) | N | Y | N | Y | Y | Y | P | **CLEARLY PRIOR ART** for abstention risk--coverage methodology |
| Xiao & Doclo (2024), spatially selective ANC | Y | Y | Y | N | N | P | P | **CLOSE PRIOR ART** for target preservation, noise reduction, and control trade-offs |
| Lu et al. (2017), denoising plus machinery diagnosis | N | P | N | Y | N | P | N | **PARTIAL OVERLAP** for downstream diagnostic utility after denoising |
| DCASE 2026 dual-channel ASD systems | Y | N | N | Y | N | P | N | **PARTIAL OVERLAP**; demonstrates far-channel usefulness, not certified removability |
| CARE-ASD declared audit | Y | Y | Y | Y | Y | Y | Y | **POSSIBLE DIFFERENTIATOR** only as the joint ASD-specific decision contract |

The matrix is intentionally unfavorable to CARE-ASD. A **P** may still be enough for a reviewer to
call an element routine. “Joint safe-use decision” is not credited merely because a paper chooses
the best average score; it requires an explicit acceptance decision in which nuisance efficacy and
desired-component safety cannot compensate for one another.

## Closest prior frameworks

### 1. Residual-echo suppression during double-talk

[Ivry, Cohen, and Berdugo (2022)](https://doi.org/10.21437/Interspeech.2022-673)
show that stereo SDR mixes desired-speech distortion with residual echo, introduce separate stereo
desired-speech-maintained and residual-echo-suppression metrics, and study their design trade-off
using real and simulated recordings. This directly defeats any claim that CARE-ASD invented the
principle “measure nuisance attenuation and desired-component retention separately.” Its gap
relative to CARE-ASD is narrow: it does not formulate fault-proxy preservation for normal-only ASD,
evaluate false-safe selector risk/coverage, or apply prospective stop rules to a joint safe-use
certificate.

### 2. Component-wise source-separation and enhancement evaluation

[Vincent, Gribonval, and Févotte (2006)](https://doi.org/10.1109/TSA.2005.858005)
separate target distortion, interference, noise, and artifacts in BSS performance measures.
[Hu and Loizou (2007)](https://doi.org/10.1016/j.specom.2006.12.006) use the P.835
signal-distortion, background-intrusiveness, and overall-quality dimensions across thirteen speech
enhancers. [Iwamoto et al. (2022)](https://doi.org/10.21437/Interspeech.2022-318) decompose
enhancement error into noise and artifact components and separately measure downstream ASR impact.
Together these sources substantially pre-empt the claim that aggregate task or signal metrics are
adequate, or that CARE-ASD first separates fidelity from task utility.

### 3. Contaminated references and target-distortion control

[Al-Kindi and Dunlop (1989)](https://doi.org/10.1016/0165-1684(89)90005-4) directly treat desired
signal leakage into a noise-reference channel and seek noise cancellation with low signal
distortion. Linearly constrained and distortionless beamforming has long made desired-response
preservation an explicit constraint
([Griffiths and Jim, 1982](https://doi.org/10.1109/TAP.1982.1142739)). Speech-distortion-weighted
multichannel Wiener filtering makes noise reduction and speech distortion an explicit trade-off
([Doclo and Moonen, 2005](https://doi.org/10.1109/LSP.2005.859530);
[Doclo et al., 2007](https://doi.org/10.1016/j.specom.2007.02.001)). CARE-ASD therefore cannot
claim that contaminated references or desired-signal preservation are new problems.

### 4. Downstream utility and blind evaluation

[Maciejewski, Watanabe, and Khudanpur (2021)](https://doi.org/10.21437/Interspeech.2021-1924)
explicitly contrast intrinsic waveform-fidelity metrics with extrinsic speaker-verification utility.
The [ICASSP 2022 AEC Challenge](https://signalprocessingsociety.org/publications-resources/data-challenges/acoustic-echo-cancellation-challenge-icassp-2022)
combines talk-condition-specific quality with speech-recognition rate and blind evaluation. These
works make “evaluate preprocessing by a downstream task as well as signal quality” prior art.

### 5. Selector safety and coverage

The reject-option/selective-classification literature predates CARE-ASD. El-Yaniv and Wiener define
the risk--coverage trade-off and controlled selective decisions
([JMLR, 2010](https://jmlr.org/papers/v11/el-yaniv10a.html)); SelectiveNet evaluates a learned
selection function under the same trade-off
([Geifman and El-Yaniv, 2019](https://proceedings.mlr.press/v97/geifman19a.html)). CARE-ASD's
false-safe event has domain-specific semantics, but the idea that low risk must be reported jointly
with nontrivial coverage is **CLEARLY PRIOR ART**.

## Domain-by-domain rejection attack

### BSS and ICA

Classical ICA/BSS literature already explains which added assumptions can restore identifiability.
[Comon (1994)](https://doi.org/10.1016/0165-1684(94)90029-9) and
[Hyvärinen and Oja (2000)](https://doi.org/10.1016/S0893-6080(00)00026-5) make the current weak
latent-coordinate lemma elementary rather than a new theory result. Convolutive BSS has an extensive
model-and-separability literature, including
[Nguyen Thi and Jutten (1995)](https://doi.org/10.1016/0165-1684(95)00052-F). Proposition 1 is a
conditional no-free-lunch statement about unrestricted future-fault support, not an ICA/BSS
identifiability advance. A DSP reviewer would be justified in rejecting theory-first positioning.

### ANC, beamforming, and echo cancellation

These fields already treat reference crosstalk, double-talk, desired-signal cancellation, target
distortion, distortionless constraints, and attenuation--distortion trade-offs. CARE-ASD's
terminology does not create novelty. The residual difference is that the desired component of
interest is *unseen fault evidence*, the learner is normal-only, and the acceptance decision is
coupled to a frozen downstream ASD comparison.

### Speech enhancement and multichannel denoising

Component-specific quality and downstream utility are well established. CARE-ASD adds neither
clean/noise decomposition nor task-aware evaluation as a primitive. Its possible addition is to
make those quantities jointly non-compensatory in a prospective safe-use decision and to test an
abstaining selector's false-safe risk.

### Selective prediction and safety-constrained decisions

Risk and coverage are standard. “False-safe” is best described as the task-specific loss attached
to accepted cases, not a new abstention theory. CARE-ASD has no distribution-free risk guarantee,
conformal guarantee, or new selective-learning algorithm and must not imply otherwise.

### Machinery monitoring and ASD

Machinery literature already uses denoising as preprocessing and reports downstream fault
classification. DCASE systems already show that dual-channel information can help ASD. What was not
identified in the reviewed ASD sources is a joint component-provenance and selector-safety decision
under frozen negative-result stop rules. This absence is bounded by the searched sources and cutoff.

## What is not novel

The following must not appear as standalone novelty claims:

- that an auxiliary noise reference may contain desired-signal leakage;
- that noise attenuation and desired-signal distortion should be measured separately;
- that source-separation errors can be decomposed by component;
- that enhancement should also be evaluated on a downstream task;
- that abstention requires a risk--coverage trade-off;
- that blind/held-out evaluation reduces adaptive overfitting;
- that prospective decision rules strengthen negative evidence; or
- that ICA/BSS requires structural assumptions for identifiability.

## What may remain distinctive

The defensible methodological statement is:

> In the sources reviewed, we did not identify a normal-only ASD study that operationally joins a
> locked downstream comparator, separately known nuisance and fault-proxy components, non-
> substitutable attenuation and retention endpoints, selector false-safe risk with coverage,
> prospective stop rules, and an untouched holdout into one contaminated-reference safe-use
> decision.

This should be called an **ASD-specific decision-oriented audit specification** or **operational
synthesis**, not a generally new signal-processing theory. Its usefulness is that it makes the
accept/reject semantics reproducible and prevents a favorable downstream metric, large signal
change, high retention without attenuation, or low false-safe risk at zero coverage from standing
in for the other requirements. The empirical validation remains specific to CARE-ASD; transfer to
AEC, vibration cancellation, or biomedical reference channels is only a proposed use, not evidence.

## Recommended contribution hierarchy

### Contribution 1: frozen empirical component-safety/efficacy boundary

The strongest true contribution is the frozen DCASE-2026-aligned finding: for the locked detector,
machines, interventions/controllers, and controlled component families, broad suppression harmed a
prospective secondary endpoint, the conservative intervention did not establish improvement, and
the selector/controller evidence did not establish a joint useful-attenuation/fault-retention
region. The chronology and frozen stopping make this unusually auditable negative evidence.

### Contribution 2: ASD-specific operational audit synthesis

The second contribution is the decision contract that connects fixed-comparator utility,
component efficacy/retention, false-safe risk/coverage, and prospective decisions. Its individual
elements are prior art; only their integration around the normal-only contaminated-reference ASD
decision is a possible differentiator. A human DSP expert must decide whether that integration is
substantial enough for the journal.

### Contribution 3: scoped formal rationale

Lemma 1 and Proposition 1 should support, not lead, the paper. Lemma 1 is elementary latent-factor
non-uniqueness. Proposition 1 is logically independent of it and is conditional on a future-fault
class that contains an adverse suppressed direction. It clarifies why normal utility evidence is
not a uniform fault-safety certificate, but it should not carry the novelty claim.

## Title, abstract, and contribution attack

The word “Identifiability” in the prior title can reasonably lead an expert reader to expect a
physical source-identifiability theorem. The paper has no such theorem. Retaining the word would
require the subtitle and abstract to work against the title's natural implication and creates an
avoidable editorial-screening risk. The recommended title removes it.

Ranked title candidates:

1. **Auditing Safe Removability from a Contaminated Far-Microphone Reference in Normal-Only Anomalous Sound Detection** — strongest accurate statement.
2. **A Component-Aware Safety Audit of Contaminated-Reference Processing for Normal-Only Anomalous Sound Detection** — safest for DSP editorial screening.
3. **Utility Is Not Safe Removability: A Decision-Oriented Audit for Far-Microphone ASD Preprocessing** — most method-oriented.
4. **A Frozen Safety--Efficacy Boundary for Far-Reference Preprocessing in DCASE-2026-Aligned Anomalous Sound Detection** — most empirical.
5. **When the Noise Reference Contains the Machine: Identifiability and Safety Limits in Normal-Only Anomalous Sound Detection** — current title; technically recoverable only with immediate scope qualifiers, but it over-promises relative to the surviving result.

Required manuscript alignment:

- lead the abstract with the decision problem and audit rather than Lemma 1;
- rank empirical evidence and operational audit ahead of the formal rationale;
- state that Proposition 1 is logically independent of Lemma 1 and is an application-specific
  conditional no-free-lunch result;
- replace “source non-uniqueness” as a keyword with “safe-removal certification”; and
- describe the conclusion as a component-safety/efficacy boundary, not an identifiability result.

## Bibliographic verification register

The following metadata was checked against a publisher, DOI landing page, or official proceedings
record during this audit:

| Source | Verified metadata |
|---|---|
| B. Widrow et al., “Adaptive Noise Cancelling: Principles and Applications” | *Proceedings of the IEEE*, 63(12), 1692–1716, 1975; [DOI 10.1109/PROC.1975.10036](https://doi.org/10.1109/PROC.1975.10036). |
| M. J. Al-Kindi and J. Dunlop, “Improved adaptive noise cancellation in the presence of signal leakage on the noise reference channel” | *Signal Processing*, 17(3), 241–250, 1989; [publisher record and DOI](https://doi.org/10.1016/0165-1684(89)90005-4). |
| L. J. Griffiths and C. W. Jim, “An Alternative Approach to Linearly Constrained Adaptive Beamforming” | *IEEE Transactions on Antennas and Propagation*, 30(1), 27–34, 1982; [DOI 10.1109/TAP.1982.1142739](https://doi.org/10.1109/TAP.1982.1142739). |
| S. Doclo and M. Moonen, “On the Output SNR of the Speech-Distortion Weighted Multichannel Wiener Filter” | *IEEE Signal Processing Letters*, 12(12), 809–811, 2005; [DOI 10.1109/LSP.2005.859530](https://doi.org/10.1109/LSP.2005.859530). |
| S. Doclo et al., “Frequency-domain criterion for the speech distortion weighted multichannel Wiener filter for robust noise reduction” | *Speech Communication*, 49(7–8), 636–656, 2007; [publisher record and DOI](https://doi.org/10.1016/j.specom.2007.02.001). |
| E. Vincent, R. Gribonval, and C. Févotte, “Performance Measurement in Blind Audio Source Separation” | *IEEE Transactions on Audio, Speech, and Language Processing*, 14(4), 1462–1469, 2006; [DOI 10.1109/TSA.2005.858005](https://doi.org/10.1109/TSA.2005.858005). |
| Y. Hu and P. C. Loizou, “Subjective comparison and evaluation of speech enhancement algorithms” | *Speech Communication*, 49(7–8), 588–601, 2007; [publisher record and DOI](https://doi.org/10.1016/j.specom.2006.12.006). |
| M. Maciejewski, S. Watanabe, and S. Khudanpur, “Speaker Verification-Based Evaluation of Single-Channel Speech Separation” | Interspeech 2021, 3520–3524; [official ISCA record](https://doi.org/10.21437/Interspeech.2021-1924). |
| A. Ivry, I. Cohen, and B. Berdugo, “Objective Metrics to Evaluate Residual-Echo Suppression During Double-Talk in the Stereophonic Case” | Interspeech 2022, 5348–5352; [official ISCA record](https://doi.org/10.21437/Interspeech.2022-673). |
| K. Iwamoto et al., “How bad are artifacts?: Analyzing the impact of speech enhancement errors on ASR” | Interspeech 2022, 5418–5422; [official ISCA record](https://doi.org/10.21437/Interspeech.2022-318). |
| R. El-Yaniv and Y. Wiener, “On the Foundations of Noise-free Selective Classification” | *JMLR*, 11(53), 1605–1641, 2010; [official JMLR record](https://jmlr.org/papers/v11/el-yaniv10a.html). |
| Y. Geifman and R. El-Yaniv, “SelectiveNet: A Deep Neural Network with an Integrated Reject Option” | ICML 2019, PMLR 97, 2151–2159; [official PMLR record](https://proceedings.mlr.press/v97/geifman19a.html). |

## Unresolved literature and bibliography issues

1. This is not an exhaustive systematic review; a human BSS/ANC expert may identify a closer
   decision framework, especially in hearing-aid target preservation, robust control, or
   safety-critical condition monitoring.
2. The ICASSP 2022 AEC Challenge paper's final author list, page range, and DOI should be imported
   from the IEEE record during bibliography assembly; the official SPS challenge page currently
   verifies its evaluation design.
3. Xiao and Doclo (ICASSP 2024) and machinery-diagnosis sources should receive a final line-by-line
   bibliography check before citation in the assembled paper; they are used here for positioning,
   not to support a quantitative CARE-ASD result.
4. DCASE 2026 system reports remain challenge technical reports unless an independently verified
   peer-reviewed version exists. Their publication status must be refreshed at submission time.
5. The existing Audit-A1 matrix is a frozen direct-ASD search with a fixed cutoff. This new audit
   supplements it; it must not be described as retroactively preregistered or exhaustive.

## Novelty decision

**Audit-method novelty survived only in narrowed form.** The paper may claim an ASD-specific,
decision-oriented operational synthesis and a frozen empirical boundary. It should not claim a new
general evaluation framework, new identifiability theory, or first recognition of contaminated
reference safety. If an independent expert finds a prior framework with the complete joint decision
contract, contribution 2 should be removed and the paper judged as a rigorous empirical case study.
