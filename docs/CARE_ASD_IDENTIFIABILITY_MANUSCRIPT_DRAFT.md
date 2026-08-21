# Auditing Safe Removability from a Contaminated Far-Microphone Reference in Normal-Only Anomalous Sound Detection

Status: DSP internal-review draft built on evidence frozen at Audit-A4 commit
`f1d5f7fadea74de6e9c7fdefcb172962b3298b63`. Newly derived DCASE and
real-versus-emulated summaries are explicitly post hoc and consume only frozen development
predictions. This document authorizes no training, tuning, threshold change, evaluation-label
access, or new V-series method.

## Abstract

A synchronized far microphone can provide useful auxiliary information for anomalous sound
detection (ASD), but it is not necessarily a noise-only observation: machine-origin and
environmental signals can propagate to both near and far microphones. We formulate a
decision-oriented audit for this contaminated-reference setting. The audit treats downstream
utility, known-component environmental attenuation, machine/fault retention, selector false-safe
risk and coverage, and frozen holdout decisions as non-substitutable estimands. In a
DCASE-2026-aligned development study conditional on one locked training realization, broad
reference-correlated residual replacement did not improve the registered primary AUC endpoint and
harmed a prospectively specified secondary pAUC endpoint (paired delta -0.01740, 95% interval
[-0.03271, -0.00286]). A conservative near-primary intervention did not demonstrate improvement.
In controlled synthetic-mixture studies, a normal-only selector had a 92.85% false-safe rate on its
post-correction holdout, while AP-CARE retained injected fault-proxy energy but obtained median
environmental attenuation of -0.0397 dB and failed five of six frozen mechanism checks. These
findings establish an empirical safety/efficacy boundary for the tested controllers, component
family, comparator, and development machines. As supporting rationale, an unstructured
instantaneous/local-bin factorization has non-unique latent coordinates, and a normal-only rule
cannot issue a uniform fault-safety certificate when the declared future-fault class contains both
retention-benign and retention-adverse extensions along a suppressed direction. The statements are
not a physical source-label or general convolutive BSS theorem. The findings do not imply that
far-channel information is generally useless or that safe cancellation is universally impossible.

**Keywords:** anomalous sound detection; contaminated reference; dual microphone; safe-removal
certification; fault preservation; safety audit; domain shift; negative results

## 1. Introduction

Normal-only anomalous sound detection (ASD) learns ordinary machine operation and must detect
previously unseen failures. The
[DCASE 2026 Task 2 definition](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
adds synchronized near and far microphones to a
first-shot, source/target domain-shift problem. The far microphone is expected to be relatively
noise-dominant, but it is not guaranteed to contain noise alone. Machine-origin energy can reach
both microphones; therefore a processor that removes content correlated across channels can reduce
environmental interference, machine structure, fault evidence, or some mixture of them.

Two questions must be distinguished:

1. **Can a far microphone provide useful auxiliary information to an ASD system?** Potentially
   yes. Successful dual-channel DCASE systems provide direct evidence that multi-view information
   can be useful.
2. **Can reference-correlated energy be identified as safely removable environmental noise from
   normal-only dual-channel observations?** Not necessarily. That conclusion requires structural
   assumptions or component-level evidence beyond cross-channel correlation.

CARE-ASD primarily addresses the second question. It does not claim that dual-microphone ASD is
ineffective. A learned multi-view system may exploit the far channel without interpreting every
shared component as nuisance to be subtracted.

Desired-signal leakage in an auxiliary reference is established signal-processing prior art,
including adaptive-cancellation analysis by
[Al-Kindi and Dunlop (1989)](https://doi.org/10.1016/0165-1684(89)90005-4). Separate target,
interference, and artifact measures are established in source-separation evaluation
([Vincent et al., 2006](https://doi.org/10.1109/TSA.2005.858005)); speech-enhancement and
residual-echo studies explicitly separate desired-signal damage from nuisance suppression
([Hu and Loizou, 2007](https://doi.org/10.1016/j.specom.2006.12.006);
[Ivry et al., 2022](https://doi.org/10.21437/Interspeech.2022-673)); and downstream task utility and
selective risk--coverage are also prior art
([Iwamoto et al., 2022](https://doi.org/10.21437/Interspeech.2022-318);
[El-Yaniv and Wiener, 2010](https://jmlr.org/papers/v11/el-yaniv10a.html)). CARE-ASD therefore does
not claim to invent these ingredients. Current ASD systems using deterministic reference denoising,
spatial descriptors, embedding residuals, or learned noise-aware representations are also
counterexamples to a generic negative claim about stereo ASD.

The remaining gap is narrower. In the sources reviewed through 21 August 2026, we did not identify
a normal-only ASD study that jointly operationalizes (i) downstream utility under a locked
comparator, (ii) separately known environmental and fault-proxy components, (iii) non-substitutable
attenuation and retention endpoints, (iv) selector false-safe risk with coverage, and (v) a frozen
joint decision on an untouched holdout. This is a bounded source result, not an exhaustive “no
previous work” claim. The individual elements and several partial combinations are prior art, so
the methodological claim is only their ASD-specific decision integration. DCASE technical reports
and relevant preprints are treated as primary system descriptions rather than peer-reviewed
confirmation.

The paper makes three contributions, ordered by the strength of the surviving claim:

1. A frozen empirical component-safety/efficacy boundary in a DCASE-2026-aligned development
   setting: the tested broad and conservative interventions did not establish improved ASD, while
   controlled studies did not establish a reliable region with both useful attenuation and
   fault-proxy preservation.
2. An ASD-specific, decision-oriented contaminated-reference audit specification that jointly separates
   fixed-comparator ASD utility, known-component environmental attenuation and fault retention,
   selector safety, and prospective stop decisions. Its individual ingredients are established
   evaluation practices; the claimed contribution is only their operational integration for this
   ASD safety decision, not invention of each ingredient or validation of a universal framework.
3. A scoped formal rationale that separates algebraic latent-factor non-uniqueness from a bounded,
   decision-theoretic limitation on uniform normal-only fault-safety certification and identifies
   assumptions that can change the problem. Neither statement is claimed as a new general ICA/BSS
   identifiability result.

## 2. Problem formulation

### 2.1 Two-microphone observation model

Let \(m(t)\) denote machine-origin sound under normal operation and \(e(t)\) environmental or
interfering sound. The synchronized observations are

\[
x_n(t)=h_{nm}*m(t)+h_{ne}*e(t)+\epsilon_n(t),
\]

\[
x_f(t)=h_{fm}*m(t)+h_{fe}*e(t)+\epsilon_f(t),
\]

where `*` denotes convolution, \(h_{ij}\) are unknown acoustic transfer paths, and the error terms
collect sensor noise and model mismatch. An anomalous machine realization may be conceptualized as
\(m_a=m+q\), where \(q\) is an unseen fault-relevant increment. The controlled experiments use
specified injected fault proxies and do not assume that every physical fault is additive.

In vector form, \(\mathbf{x}=\mathbf{H}*\mathbf{s}+\boldsymbol{\epsilon}\) with
\(\mathbf{s}=[m,e]^{\mathsf T}\). Both sources may appear in both observations; “far” is a spatial
description, not an environmental-source label.

### 2.2 Algebraic ambiguity and bounded safety non-certifiability

The physical convolutional model above motivates the problem, but the algebraic statement is scoped
to an instantaneous mixture or one local frequency bin,
\(\mathbf{x}_k=\mathbf{H}\mathbf{s}_k\), with nonsingular
\(\mathbf{H}\in\mathbb{C}^{2\times2}\). Its admissible latent-factor class imposes no independence,
non-Gaussianity, support, geometry, path, or source-family constraints. For any nonsingular constant
matrix \(\mathbf{T}\),

\[
\mathbf{s}'_k=\mathbf{T}\mathbf{s}_k,\qquad
\mathbf{H}'=\mathbf{H}\mathbf{T}^{-1}
\]

reproduces every observation exactly. A non-diagonal \(\mathbf{T}\) gives an ambiguity beyond scale
and permutation. However, in this unstructured class the components are latent coordinates, not
physically certified source labels. In particular, the equality
\(m'=m+\alpha e\) does **not** establish that \(m'\) is a physically admissible machine-origin
source. The basis argument therefore proves latent-factor non-uniqueness, not physical source-label
ambiguity. This semantic limit corrects the strongest over-reading of the earlier formulation.

**Proposition 1 (bounded decision-theoretic non-certifiability).** Let a normal-only processor
declare a nonzero observed direction removable. Suppose the declared future-fault extension class
contains two models with the same complete normal observation law: one satisfies the retention rule
after removal, while the other includes a machine-fault increment that maps into the suppressed
direction strongly enough to violate that rule. Normal observations alone cannot certify the
removal as fault-safe uniformly over that class.

To see this, hold the complete normal physical model and observation law fixed. In World A, future
fault increments avoid the suppressed direction; in World B, an unseen machine fault propagates
into that direction strongly enough to violate the same retention rule. The worlds are identical
during normal-only fitting, so the processor makes the same decision, but the safety outcome differs.
This establishes a limitation of a declared uniform certificate. It does not show that World B is
probable, that every processor removes fault energy, or that the normal physical sources themselves
have two valid semantic labels. An identity or always-abstain rule can preserve fault energy but
does not demonstrate useful removal.

Proposition 1 does not depend logically on the latent-factor result above. It would still hold if
the normal physical decomposition were uniquely known, provided the declared future-fault class
retained both benign and adverse suppressed-direction extensions. It is therefore a conditional,
application-specific no-free-lunch statement supporting the audit decision, not the paper's primary
theoretical novelty.

The proposition is not a universal BSS impossibility result. Under further technical conditions,
classical ICA uses mutual independence and suitable non-Gaussianity to obtain identifiability up to
its conventional ambiguities
([Comon, 1994](https://doi.org/10.1016/0165-1684(94)90029-9);
[Hyvärinen and Oja, 2000](https://doi.org/10.1016/S0893-6080(00)00026-5)). Known geometry or paths,
spatial priors, source-specific generative models, sparsity or temporal structure, reference-only
segments, supervised components, sufficiently diverse mixtures, additional microphones, or a
validated fault-support restriction can change the problem. Nor do we extend the proposition to an
arbitrary filter transform \(\mathbf{T}(z)\): invertibility alone does not guarantee a stable causal
inverse or a physically admissible transformed acoustic system. The complete definitions and proof
boundary are given in
[`IDENTIFIABILITY_FORMALIZATION.md`](IDENTIFIABILITY_FORMALIZATION.md).

### 2.3 Safety and efficacy are joint requirements

Let a deterministic processor be \(y=\Phi_{\theta}(x_n,x_f)\), with controller state θ fitted
without anomaly labels. In a controlled decomposition, θ is realized on the complete mixture and
then held fixed while the processor is applied to the known component pairs. Conceptually,
environmental attenuation and fault retention are

\[
A_e=10\log_{10}\frac{E(h_{ne}*e)}
{E\!\left(\Phi_{\theta}(h_{ne}*e,h_{fe}*e)\right)},
\]

\[
R_q=\frac{E\!\left(\Phi_{\theta}(h_{nm}*q,h_{fm}*q)\right)}{E(h_{nm}*q)}.
\]

The exact frozen experiments retain their registered conventions and thresholds. These equations
express the distinction: a processor is “safe and useful” only over a stated component family and
only if prespecified lower bounds on retention and attenuation hold jointly. High retention with
near-zero attenuation is not evidence of effective safe cancellation.

Latent-factor non-uniqueness, the decision-theoretic certification limit, empirical selector
failure, and downstream ASD degradation are separate claims. The experiments below test particular
processors and controlled component families; they neither prove the formal statements nor make
them universal beyond their declared classes.

## 3. Controlled contaminated-reference safety-audit methodology

The audit is decision-oriented. A **processor** \(\Phi_\theta\) maps primary and reference
observations to an output; a **selector** \(D\) accepts or abstains from applying it; and a hidden
controlled **safe-use label** \(S\) records whether the realized case jointly meets prespecified
efficacy and retention requirements. The estimands are downstream utility relative to a controlled
comparator, environmental attenuation, machine/fault retention, and false-safe risk
\(P(S=0\mid D=1)\), together with coverage. The **audit decision** applies a frozen joint rule to
these estimands on declared holdouts. Each element is familiar experimental practice. The
methodological contribution claimed here is limited to an ASD-specific explicit decision contract
that prevents utility, signal modification, abstention, or retention-without-attenuation from being
substituted for one another. Component-separated metrics, downstream evaluation, target-distortion
control, and risk--coverage are clearly prior art; whether their present integration is sufficient
methodological novelty remains subject to independent expert review. Reuse in other
contaminated-reference domains is possible in principle but is not validated by the ASD experiments.

### 3.1 Downstream utility under a controlled comparator

The development files, training restriction, feature/scoring conventions, detector or capacity
constraint, seed, endpoints, bootstrap, and decision rules are frozen before candidate outcomes.
Candidates score the same observations, and effects are paired within prespecified strata. When
only preprocessing changes, the downstream detector remains identical. If an interface change is
necessary, it must be stated and capacity-controlled rather than called an identical comparator.

### 3.2 Controlled known-component decomposition

The processor is evaluated on mixtures whose environmental and machine/fault-proxy components are
available separately. For adaptive or nonlinear processing, the controller is realized on the
complete mixture and then frozen for component-wise counterfactual application. Environmental
attenuation and machine/fault retention are reported separately, including lower-tail retention.
A generic SNR or feature displacement cannot identify which source was changed.

### 3.3 Selector and controller safety

For hidden controlled safe-use label (S) and normal-only accept decision (D), the audit reports
coverage (P(D=1)), false-safe risk (P(S=0\mid D=1)), an uncertainty bound, safe-use prevalence,
risk-score tracking, and useful attenuation. Coverage and safety must be interpreted jointly:
abstaining everywhere is not useful, while high coverage can conceal unsafe acceptance.

### 3.4 Frozen prospective decision rules

The chronology separates hypothesis generation, method search on declared development or
calibration information, a frozen holdout evaluation, and a stop decision. Corrections after an
outcome is inspected retain the prior record and are labeled post-correction. Sequential studies
are research history and narrowing evidence, not independent replications.

The full reusable protocol is documented in
[`CONTAMINATED_REFERENCE_SAFETY_AUDIT_PROTOCOL.md`](CONTAMINATED_REFERENCE_SAFETY_AUDIT_PROTOCOL.md).

## 4. Experimental setup

### 4.1 DCASE-aligned development data

The aligned development analysis contains 1,400 files for each candidate: seven machine types,
one section, source and target domains, normal and anomalous conditions, and 50 clips in each of 28
machine-by-domain-by-condition strata. ToyCar and fan are real synchronized two-channel machine
types; ToyCarEmu, bearingEmu, gearboxEmu, sliderEmu, and valveEmu are emulated two-channel types.
No evaluation-set labels or scores are used.

### 4.2 Locked detector and interventions

**B00** is the locked near-only reference: channel-0 audio; 128-bin power Mel features
(`n_fft=1024`, `hop_length=512`); log-power conversion; five-frame stacking; and the pinned
`640-128-128-128-128-8-128-128-128-128-640` autoencoder. It uses Adam, learning rate 0.001,
batch size 256, 100 epochs, and one locked training realization with seed 13711. The file score is
the mean reconstruction MSE.

**B01** replaces the near waveform with a bounded causal residual obtained from the synchronized
far reference. Feature extraction, detector architecture, training settings, seed, and scoring are
otherwise locked to B00.

**B02** keeps the near view as reconstruction target and primary branch. Near and residual features
are separately projected from 640 to 64 dimensions, and the residual branch is multiplied by a
frozen normal-only path-confidence value. Its first projection is parameter-count matched to B00,
but the two-branch input is an architectural change; B02 is therefore capacity- and
protocol-controlled rather than detector-identical.

### 4.3 Frozen estimands and uncertainty

The registered primary endpoint for B01 and B02 is mean development AUC. Standardized pAUC at
maximum false-positive rate 0.1 is a prospectively specified secondary endpoint. Candidate-minus-
B00 intervals use 5,000 paired bootstrap replicates stratified by machine type, section, and
condition. Within each machine-section stratum, normal and anomaly files are resampled separately
with replacement and each B00/candidate file pair remains intact. The interval therefore requires
exchangeability at the development-file level within those strata. It does not model dependence
among clips from a shared acquisition sequence, nor does it sample new machine types or recording
sessions. Audit-A2 adds descriptive per-machine/domain estimates, a 2,000-replicate sensitivity
bootstrap, and leave-one-machine-out summaries; these do not replace the original inference.

All B00/B01/B02 models use seed 13711. The paired bootstrap quantifies uncertainty over the frozen
evaluation observations and strata conditional on that locked training realization. It does not
quantify optimizer or training-initialization uncertainty. No additional seeds are introduced
after observing the outcomes.

### 4.4 Mechanism and controlled-component studies

Phase 8 freezes the association between B01-induced absolute log-Mel displacement and B01-minus-
B00 anomaly-score change. Displacement is a feature-space quantity, not an energy decomposition,
and the association is non-causal.

Corrected SAFE-REF uses controlled synthetic anomalies mixed with recorded normal carriers and
separate calibration and holdout sets. Its hidden label defines safe use from known simulation
components; the normal-only selector does not receive that label. AP-CARE G1 spans fixed leakage,
environmental gain, path mismatch, fault-proxy amplitude/band, support relation, and seeds. One
realized controller is applied separately to its environmental, normal-machine, and fault-proxy
components. Its 256 calibration and 256 internal holdout cases are disjoint under the frozen
generator.

These controlled mixtures provide component identity within their generators, not universal
physical ground truth for real mechanical faults.

### 4.5 SAFE-REF chronology and stop rules

| Time | Recorded event | Evidential implication |
|---|---|---|
| 2026-08-15 16:57 +07 | Original SAFE-REF pipeline committed (`75a8c243`), including 20% minimum coverage. | Original prospective contract. |
| 2026-08-15 17:25 +07 | First run `...T102535Z` yielded zero accepted cases. | Outcome was observed before the correction. |
| 2026-08-15 21:49 +07 | Commit `d78e0cb` changed profile fitting to independent source pairs, fixed far anomaly propagation, and used a Wilson false-safe upper bound in calibration. | Corrected protocol is post-correction, not pristine preregistration. |
| 2026-08-15 21:51 +07 | Corrected run `...T145112Z` executed with separate calibration and holdout simulations. | Its holdout is untouched within the corrected protocol, but the whole protocol followed the first outcome. |
| 2026-08-16 14:48 +07 | AP-CARE planning (`ead40c9`) documented the safe-prevalence feasibility ceiling. | Post-result diagnosis; it did not change or rescue the SAFE-REF gate. |

The 20% coverage requirement remained unchanged in the corrected gate. After the corrected holdout
showed 8.40% safe prevalence, the maximum theoretical coverage under a 5% false-safe constraint was
calculated as approximately `0.08398/0.95 = 8.84%`. This revealed that the joint targets were
infeasible for that realized controlled distribution, but the criterion was not retroactively
modified. SAFE-REF also failed through false-safe risk, risk tracking, and tail-loss reduction, so
the scientific stop does not depend only on the infeasible coverage target.

AP-CARE was designed after SAFE-REF. Its G1 mechanism gate was frozen before the AP-CARE holdout
run; failure stopped G2-G5, GPU replication, and evaluation access. FP-NAA V1-V10 followed separate,
outcome-informed mechanism revisions. They support research-history transparency and stopping but
are not ten replications and are not pooled with the primary inference.

## 5. Results

### 5.1 B00 locked baseline

B00 obtained mean AUC 0.60813 and mean pAUC 0.55391 on the frozen 1,400-file development set. The
external alignment tolerance was passed before B01/B02 interpretation. These values define the
locked comparator rather than a claim of state-of-the-art performance.

### 5.2 B01 broad reference-correlated suppression

| Endpoint | B00 | B01 | Paired B01-B00 delta | 95% interval | Registered role |
|---|---:|---:|---:|---:|---|
| Mean AUC | 0.60813 | 0.60251 | -0.00566 | [-0.02491, 0.01297] | Primary |
| Mean pAUC | 0.55391 | 0.53459 | -0.01740 | [-0.03271, -0.00286] | Secondary |

B01 did not demonstrate improvement on the registered primary AUC endpoint. Its prospectively
specified secondary pAUC endpoint showed a harmful negative interval. The effect remained negative
after every one-machine deletion for mean pAUC (range -2.49 to -1.11 percentage points), but
machine effects were heterogeneous and this check is not validation on unseen machines.

![Aligned B00/B01/B02 effect sizes](../reports/audit/care_asd_identifiability_audit_v1/performance_deltas.svg)

### 5.3 B02 conservative near-primary intervention

| Endpoint | B00 | B02 | Paired B02-B00 delta | 95% interval | Registered role |
|---|---:|---:|---:|---:|---|
| Mean AUC | 0.60813 | 0.60627 | -0.00192 | [-0.00969, 0.00596] | Primary |
| Mean pAUC | 0.55391 | 0.54789 | -0.00540 | [-0.01398, 0.00323] | Secondary |

B02 had smaller negative point estimates than B01 but did not demonstrate improvement. The
intervals do not establish equivalence, safety, or absence of an effect. Source-domain changes were
more favorable than target-domain changes for both interventions, indicating domain sensitivity
rather than uniform failure.

![Machine-level pAUC effects](../reports/audit/care_asd_robustness_appendix_v1/machine_pauc_forest.svg)

### 5.4 Controlled component evidence and feature displacement

Across 1,400 paired clips, stronger B01-induced absolute log-Mel displacement was associated with
lower B01-minus-B00 anomaly-score change (Spearman ρ = -0.5524). The association differed across
machines: groupwise ρ ranged from -0.5415 to +0.5597. The global result is consistent with removal
of anomaly evidence in this pipeline, but it neither identifies the removed physical source nor
establishes causality.

The controlled studies provide the required decomposition. SAFE-REF evaluates whether a selector
recognizes a known joint safe-use label. AP-CARE separately measures injected fault-proxy retention
and environmental attenuation under a controller fixed on the full mixture. Thus aggregate feature
change is not treated as evidence of safe denoising.

### 5.5 SAFE-REF selector safety

On the corrected 2,048-case holdout, 172 cases (8.40%) met the generator's joint safe-use
definition. The normal-only policy accepted 1,776 cases (86.72% coverage), including 1,649 unsafe
cases. The false-safe rate was 92.85% with Wilson upper bound 93.96%; risk-score Spearman ρ with
known risk was 0.0400, and tail-loss reduction was 1.78%.

These are demonstrated failures of the corrected selector under its controlled mixture family.
They are not evidence that every normal-only selector must fail. Because the protocol was corrected
after the first run outcome, this result is labeled post-correction with an internal untouched
holdout, not as an unmodified original preregistration.

### 5.6 AP-CARE untouched internal holdout

AP-CARE G1 evaluated 256 frozen holdout cases. Leakage and uncertainty statistics had Spearman
ρ = 0.5326 and 0.4060, below their 0.60 thresholds. In-support fault-proxy retention passed its
absolute requirement: median 1.00012 and q05 0.98840 versus registered minima 0.90 and 0.75.
However, median environmental attenuation among 240 eligible cases was -0.03969 dB versus a 1 dB
minimum, and none of the 256 holdout cases reached 1 dB. The registered matched-retention
improvement was 0.05258 versus 0.10, but only four medium/high-contamination holdout cases satisfied
the matched-attenuation rule; that comparison is therefore too sparse for a strong general
inference. Five of six checks failed.

The demonstrated result is coexistence of high fault-proxy retention with negligible or adverse
median attenuation. The interpretation that retention remained high because the realized
controller removed very little is consistent with these component measurements, but it is not a
separate causal estimate.

![Identifiability checks against frozen gates](../reports/audit/care_asd_identifiability_audit_v1/identifiability_gates.svg)

![Fault-retention/noise-attenuation frontier](../reports/audit/care_asd_identifiability_audit_v1/ap_care_mechanism_frontier.svg)

### 5.7 Derived post-hoc DCASE and real/emulated sensitivity

All quantities required by the exact DCASE 2026 development metric were present in the three frozen
score files. The official definition takes the harmonic mean over source-domain AUC, target-domain
AUC, and pAUC cells for every machine type and section. A new deterministic script checked
identical file metadata and hashes, then derived the 21-cell harmonic score without retraining,
tuning, threshold changes, subset search, or evaluation access.

| Scope | Machines | B00 | B01 | B01-B00 | B02 | B02-B00 |
|---|---:|---:|---:|---:|---:|---:|
| All | 7 | 57.257% | 55.336% | -1.921 pp | 57.125% | -0.132 pp |
| Real synchronized | 2 | 52.732% | 51.124% | -1.609 pp | 52.989% | +0.256 pp |
| Emulated | 5 | 59.292% | 57.221% | -2.071 pp | 58.966% | -0.326 pp |

These values are **derived post hoc from frozen predictions/artifacts**. The overall harmonic score
is a secondary benchmark-alignment analysis; the paired delta AUC/pAUC remains the frozen
inferential quantity. The real-versus-emulated split follows dataset construction rather than
observed performance. With two real and five emulated machine types, subgroup values are
descriptive only. In particular, B02's small positive real-only delta is not evidence of a reliable
real-machine benefit.

## 6. Discussion

### 6.1 What is demonstrated

Under one locked detector realization, B01 did not improve the primary AUC and harmed the frozen
secondary pAUC. B02 did not establish improvement. Controlled tests showed that the corrected
SAFE-REF selector accepted many cases that failed its hidden safety/efficacy label, while AP-CARE
did not jointly achieve useful environmental attenuation and the declared mechanism checks. These
results support a bounded empirical conclusion: the tested normal-only observables and controllers
did not establish a reliable safe-removal region in this DCASE-2026-aligned development regime.

B01 is deliberately a broad residual-replacement stress test, not a surrogate for modern BSS or
every reference canceller. B02 is one conservative, capacity-controlled near-primary design, not a
representative sample of all gates. SAFE-REF and AP-CARE similarly evaluate their declared selectors
and component generators. Their joint value is triangulation of one safety mechanism under frozen
contracts; the set does not exhaust the algorithm class and cannot support a claim that another
selector, representation, or structurally informed BSS method would fail.

The formal statements are different. An unstructured instantaneous/local-bin latent model has
non-unique coordinates, but this alone does not prove physical machine/environment relabelling.
Separately, a declared unseen-fault class containing retention-benign and adverse extensions along a
suppressed direction prevents a normal-only rule from uniformly certifying nontrivial removal as
fault-safe. These statements explain why correlation alone is not
a safety certificate; they neither follow from the empirical failures nor predict that every
algorithm will fail. Proposition 1 is logically independent of the latent-coordinate lemma and is a
conditional no-free-lunch rationale rather than the paper's primary novelty.

### 6.2 Why a far microphone can still be useful

A far channel can improve representation learning, provide spatial or contextual features, support
multi-view consistency, or diversify an ensemble. None of these uses requires the assumption that
all reference-correlated energy is removable noise. Auxiliary-reference usefulness and safely
removable-noise certification are therefore distinct. The CARE-ASD result concerns the latter and
does not contradict successful dual-channel systems addressing the former.

### 6.3 What could make safe cancellation identifiable

Safe removal may become defensible with measured geometry and transfer paths, validated source
independence and suitable non-Gaussianity, source-specific generative models, sparsity or temporal
structure, identifiable nonstationarity, reference-only intervals, supervised components,
additional sufficiently diverse spatial observations, or a justified model of the admissible fault
subspace. These are the kinds of structure used by ICA/BSS and supervised separation; CARE-ASD does
not dispute their identifiability results. Each assumption must be supported in the target regime
rather than inferred from a good aggregate ASD score.
Component-level attenuation, retention, false-safe risk, and untouched prospective validation
would still be required.

### 6.4 How future ASD preprocessing should be evaluated

Future work should retain an unchanged near path where possible; compare interventions under a
fixed downstream detector; measure environmental attenuation and fault retention jointly; report
coverage and false-safe behavior for abstaining selectors; stratify source/target and real/emulated
conditions without outcome-driven subset selection; and freeze holdout decisions before outcomes
are read. A positive ASD metric is utility evidence, not by itself a safety certificate.

### 6.5 Epistemic role of V1-V10

FP-NAA V1-V10 were sequential, outcome-informed attempts to localize and preserve evidence after
the frozen failures. Their value is transparency of research history, narrowing of mechanisms,
evidence against reporting one conveniently selected failed variant, and support for the final stop
decision. They are not ten independent replications, are not pooled with B01/B02 inference, and are
kept in supplementary and reproducibility records.

## 7. Limitations

1. B00/B01/B02 use a single locked training realization (seed 13711). Evaluation bootstrap does
    not quantify training-initialization or optimizer uncertainty; primary inference is conditional
    on that realization.
2. The paired bootstrap resamples files within machine-section-condition strata. Its intervals are
   conditional on file-level exchangeability and do not account for unidentified acquisition-level
   clustering or population variation over machine types. They must not be read as deployment-wide
   confidence intervals.
3. The registered primary endpoint for B01/B02 is mean AUC. The harmful B01 pAUC interval is an
   informative prospectively specified secondary result, not a replacement primary endpoint.
4. The exact DCASE harmonic metric and real/emulated split were derived post hoc from frozen
   development predictions. They are descriptive and have no new tuning or confirmatory status.
5. Only ToyCar and fan are real synchronized development machine types. The five other types are
   emulated, and neither group supports general inference to unseen evaluation machines.
6. The detector is an official-compatible autoencoder, not the full class of contemporary audio
   foundation models, learned stereo representations, raw-waveform systems, or ensembles.
7. Controlled faults and acoustic paths are synthetic proxies built with recorded normal carriers.
   They provide component identity inside the generator but do not span all physical failure modes,
   rooms, nonlinear propagation, or sensor mismatch.
8. SAFE-REF's corrected protocol followed inspection of a first failed simulation. Its corrected
   calibration/holdout split remains diagnostically useful, but the study is not a pristine
   original preregistration.
9. AP-CARE was designed after SAFE-REF. Only four holdout cases entered its matched medium/high
   retention comparison, limiting the strength of that check despite the frozen rule.
10. Phase 8 is heterogeneous, feature-space, and non-causal. It cannot identify which physical
   component caused a score change.
11. The change-of-basis lemma proves only latent-coordinate non-uniqueness in an unstructured
    instantaneous/local-frequency-bin model. It does not prove that transformed coordinates retain
    physical machine/environment provenance, and no arbitrary convolutional theorem is claimed.
12. Proposition 1 is conditional on an admissible future-fault class that has not been restricted
    away from suppressed directions. It is a uniform-certification limitation, not a probability
    statement about real future faults or a claim that all processors fail.
13. V1-V10 are dependent research history, not replications. Their gates cannot be pooled to inflate
    statistical certainty.
14. The literature search has a fixed cutoff and a bounded source set; it is not an exhaustive
    systematic review. The expanded prior-art audit found close precedents for component-separated
    evaluation, contaminated-reference distortion control, downstream utility, and risk--coverage;
    only their joint ASD-specific decision integration remains a possible differentiator. Several
    relevant DCASE reports and arXiv manuscripts were not peer reviewed at the cutoff, and all
    citation metadata requires final manual verification before submission.

## 8. Reproducibility and evidence integrity

Audit-A4 is the frozen evidence boundary. Its regeneration at commit
`f1d5f7fadea74de6e9c7fdefcb172962b3298b63` matched 23/23 derived artifacts and 14/14 quantitative
source hashes, with 29 audit tests passing. The audit package is under
`reports/audit/care_asd_identifiability_audit_v1/`; robustness is under
`reports/audit/care_asd_robustness_appendix_v1/`; literature evidence is under
`reports/audit/care_asd_literature_audit_v1/` and
`reports/audit/fp_naa_post_v10_literature_delta/`.

The exact DCASE and real/emulated sensitivity analysis is new and isolated under
`reports/audit/care_asd_dsp_frozen_sensitivity_v1/`. Its `run.json` records the post-hoc label,
frozen input paths and hashes, source snapshot, machine partition, pairing checks, code hashes, and
prohibited actions; `manifest.json` records generated-output hashes. The generator refuses to
overwrite the directory.

No Audit-A4 source or derived artifact is modified by this manuscript revision. No evaluation
labels or scores were accessed. Restricted audio is not redistributed. The source documents for
the formalization, red-team review, and audit protocol are
[`IDENTIFIABILITY_FORMALIZATION.md`](IDENTIFIABILITY_FORMALIZATION.md),
[`DSP_REVIEWER_RED_TEAM.md`](DSP_REVIEWER_RED_TEAM.md),
[`DSP_BSS_REVIEWER_ATTACK.md`](DSP_BSS_REVIEWER_ATTACK.md), and
[`CONTAMINATED_REFERENCE_SAFETY_AUDIT_PROTOCOL.md`](CONTAMINATED_REFERENCE_SAFETY_AUDIT_PROTOCOL.md).
The expanded positioning, requested expert review, and conditional-inference sign-off are recorded
in [`DSP_NOVELTY_PRIOR_ART_AUDIT.md`](DSP_NOVELTY_PRIOR_ART_AUDIT.md),
[`BSS_EXPERT_REVIEW_PACKET.md`](BSS_EXPERT_REVIEW_PACKET.md), and
[`DSP_STATISTICS_EXTERNAL_VALIDITY_SIGNOFF.md`](DSP_STATISTICS_EXTERNAL_VALIDITY_SIGNOFF.md).

## 9. Conclusion

A synchronized far microphone may be useful auxiliary evidence without being a noise-only
reference. When machine-origin and environmental signals reach both microphones, correlation alone
does not certify a shared component as safely removable. The controlled CARE-ASD audit separates
downstream utility from component efficacy, retention, selector risk/coverage, and frozen holdout
decisions. In the tested DCASE-2026-aligned development regime, broad
reference-correlated suppression failed to improve primary AUC and harmed secondary pAUC;
conservative gating did not establish improvement; and controlled selectors did not demonstrate a
joint region of useful environmental attenuation and fault-proxy preservation. This is a bounded
empirical component-safety/efficacy result, not a claim that dual-microphone ASD is ineffective.
The latent-coordinate lemma and conditional uniform-certification proposition supply scoped
rationale; they do not prove physical source-label non-identifiability or general BSS impossibility.
The practical contribution is a reproducible ASD decision contract requiring utility, efficacy,
retention, selector safety, and prospective stopping before a contaminated reference is called
safely removable noise.

## Provisional references and verification status

The citation links and source roles are frozen in `docs/LITERATURE_MATRIX_AUDIT_A1.md` and the two
literature-audit packages. Before journal submission, authors must manually verify every title,
author list, year, DOI, venue, pagination, and publication status against the publisher or official
record. In particular, DCASE challenge technical reports must remain labeled non-peer-reviewed,
and 2026 arXiv manuscripts must not be presented as reviewed journal evidence unless their status
is independently updated and verified.
