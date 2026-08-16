# Audit-A1 literature matrix and novelty boundary

## Decision

Literature was frozen on 2026-08-16 using primary sources only. The defensible
paper is not a claim that dual-microphone ASD fails. Several DCASE 2026 systems
report useful dual-channel processing, and NA-SSL reports the winning Task 2
score. The retained paper instead asks a narrower measurement question:

> Does deterministic, normal-only signal-level processing of a contaminated
> far reference remove environmental noise without suppressing unknown fault
> evidence?

Working title:

> **When the Noise Reference Contains the Machine: A Controlled Safety Audit
> for Anomalous Sound Detection**

The contribution is the frozen ASD-specific audit protocol: an exact aligned
near-only comparator, anomaly-score deltas with paired uncertainty, injected
known components that separately expose noise attenuation and fault retention,
and preregistered stopping. It is not a new claim about the existence of
reference leakage, which has signal-processing prior art at least as early as
Al-Kindi and Dunlop (1989).

## Primary-source matrix

| Source | Method and evidence | What it establishes | Boundary for this paper |
|---|---|---|---|
| [DCASE 2026 Task 2 specification](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring) | Official synchronized near/far benchmark; normal-only training; AUC and pAUC | Far audio is auxiliary evidence in a first-shot ASD problem | Development results do not support claims about hidden evaluation machines |
| [Nishida et al. (2026)](https://arxiv.org/abs/2606.01578) | Task description; far mic expected to contain stronger noise and weaker direct machine sound | The reference is relatively noise-dominant, not guaranteed noise-only | Reference purity must be measured rather than assumed |
| [DCASE submission policy](https://dcase.community/challenge2026/submission) | Official publication-status definition | Challenge technical reports are not peer reviewed; workshop papers follow a separate review route | Controls evidence labels throughout the manuscript |
| [Fujimura et al. (2026)](https://arxiv.org/abs/2608.00447) | Simulated dual-channel NA-SSL; BEATs/EAT/DaSheng; reported winning score | Learned use of far audio can substantially help ASD | Direct counterexample to generic “dual-mic is ineffective” language; arXiv preprint |
| [Kim (2026)](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Kim_91_t2.pdf) | Minimum-statistics noise transfer and floored/adaptive spectral subtraction | Deterministic normal-only signal processing can improve reported ASD score | Closest direct prior; no separate injected-fault retention endpoint; non-peer-reviewed technical report |
| [Morita et al. (2026)](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Morita_50_t2.pdf) | Coherence-weighted signal route and spatial descriptors with a memory bank | Spatial information can help, but observed coherence can be weak | Supports an observability problem; no known-component safety endpoint; non-peer-reviewed technical report |
| [Ozeki et al. (2026)](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Ozeki_101_t2.pdf) | Far-reference linear noise cancellation before AE and pretrained backends | Direct deterministic reference cancellation is an active ASD approach | Close comparator; assumes the far signal can serve as the noise reference; non-peer-reviewed technical report |
| [Chu and Qian (2026)](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Qian_65_t2.pdf) | Far-reference spectral subtraction before fine-tuned EAT and kNN | Signal enhancement plus a learned backend reports strong development metrics | Same frontend family but no component-level retention endpoint; non-peer-reviewed technical report |
| [Kim et al. (2026)](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Kim_27_t2.pdf) | Scaled near-minus-far residual in pretrained embedding space | Selected embedding residuals improve reported development score | Outside the signal-level deterministic claim; non-peer-reviewed technical report |
| [Lei (2026)](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Lei_23_t2.pdf) | Task-adapted dual-mic representation and calibrated fusion | Explicitly treats far audio as a mixed observation and avoids direct subtraction | Independently states the risk but does not perform a component-frozen safety audit; non-peer-reviewed technical report |
| [Al-Kindi and Dunlop (1989)](https://doi.org/10.1016/0165-1684(89)90005-4) | Cross-coupled adaptive cancellation with desired-signal leakage in the reference | Reference contamination and low-distortion mitigation are established prior art | Novelty cannot be “first contaminated-reference canceller” |
| [Xiao and Doclo (2024)](https://doi.org/10.1109/ICASSP48485.2024.10445843) | Component-aware simulations of target/delay choices in active noise control | Noise reduction must be interpreted jointly with desired-signal quality and causality | Provides evaluation precedent outside ASD, not an ASD novelty conflict |
| [Zhao and Wang (2025)](https://arxiv.org/abs/2511.03244) | Learned Wiener-mask purification of an auxiliary AEC reference contaminated by near-end speech | Learned purification can make a contaminated auxiliary mic useful | Alternative supervision/model family outside this deterministic normal-only audit; arXiv preprint |

DCASE explicitly states that challenge technical reports are not peer reviewed;
they are valuable current primary descriptions but must not be presented as
journal evidence. The two 2026 arXiv works were also treated as preprints at the
cutoff date.

## Novelty test

### What is already known

1. Desired-signal leakage into a reference microphone can distort adaptive
   cancellation, and specialized controllers can mitigate it under stated
   assumptions.
2. DCASE 2026 near/far information can improve ASD through learned
   representations, embedding residuals, spatial features, and some
   deterministic frontends.
3. Development AUC/pAUC is the accepted task performance endpoint.

### Gap retained by this study

Across the direct ASD primary sources reviewed through the cutoff, none jointly
reports all four of the following:

1. an aligned downstream ASD effect against the same near-only detector;
2. known noise and fault components held fixed for physical interpretation;
3. separate noise-attenuation and fault-retention endpoints; and
4. a frozen rule that stops unsafe or non-identifiable processing.

This is a bounded literature-audit finding, not proof that no unpublished or
unindexed study exists. The paper should use “to our knowledge, among the first
ASD-specific controlled safety audits” until journal submission search is
refreshed.

## Claim ledger

Supported wording:

- “Under the tested deterministic normal-only signal-level controllers,
  observable stereo heuristics did not identify a reliable safe cancellation
  regime.”
- “A downstream ASD-score improvement is not, by itself, evidence that unknown
  fault energy was preserved.”
- “The controlled findings motivate known-component safety endpoints alongside
  AUC/pAUC for contaminated-reference ASD.”

Prohibited wording:

- “Dual-microphone ASD is ineffective.”
- “Far microphones are harmful.”
- “Reference contamination is a novel problem.”
- “Safe normal-only reference processing is universally impossible.”
- “The result generalizes to DCASE evaluation machine types.”
- “DCASE challenge technical reports are peer reviewed.”

## Journal feasibility

The direction remains feasible for *Digital Signal Processing* only if the
manuscript is presented as a signal-processing evaluation methodology and
empirical boundary, with the negative result supported by the aligned and
known-component evidence chain. It is weak as a new-algorithm paper because the
proposed controllers failed their frozen gates and stronger dual-mic methods
already report positive task results.

No new GPU training, evaluation-set access, or Jetson benchmark is justified by
Audit-A1. Audit-A2 should now test machine/domain robustness using only frozen
scores.

## Reproduction

```bash
uv run care-asd audit literature --config configs/research/audit_literature_v1.yaml --repo-root . --output-dir reports/audit/care_asd_literature_audit_v1
```

The config is the curated source of truth. The command validates publication
status, unique source identifiers/URLs, and every claim-to-source link before
writing immutable CSV, JSON, Markdown, and provenance hashes.
