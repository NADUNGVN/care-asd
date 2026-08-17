# FP-NAA literature and claim update — 2026-08-17

Status: successor-track evidence update. This document does not alter the frozen CARE-ASD
identifiability audit or reinterpret any AP-CARE result.

## Decision

FP-NAA remains a defensible research direction, but only as a **mechanism-and-safety**
contribution. It is not a claim to have invented reference-conditioned denoising, cross-attention,
BEATs adaptation, pseudo faults, or dual-microphone ASD. A development score above the registered
63.00% gate is not by itself a state-of-the-art result.

The positive paper claim is permitted only if the same-capacity C2 adapter improves over C1 under
the exact paired official metric, the confidence interval excludes zero, the advantage generalizes
across held-out machine types, and the injected-fault retention gates pass. This is stronger causal
evidence for the proposed objective than a leaderboard-only comparison, while remaining honest
about absolute task performance.

## Primary-source evidence

| Source | Direct evidence | Consequence for FP-NAA |
| --- | --- | --- |
| [NABEATs](https://arxiv.org/abs/2607.16688) | Trains reference-conditioned BEATs denoising with clean-representation MSE. It reports a case where representation SNR improves while downstream performance degrades and leaves better objectives to future work. | Directly motivates a downstream-aware preservation objective. Cross-attention and MSE distillation are prior art and cannot be claimed. |
| [Anomalous Sound Detection Meets Noise-Aware SSL](https://arxiv.org/abs/2608.00447) | Applies noise-aware SSL to DCASE 2026 and reports 70.24% official evaluation score for the winning NA-BEATs system versus 65.46% for second place. | FP-NAA cannot claim generic dual-microphone effectiveness or SOTA from a 63% development result. It must isolate the objective effect and safety evidence. |
| [DCASE 2026 Task 2 description](https://arxiv.org/abs/2606.01578) | Defines normal-only first-shot ASD with synchronized near/far channels and the source-AUC, target-AUC, and pAUC evaluation cells. | The exact official harmonic score and normal-only training boundary remain mandatory. |
| [Official DCASE 2026 results](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring-results) | Shows a broad performance frontier and substantial machine-to-machine heterogeneity. | Mean development improvement alone is insufficient; per-machine drops and LOMO are necessary. |
| [MERL technical report](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Fujimura_17_t2.pdf) | Reports 60.28% for original BEATs with frequency-wise memory, average pooling, and score rescaling, and 64.57% for Dis NA-BEATs with RDP(4). It does not report the exact FP-NAA C0 backend. | These are non-identical context points; the 60.50% C0 gate is an internal fidelity threshold, not a claimed reproduction of a published 62.02% result. |
| [AITHU technical report](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Jiang_125_t2.pdf) | Reports 68.20% development score using heterogeneous BEATs scoring/fine-tuning branches and score fusion; its single-branch reference is 64.04%. | Current strong systems exceed the FP-NAA absolute gate, but use materially different capacity and selection. They are context, not a capacity-matched causal comparator. |
| [Noise-Aware Reference Denoising](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Kim_91_t2.pdf) | Uses far-channel minimum-statistics noise transfer plus floored spectral subtraction, reaching 64.70% with an ensemble and documenting over-suppression failures. | Far-reference denoising and signal-loss safeguards are prior art. V4 must claim only its representation-space structural invariant and evidence protocol. |
| [Deep ANC](https://www.isca-archive.org/interspeech_2020/zhang20i_interspeech.html) | Predicts a cancelling signal from a reference input with a CRN. | A learned reference-only correction is not novel by itself. |
| [Interference-aware AEC target](https://www.isca-archive.org/interspeech_2024/khanagha24_interspeech.html) | Documents near-end deletion during double talk and modifies the training target to reduce it. | Target deletion is an established safety problem; V4 differs by imposing an exact token-input identity Jacobian rather than changing a waveform target. |

DCASE technical reports are current primary system descriptions but are not peer-reviewed journal
evidence. The two noise-aware SSL papers are preprints at this cutoff. Claims must retain these
publication-status qualifiers.

## Frozen comparator interpretation

- **C0** is a reproducible, training-free BEATs + frequency-RDP8 + BEAM backend. Its 60.50% gate
  is a pre-registered internal implementation-fidelity threshold; it is strong and
  transparent, but not leaderboard SOTA.
- **C1** is the target-conditioned cross-attention adapter trained with normal-representation MSE.
  It is the primary causal baseline for the V4 claim.
- **V4 C2** has the same module inventory, parameter count, data, seeds, optimizer, MSE objective,
  and scoring backend as C1, but its learned correction depends only on the reference. The isolated
  intervention is the exact `F(x + delta, r) - F(x, r) = delta` architecture. C2 can support a
  contribution only through C2-minus-C1 paired evidence plus the absolute C0 and machine-level
  safeguards. Comparing C2 only with the official baseline is insufficient.
- Contemporary 67–70% systems provide external performance context. Their unavailable training
  assets, ensembles, label-selection policies, or different model capacities must not be
  reconstructed approximately and presented as exact replications.

## Claim ledger

Permitted after a full G3 pass:

> Under a frozen BEATs backend, a capacity-matched reference-only correction with exact target-token
> perturbation equivariance improved the exact DCASE score over target-conditioned MSE denoising,
> with a positive stratified paired-bootstrap interval, cross-machine consistency, and preserved
> held-out pseudo-fault evidence.

Still prohibited:

- FP-NAA is state of the art on DCASE 2026;
- a 63% development score is competitive with the winning evaluation system;
- pseudo-fault retention proves preservation of every real mechanical fault;
- development-set LOMO proves generalization to hidden evaluation machines;
- the far microphone is noise-only; or
- reference-conditioned cross-attention is novel.
- reference-only noise cancellation, residual addition, or equivariance in general is novel.

## Evidence required before manuscript conversion

1. C0 must reproduce at or above 60.50%; otherwise no candidate result is interpretable.
2. C2 must beat both C0 and capacity-matched C1 by the frozen G2 margins without a machine drop
   beyond the registered bound.
3. The five-seed C2-minus-C1 official-score bootstrap interval must exclude zero; seed means or an
   unpaired interval are insufficient.
4. At least six of seven confirmatory LOMO folds must favor C2.
5. In-support, held-out-family, unmatched-reference, dropout, channel-swap, and leakage retention
   gates must pass.
6. Absolute scores must be reported alongside the 67–70% contemporary frontier, without implying
   an apples-to-apples replication.

Failure of any item closes the positive method-paper claim under the current formulation. The
completed negative audit remains publishable on its own and is never overwritten by this successor
track.
