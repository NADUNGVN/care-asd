# FP-NAA literature and claim update — 2026-08-17

Status: successor-track evidence update. This document does not alter the frozen CARE-ASD
identifiability audit or reinterpret any AP-CARE result.

## Decision

FP-NAA remains a defensible research direction, but only as an **objective-and-safety**
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
| [AITHU technical report](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Jiang_125_t2.pdf) | Reports 68.20% development score using heterogeneous BEATs scoring/fine-tuning branches and score fusion; its single-branch reference is 64.04%. | Current strong systems exceed the FP-NAA absolute gate, but use materially different capacity and selection. They are context, not a capacity-matched causal comparator. |

DCASE technical reports are current primary system descriptions but are not peer-reviewed journal
evidence. The two noise-aware SSL papers are preprints at this cutoff. Claims must retain these
publication-status qualifiers.

## Frozen comparator interpretation

- **C0** is a reproducible, training-free BEATs + frequency-RDP8 + BEAM backend. Its 60.50% gate
  tests implementation fidelity against the published 62.02% reference; it is strong and
  transparent, but not leaderboard SOTA.
- **C1** has exactly the same architecture, parameter count, data, initialization seeds, optimizer,
  and scoring backend as C2, with normal-representation MSE as the only changed objective. It is
  the primary causal baseline for the FP-NAA claim.
- **C2** can support a contribution only through C2-minus-C1 paired evidence plus the absolute C0
  and machine-level safeguards. Comparing C2 only with the official baseline is insufficient.
- Contemporary 67–70% systems provide external performance context. Their unavailable training
  assets, ensembles, label-selection policies, or different model capacities must not be
  reconstructed approximately and presented as exact replications.

## Claim ledger

Permitted after a full G3 pass:

> Under a frozen BEATs backend and capacity-matched reference-conditioned adapter, counterfactual
> fault-delta preservation improved the exact DCASE score over MSE-only distillation, with a
> positive stratified paired-bootstrap interval, cross-machine consistency, and preserved held-out
> pseudo-fault evidence.

Still prohibited:

- FP-NAA is state of the art on DCASE 2026;
- a 63% development score is competitive with the winning evaluation system;
- pseudo-fault retention proves preservation of every real mechanical fault;
- development-set LOMO proves generalization to hidden evaluation machines;
- the far microphone is noise-only; or
- reference-conditioned cross-attention is novel.

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
