# FP-NAA v9 pre-encoder tangent repair and evidence bypass

## Frozen decision after V8

The valid V8-M run `server02_fp_naa_layerwise_preflight_20260818T060510Z` used source
`25d42347332a71d7898d3e6bc7eca822dc048a37`, passed the pinned-BEATs runtime probe, completed all
30 registered epochs, and produced finite updates. It nevertheless failed its immutable mechanism
gate. L2 improved in-support retention median over the capacity-matched L1 control by 0.0665, but
improved q05 by only 0.0255. Its absolute in-support median/q05 were 0.7811/0.1092, and held-out
friction-burst median/q05 were 0.8311/0.3002. Only 4.3% of held-out clips improved over L1.

V8 is therefore closed without G2. Its loss weights, layer set, seed, and thresholds are not tuned.
The result localizes the failure: a target-conditioned repair repeated through twelve nonlinear
Transformer blocks can improve familiar perturbations while systematically damaging unseen fault
evidence. V9 moves the repair to a scored pre-encoder branch rather than changing V8.

## Hypothesis and novelty boundary

BEATs tap 0 is the normalized patch projection before positional convolution and the twelve
Transformer blocks. V6 showed that it retains more counterfactual evidence than every deeper tap,
although its raw 0.8465/0.4094 median/q05 was not sufficient to authorize the rejected V7 method.
V5 independently showed that anchored counterfactual tangent transport (ACTT) can preserve the
delta presented to one adapter, including an unseen friction family (approximately 0.91/0.79
adapter-level median/q05), while its final-token location could not recover evidence already lost
upstream.

V9 combines these two pieces of negative-result evidence. A single capacity-matched bandwise
near/far adapter repairs noisy tap-0 tokens toward clean tap-0 teacher tokens. The repaired tokens
are scored directly by RDP(8)/BEAM and never pass through the twelve BEATs blocks. A later G2
candidate, authorized only by the mechanism preflight, joins this branch with the immutable final
C1 score using the already specified cross-fitted normal-calibrated evidence union.

Neither an intermediate feature, cross-attention, ACTT/Jacobian matching, RDP/BEAM, multi-depth
fusion, nor score calibration is claimed as new. The provisional contribution is narrower:

> a normal-only pre-encoder counterfactual repair whose scored evidence path bypasses the exact
> nonlinear depth at which a registered observability audit measured fault attenuation, tested
> against raw-bypass and capacity-matched MSE-repair controls under held-out-family, LOMO, paired
> bootstrap, and reference-safety gates.

Jacobian matching is established prior art and is equivalent to distillation under local input
perturbations. NA-BEATs layerwise denoising is also prior art and achieved the strongest DCASE 2026
result. DCASE systems already use multi-layer and DSP fault-sensitive score branches. These facts
forbid broad novelty claims; V9 can support only the localized repair-and-bypass claim above.

Primary boundary sources:

- [Knowledge Transfer with Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html)
- [Anomalous Sound Detection Meets Noise-Aware SSL](https://arxiv.org/abs/2608.00447)
- [NABEATs](https://arxiv.org/abs/2607.16688)
- [Official BEATs implementation](https://github.com/microsoft/unilm/tree/833df7e7832e5064a281131ee64a481afa8e5b95/beats)
- [DCASE 2026 Task 2 systems and reports](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring-results)

## Frozen V9-M mechanism preflight

The authoritative configuration is `configs/experiment/fp_naa_v9.yaml`. Before any new
development anomaly label is read, seed 2608 selects the same stable 1,024/512 normal-only
train/validation split and 256 held-out clips used by V8-M. A new immutable cache contains tap-0
grids for noisy normal, far reference, pseudo-fault noisy, clean teacher, and fault teacher views.

One zero-initialized adapter receives ten common normal-MSE epochs. Two byte-identical copies then
receive ten more epochs and the same data order:

- `P1_tap0_mse`: normal clean-teacher MSE only;
- `P2_tap0_actt`: the same normal MSE plus the registered ACTT relative-error mean/tail term and
  common-function anchor.

Periodic resonance, amplitude modulation, and frequency modulation are optimizer-visible only for
P2. Friction burst remains wholly unseen. No tap, layer, width, epoch, seed, weight, threshold, or
checkpoint search is permitted. The ACTT constants are copied from V5/V8 rather than selected from
V9 results.

V9-M passes only if all checks hold:

- zero-initialized tap-0 adapter identity and a finite non-zero real-CUDA update;
- P2 in-support retention median >= 0.90 and q05 >= 0.60;
- P2-P1 median >= 0.05 and q05 >= 0.10;
- held-out friction median >= 0.85 and q05 >= 0.65;
- median normal-function drift from the common branch <= 0.10;
- every stored tensor, loss, gradient norm, and update is finite.

A pass authorizes implementation and execution of V9 G2 only. It is not a development performance
result and does not authorize LOMO. A failure closes V9 without development-label scoring or a
hyperparameter sweep.

## Conditional G2/G3 protocol

If V9-M passes, each registered screening seed produces four fixed score sets:

- `B0_final_c1`: immutable strong final-layer C1;
- `B1_raw_tap0_union`: C1 plus raw tap-0 evidence;
- `B2_mse_repair_union`: C1 plus P1;
- `B3_actt_repair_union`: C1 plus P2, the proposed candidate.

Every union uses leave-one-out train-normal ECDF calibration followed by a joint normal calibration;
there is no weight search. In addition to all unchanged G2 checks, B3 must improve mean official
score over B0 by at least 0.005, over B1 by at least 0.0025, and over B2 by at least 0.0025. Failure
against B1 attributes any gain to prior-art multi-depth fusion; failure against B2 rejects the ACTT
mechanism. LOMO, five-seed confirmation, paired bootstrap, reference safety, paper freeze, and
Jetson measurement remain strictly conditional on the preceding gate.
