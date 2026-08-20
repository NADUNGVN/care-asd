# FP-NAA v10 counterfactual-certified monotone evidence union

## Decision after V9

The authoritative V9-M run
`server02_fp_naa_tap_repair_preflight_20260818T072423Z` completed on source `c4ea7c4` with finite
real-CUDA updates and normal-function drift 0.0888, but failed every registered fault-preservation
check. The ACTT repair retained only 0.6674/0.2433 median/q05 in-support evidence and
0.6739/0.3695 on held-out friction burst. The capacity-matched MSE repair was similarly weak, while
the untrained tap-0 input retained 0.8429/0.4240. V9 is therefore closed without development-label
screening, LOMO, or a repair hyperparameter sweep.

These results reject a fifth representation-repair attempt. They do not reject the observed
complementarity among immutable normal-trained score branches. V10 tests that narrower mechanism
without changing BEATs or training another feature adapter.

## Prior-art and novelty boundary

Domain-aware score calibration, maximum fusion, multi-view anomaly scoring, RDP, BEAM, global
nearest-neighbour scoring, and synthetic-anomaly model selection all predate V10. In particular,
DACo already estimates latent domains and performs domain-aware score calibration, and strong DCASE
2026 systems already combine multiple feature depths and backends. V10 does not claim any of these
components separately.

The candidate contribution is restricted to this falsifiable combination:

> Counterfactual-Certified Monotone Evidence Union (CC-MEU): preserve an immutable strong anomaly
> score for every clip; calibrate heterogeneous supplementary experts from cross-fitted training
> normals; charge each expert for its upper normal-tail activation; admit an expert using only
> in-support pseudo-fault counterfactual gain; require transfer to a never-selected fault family;
> and confirm the frozen policy on unseen evaluation machines.

The structural property is per-clip rather than average: for every input `x`,
`S_CC-MEU(x) >= S_C1(x)`. This prevents a supplementary branch from suppressing a C1 alarm. It does
not guarantee that AUC or pAUC improves, so all performance and generalization gates remain
necessary.

Boundary sources:

- [DCASE 2026 Task 2 definition and metric](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
- [DCASE 2026 systems and technical reports](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring-results)
- [DACo latent-domain calibration](https://arxiv.org/abs/2607.04526)
- [Synthetic anomaly model selection](https://openreview.net/forum?id=HW2lIdrvPb)
- [Official post-challenge evaluator](https://github.com/nttcslab/dcase2026_task2_evaluator)

## Frozen V10-M preflight

The authoritative configuration is `configs/experiment/fp_naa_v10.yaml`. V10-M is a
development-anomaly-label-free mechanism preflight and reuses only immutable artifacts:

- C1 checkpoints for seeds 42, 2026, and 13711 from
  `server02_fp_naa_screening_20260817T083029Z`;
- the validated final BEATs cache `beats_iter3_stereo_10s_fp32infer_v2`;
- the normal/pseudo-fault cache `counterfactual_fp32infer_v3`;
- V9's immutable normal-only tap-0 selection and cache.

For each machine, five-fold deterministic cross-fitting produces training-normal anomaly scores.
Each score branch is mapped to add-one empirical upper-tail evidence `-log(p)`. C1 evidence is the
mean across its three frozen seeds. The fixed supplementary experts are:

1. raw tap-0 RDP(8)/BEAM;
2. final-token RDP(4)/BEAM;
3. final-token global average-pooled nearest-neighbour score.

For expert `j`, the penalty is the higher empirical quantile at `1 - 0.01` of its normal evidence
minus C1 evidence. The candidate is

`S(x) = max(S_C1(x), max_j(S_j(x) - penalty_j))`.

Expert selection uses training normals and the three registered in-support pseudo-fault families
only. An expert must stay within 3.5% clean-tail activation and achieve counterfactual evidence-gain
median >= 0.05 and q05 >= 0. The same expert must pass on at least 70% of machines before it is
globally authorized. The held-out friction family never selects an expert or penalty; it is a gate
only.

V10-M passes only if:

- at least one expert is globally authorized;
- the resulting machine-specific policy covers at least 70% of machines;
- mean clean activation remains <= 3.5%;
- aggregate in-support gain median/q05 are >= 0.05/0;
- held-out friction gain median/q05 are >= 0.025/0;
- exact per-clip base monotonicity holds;
- the contract attests that development anomaly labels were not read;
- the policy cryptographically binds the exact V9 source run, C1 checkpoints, normal calibration,
  expert certificates, and summary used for the decision.

A pass authorizes one frozen development screening. It is not a performance result and does not
authorize LOMO, evaluation labels, or paper claims. A failure closes V10 without tuning the expert
list, tail probability, folds, penalties, or thresholds.

## Conditional development screening

If V10-M passes, one three-seed development run evaluates these fixed controls under the exact
DCASE metric:

- raw C1 score ensemble;
- calibration-only C1;
- unpenalized maximum of all registered experts;
- normal-tail-penalized union without counterfactual eligibility;
- full CC-MEU with the immutable V10-M policy.

CC-MEU must improve the official score by at least 0.0075 over raw C1 and 0.0025 over calibrated
C1, with no machine dropping more than 0.010. It must also retain all V10-M counterfactual gates.
Failure closes V10 without LOMO.

## Frozen unseen-machine confirmation

DCASE 2026 evaluation audio and ground truth became public after the challenge. They are suitable
for a post-challenge unseen-machine confirmation only if leakage is controlled. Before downloading
or reading evaluation labels, the code, config, V10-M policy, development result, dependency lock,
and Git commit are frozen and hashed. Evaluation audio is then scored without labels. Ground truth
is loaded only after score files and hashes are final.

The confirmatory claim requires all of the following:

- mean CC-MEU minus calibrated-C1 score >= 0.0025 on unseen evaluation machines;
- machine/domain-stratified paired-bootstrap 95% CI lower bound > 0;
- no hidden development retuning or evaluation-machine-specific policy;
- positive generalization on the preregistered machine-level robustness analysis;
- complete disclosure that this is a post-challenge evaluation, not an official leaderboard
  submission.

Only that result can support a statistically credible positive method claim. Otherwise V10 remains
a bounded negative mechanism study and the research must move to a new preregistered hypothesis.
