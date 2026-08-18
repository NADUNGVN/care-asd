# FP-NAA v7 conditional counterfactual-observability-gated evidence bypass

## Status and authorization boundary

This conditional mechanism was specified after the valid V5 negative result and before any V6
encoder-tap result was available. Its working name is **counterfactual-observability-gated
evidence bypass (COGEB)**. Committing this document does not authorize an experiment. V7 may be
implemented only if the immutable V6 gate selects a tap. The selected tap is copied mechanically
from that gate; it cannot be changed after development anomaly labels are inspected.

If V6 selects no tap, COGEB is rejected without implementation. If a tap is selected but the
held-out friction diagnostic later fails the unchanged G2 retention gate, COGEB is rejected at G2.
Neither event permits a tap, fusion rule, threshold, or pseudo-fault family sweep.

## Evidence that motivates the conditional mechanism

The strongest frozen comparator remains the registered three-seed C1 mean at 0.6340566. Valid V5
ACTT reached 0.6341576, only +0.0001010 over C1 instead of the frozen +0.005 minimum. Its
adapter-level in-support retention was 0.9351 median and 0.6647 q05, but final-BEATs frontend
retention was only 0.7731/0.1340. Training the final-token adapter harder is therefore not the
registered next hypothesis.

V4 established a useful structural control. A reference-only map

```text
F(x, r) = x + G(r)
```

has an exact identity Jacobian with respect to the target and cannot further attenuate a target
perturbation already present at its input. V4 nevertheless failed because the final frozen BEATs
input had already lost much of the clean-teacher delta. V6 tests whether an earlier frozen tap
retains that delta. COGEB uses an eligible shallow/intermediate representation as a separate
anomaly-evidence path instead of forcing it through all remaining Transformer blocks and risking
the same loss again.

## Frozen architecture if V6 passes

Let `k` be the deepest V6-eligible tap, `x_k` and `r_k` the paired near/far grids at that tap, and
`s_C1` the immutable final-layer C1 anomaly score for the matching seed.

### Safe intermediate branch

The only trainable V7 component is a tap-local reference-only correction

```text
H_k(x_k, r_k) = x_k + G_k(r_k).
```

`G_k` uses the same bandwise module inventory as the registered reference-only V4 adapter and is
trained only on normal counterfactual-cache inputs to minimize clean-teacher tap MSE. It never
reads the target tokens when computing its correction. Therefore

```text
H_k(x_k + delta, r_k) - H_k(x_k, r_k) = delta
```

for every finite representable `delta`. The candidate is scored directly at tap `k` with the
unchanged RDP(8)/BEAM backend. Its output is not passed through later BEATs blocks.

### Cross-fitted normal-calibrated evidence union

The final C1 branch is retained rather than replaced. For each machine/section and branch, normal
training clips are scored against a bank that excludes the query clip. The resulting leave-one-out
normal scores define an empirical CDF. A test score `s_b` becomes

```text
q_b = ECDF_b(s_b),
```

with deterministic mid-rank handling and clipping to `[1/(n+1), n/(n+1)]`. The pre-calibration
union statistic is

```text
u = max(q_C1, q_safe).
```

The joint leave-one-out normal pairs define `ECDF_union`; the reported anomaly score is

```text
score = -log(1 - ECDF_union(u)).
```

This second calibration accounts empirically for dependence between the two branches. It uses no
anomaly label, proxy-outlier accuracy, tunable fusion weight, or test-set statistic. Before the
last monotone calibration, `u` cannot be smaller than either branch's calibrated evidence, which
is the score-level evidence-bypass invariant.

## Required ablations and frozen decisions

For every registered seed, the same selected tap and backend produce:

- `B0_final_c1`: immutable final-layer C1 score;
- `B1_raw_bypass`: C1 plus the uncorrected selected-tap branch using the identical calibrated
  union, isolating ordinary multi-depth fusion;
- `B2_cogeb`: C1 plus the reference-only corrected tap branch, the proposed candidate.

There is no branch-weight search. C1 artifacts remain byte-identical to their registered source.
The three screening seeds remain `[13711, 42, 2026]`; the five confirmatory seeds remain
`[13711, 42, 2026, 3407, 777]`. Final epochs are used without anomaly-label checkpoint selection.
All original G2/G3, LOMO, bootstrap, worst-machine, held-out retention, and reference-safety gates
remain unchanged.

In addition to those existing gates, a positive mechanism claim requires:

- `B2_cogeb - B1_raw_bypass >= 0.0025` in mean screening official score;
- positive per-machine deltas on at least five of seven machine categories versus B0;
- a positive paired-bootstrap 95% lower confidence bound versus both B0 and B1 at confirmation.

Failure against B1 means that any benefit is attributable to prior-art multi-depth fusion rather
than the safe reference correction. Such a run may remain a useful system result but cannot support
the proposed mechanism claim.

## Literature and novelty boundary

The following components are explicitly prior art and are not claimed as new:

- BEATs and intermediate SSL representations;
- retaining mixture information rather than treating all interference as removable noise;
- reference-only cancellation and exact residual identity paths in general;
- multi-branch, multi-layer, dual-channel, and score-level ASD fusion;
- empirical score calibration, nearest-neighbour memory banks, and BEAM-style backends.

The 2026 backend study also reports that choosing backends with proxy outliers can fail because the
proxy task saturates. COGEB therefore uses pseudo faults only for a representation-observability
eligibility test, never for branch weights or score selection. Development anomaly labels are
accessed once for the frozen G2 evaluation.

The candidate contribution is deliberately narrower: a normal-only counterfactual observability
gate authorizes an exact target-equivariant intermediate evidence path, and a cross-fitted joint
normal calibration lets that path bypass downstream evidence attenuation without anomaly-label
fusion tuning. Novelty remains unproven until the full literature audit and all registered
mechanism, LOMO, confirmatory, bootstrap, and safety gates pass.

Primary sources:

- [Official Microsoft BEATs implementation](https://github.com/microsoft/unilm/tree/833df7e7832e5064a281131ee64a481afa8e5b95/beats)
- [BEATs: Audio Pre-Training with Acoustic Tokenizers](https://proceedings.mlr.press/v202/chen23ag.html)
- [Retaining Mixture Representations for Domain Generalized ASD](https://arxiv.org/abs/2510.25182)
- [Scoring Backends Matter More Than Pooling](https://arxiv.org/abs/2606.19269)
- [DCASE 2026 AITHU BEATs scoring/fusion system](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Jiang_125_t2.pdf)
- [DCASE 2026 VUI Labs dual-channel BEATs fusion](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Qian_108_t2.pdf)
- [ECHOv2 two-level band-splitting ASD representation](https://arxiv.org/abs/2607.10596)

