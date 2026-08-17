# FP-NAA v3: RDP-salient reference-contraction projection

## Status

This amendment was frozen after the valid deterministic v2 screening run
`server02_fp_naa_screening_20260817T120203Z` and before any v3 result existed. V1 and v2 remain
immutable negative ablations. V3 is a new structural mechanism, not another weighting of the
failed counterfactual objective.

## Evidence that closes v2

V2 completed all six registered runs on source commit `9f82112`:

- C0 official score: 61.9768%;
- C1 mean: 63.4057%;
- v2 C2 mean: 61.0112%;
- C2 - C0: -0.9656 percentage point;
- C2 - C1: -2.3945 percentage points;
- in-support retention median: 0.8795 and worst-seed q05: 0.3037;
- held-out retention median: 0.9662 and worst-seed q05: 0.7435.

The optimizer diagnostic is decisive. Across the three seeds, 84.1--85.4% of auxiliary updates
conflicted with normal-MSE gradients. As soon as the auxiliary curriculum started, C2 normal MSE
moved from approximately 0.036 to 0.063, versus approximately 0.0348 for C1. PCGrad removed the
negative component but retained a large orthogonal component, so it did not constrain the finite
parameter displacement. The retained pseudo-fault magnitude improved while the detector geometry
degraded on fan, gearboxEmu, sliderEmu, and especially valveEmu. V2 therefore failed both its
performance and in-support tail-retention hypotheses. LOMO is prohibited.

## V3 hypothesis

The DCASE two-microphone premise is asymmetric: the far channel contains relatively more
environmental noise, but it may also contain weaker machine sound. A safe adapter should use the
far microphone to explain interference without freely contracting evidence present primarily in
the near microphone. V3 enforces that property in the adapter output geometry instead of asking a
conflicting synthetic-fault loss to learn it.

Let `x_t` and `r_t` be the near and far BEATs grids at temporal row `t`, and let `c_t` be the raw
C1 correction. Define the near--far evidence vector `d_t = x_t - r_t`. On the 20% of temporal rows
with largest `||d_t||^2`, V3 takes the Euclidean projection of `c_t` onto

```text
c'_t dot d_t >= -rho ||d_t||^2,  rho = 0.10.
```

Thus a protected row retains at least 90% of its original near--far component. Rows already inside
the safe half-space are unchanged. Other rows are unchanged. Selecting temporal rows rather than
individual tokens matches the relative-deviation axis used by the frozen RDP(8) backend.

The operator has no trainable parameter. For every seed, C2 loads the exact C1 state dictionary;
the C2 checkpoint records the SHA-256 of its C1 parent. Consequently C1 and C2 have identical
training data, initialization, optimizer trajectory, 989,696 trainable parameters, and normal-MSE
history. Their only causal difference is the registered inference-time projection.

Pseudo faults are no longer optimization targets. They remain a blinded safety probe: the three
in-support families and held-out friction-burst family are evaluated exactly as before. This avoids
claiming that synthetic perturbations are faithful real anomalies while retaining a falsifiable
fault-preservation requirement.

## Why this mechanism is evidence-based

- The [DCASE 2026 Task 2 description](https://arxiv.org/abs/2606.01578) defines the far microphone
  as relatively noise-dominant, not machine-free. The projection encodes that asymmetry directly.
- [NABEATs](https://arxiv.org/abs/2607.16688) establishes reference-conditioned representation
  denoising and reports that improved reconstruction can fail to improve a downstream task. V3
  keeps its MSE adapter as C1 and isolates a downstream-safety operator rather than claiming the
  adapter or cross-attention as new.
- [RDP](https://arxiv.org/abs/2603.04605) emphasizes temporally deviant representation rows. V3
  protects the same temporal axis before the unchanged RDP/BEAM detector.
- Gradient-balancing work such as [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) and
  [CAGrad](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html)
  motivates controlling multi-objective interference, but the observed 84--85% conflict shows that
  continued gradient surgery would not isolate reference leakage in this experiment.

These sources motivate the components but do not establish the combined projection as novel. A
final novelty claim still requires the complete literature audit and all G2/G3 evidence.

## Frozen v3 protocol

The authoritative config is `configs/experiment/fp_naa_v3.yaml`. The projection fraction (0.20)
and contraction limit (0.10) are single preregistered values; no development-label grid search is
permitted. The BEATs cache, augmentation cache, C0 score, C1 optimizer, three screening seeds,
backend, exact metric, and every G2/G3 threshold are unchanged.

The runtime check must pass `rdp_salient_projection_probe` on SERVER-02. Screening may proceed only
in a fresh run directory. LOMO remains blocked unless every G2 core check passes.
