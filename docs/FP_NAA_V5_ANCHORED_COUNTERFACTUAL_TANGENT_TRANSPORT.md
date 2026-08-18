# FP-NAA v5: anchored counterfactual tangent transport

## Status and preregistration boundary

This mechanism was frozen after the valid V4 screening run
`server02_fp_naa_screening_20260817T165229Z` and before any V5 result existed. V1--V4 remain
immutable negative ablations. V5 changes the optimization mechanism and initialization contract;
it does not change C0, C1, pseudo-fault generation, the held-out family, scoring, seeds, or gates.

The working name is **anchored counterfactual tangent transport (ACTT)**. Exact-phrase searches
performed on 2026-08-18 found no prior method using that name. This is not evidence that tangent
matching, counterfactual distillation, function-space anchoring, or CVaR is individually new. The
candidate novelty is limited to their frozen, normal-only, two-channel ASD formulation and its
observability/safety evaluation.

## Evidence that closes V4

V4 was structurally perturbation-equivariant at the adapter input, but the end-to-end retention
gate compares a noisy-input counterfactual with a clean-teacher counterfactual. The valid V4 run
obtained 63.3105%, +1.3338 points over C0 but -0.0951 point versus C1. In-support median/q05
retention was 0.7731/0.1340 and held-out retention was 0.9516/0.5745. V4 therefore failed G2 and
LOMO is prohibited.

The failure localizes the missing mechanism. With

```text
F_v4(x, r) = x + G(r),
```

the adapter-level delta is exactly the noisy BEATs input delta. Across the V4 diagnostics, 93.55%
of counterfactuals had noisy-frontend delta gain below one relative to the clean BEATs teacher.
The low-tail attenuation was especially severe for amplitude modulation and frequency modulation.
V4 cannot reconstruct evidence already suppressed before its adapter.

V5 retains the original combined retention gate but adds a non-gating decomposition:

```text
frontend observability = ||E(x_noisy+fault)-E(x_noisy)|| / ||E(x_clean+fault)-E(x_clean)||
adapter preservation   = ||F_fault-F_clean|| / ||E(x_noisy+fault)-E(x_noisy)||
combined recovery      = ||F_fault-F_clean|| / ||E(x_clean+fault)-E(x_clean)||.
```

This prevents a future structural adapter claim from being confused with upstream observability.
It does not retroactively relax or replace any V4/V5 threshold.

## V5 causal mechanism

For each frozen seed, V5 C2 starts from the exact matching C1 checkpoint. A second frozen copy of
that checkpoint supplies the function anchor `A`. C2 retains C1's target-conditioned architecture
and its 989,696 trainable parameters; there is no extra inference branch or parameter.

For a normal noisy token grid `x`, reference `r`, its paired noisy counterfactual `x_f`, clean
teacher `t`, and clean counterfactual teacher `t_f`, define

```text
d_s = F_theta(x_f, r) - F_theta(x, r)
d_t = t_f - t
e   = ||d_s - d_t||_2 / (||d_t||_2 + eps).
```

The transport term is

```text
0.25 mean(e) + CVaR_10%(max(e - 0.25, 0)).
```

The `0.25` boundary is tied to the frozen q05 retention gate: by the reverse triangle inequality,
`e <= 0.25` implies `||d_s||/||d_t||` lies in `[0.75, 1.25]`, so symmetric magnitude retention is
at least 0.75. The mean term prevents optimization from ignoring the non-tail population; the
CVaR term directly targets the registered worst-tail failure.

Normal-function drift is normalized to C1's own teacher residual:

```text
a = RMS(F_theta(x,r) - A(x,r)) / (RMS(A(x,r) - t) + eps).
```

V5 penalizes `10 * CVaR_10%(max(a - 0.10, 0))` and retains the clean-teacher MSE with unit weight.
The anchor is therefore inactive inside a 10% C1-relative function trust region and dominates a
large violation. This is a frozen constraint-derived choice, not a development-label sweep.

## Frozen training and comparison contract

- C0 and the three C1 artifacts remain byte-identical to the registered V3 run
  `server02_fp_naa_screening_20260817T145911Z`.
- Each V5 seed is initialized from the matching V3 C1 checkpoint. Candidate, seed, C1-compatible
  config signature, path, and SHA-256 are validated and recorded. There is no fallback to another
  seed or run.
- Only the three in-support pseudo-fault families enter the optimizer. Held-out friction bursts and
  development anomaly labels remain unavailable to training and checkpoint selection.
- Fine-tuning lasts 30 epochs at `5e-5`, with two warmup epochs, cosine decay, batch size 128,
  AdamW, and dropout disabled during the deterministic fine-tune. The final epoch is used; there
  is no label-based checkpoint selection.
- The BEATs cache, augmentation cache, RDP(8)/BEAM backend, three screening seeds, and all G2/G3
  gates are unchanged.
- V5 LOMO remains blocked unless every G2 core check passes.

## Closest prior art and claim boundary

- Jacobian matching relates input-noise distillation to matching local model response and supports
  paired tangent supervision, but does not study normal-only two-channel ASD or its fault-retention
  gates: <https://arxiv.org/abs/1803.00443>.
- Learning without Forgetting and later function-space regularization preserve an old function
  during adaptation; the frozen C1 anchor is therefore prior-motivated rather than a standalone
  novelty: <https://arxiv.org/abs/1606.09282> and
  <https://openreview.net/forum?id=SkMwpiR9Y7>.
- CVaR optimization targets an upper loss tail. V5 uses that established risk principle because
  the registered q05, not only the median, failed: <https://openreview.net/forum?id=5wZDv71acVp>.
- DCASE 2026 systems already use BEATs adaptation, reference denoising, dual-channel interaction,
  and score fusion. ACTT does not claim those components as new:
  <https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring-results>.

A defensible positive claim requires the frozen V5 mechanism to pass G2 core, LOMO, five-seed G3,
paired bootstrap, and reference-safety tests. If it fails G2, V5 is reported as another bounded
negative mechanism and is not rescued by changing its thresholds or inspecting the held-out fault
family for optimization.

## Execution audit: first V5 run is invalid

The first server execution, `server02_fp_naa_screening_20260817T234156Z` at source SHA
`bcabc4331e727557d679b3db4adb224a5f2368c2`, completed operationally but is not a scientific V5
result. All three C2 outputs exactly matched their corresponding C1 outputs, and the training log
reported that the learning-rate scheduler advanced before any optimizer step.

The cause was the exact-anchor expression `sqrt(mean((F_theta-A)^2))`: V5 initializes with
`F_theta == A`, where that expression has an undefined derivative. The resulting non-finite
gradient caused AMP to skip every optimizer update. The repair replaces it with the exact-zero,
finite-gradient smooth norm `sqrt(mean(x^2)+eps^2)-eps` and adds regressions requiring both finite
exact-anchor gradients and a finite C2 checkpoint that differs from its C1 initialization.

The first CUDA preflight at commit `9c25874` also showed that the default AMP loss scale can
overflow on the exact-anchor probe before backoff. That transient overflow is not equivalent to a
scale-independent NaN. The repaired optimizer therefore lets `GradScaler` skip and reduce scale,
but requires at least one finite optimizer update in every epoch. It records successful steps,
skipped steps, and final scale; a persistent non-finite gradient fails the epoch. The server probe
likewise permits at most 32 deterministic scale reductions, then requires changed finite model
parameters. Post-training attestation records the absolute and relative C2 displacement from the
registered C1 checkpoint.

No config value, data split, candidate seed, score backend, or frozen G2/G3 threshold was changed.
The unchanged preregistered V5 experiment must be rerun from fresh C2 checkpoints before ACTT can
be accepted or rejected.
