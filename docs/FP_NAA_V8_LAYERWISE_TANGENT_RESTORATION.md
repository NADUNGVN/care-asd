# FP-NAA v8 layerwise counterfactual tangent restoration

## Decision after the frozen V6 result

The valid V6 run `server02_fp_naa_frontend_probe_20260818T033118Z` completed all 7,000
normal-training probes from source commit `70d728275824f6f26a89c960a40ebfe72e9d123f`. No frozen BEATs tap
passed the preregistered 0.90 median and 0.75 fifth-percentile in-support retention thresholds:

| BEATs tap | retention median | retention q05 | direction median | transport error median |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8465 | 0.4094 | 0.6430 | 0.7969 |
| 4 | 0.7930 | 0.2766 | 0.7093 | 0.7173 |
| 8 | 0.7839 | 0.1756 | 0.7184 | 0.7073 |
| 12 | 0.7731 | 0.1340 | 0.7260 | 0.6992 |

V7 COGEB is therefore rejected without implementation, exactly as its preregistration requires.
Tap 0 cannot be selected manually and its gate cannot be relaxed. V6 also localizes the failure:
fault-delta magnitude is already distorted by the fixed log-filterbank/patch frontend and the lower
tail deteriorates through the Transformer. A final-token adapter cannot reconstruct information it
never observes reliably.

## Primary-source feasibility and novelty boundary

The strongest positive feasibility evidence is the DCASE 2026 winning noise-aware SSL system. Its
NA-BEATs frontend inserts trainable reference-conditioned noise-aware layers after every frozen
BEATs Transformer layer and trains them by clean-representation MSE distillation. The associated
paper reports a 70.24% official evaluation score for discriminatively fine-tuned NA-BEATs. Other
DCASE 2026 systems already use LoRA, intermediate-layer fusion, dual-channel gating, pseudo-labels,
and generated augmentation. Synthetic/pseudo anomalies and Jacobian matching are also established
prior art.

Consequently, V8 does **not** claim BEATs, layerwise cross-attention, LoRA, pseudo faults, teacher
distillation, or tangent/Jacobian matching individually. The testable contribution is narrower:

> In a capacity-matched layerwise noise-aware SSL encoder, normal-only counterfactual tangent
> transport with an explicit lower-tail constraint preserves mechanically plausible fault evidence
> that ordinary clean-representation MSE distillation erases.

This claim remains provisional until the capacity-matched comparator, held-out fault family, LOMO,
paired bootstrap, and reference-safety gates pass.

Primary sources used to freeze this boundary:

- [Anomalous Sound Detection Meets Noise-Aware SSL](https://arxiv.org/abs/2608.00447)
- [NABEATs: Noise-Aware Audio Representation Learning](https://arxiv.org/abs/2607.16688)
- [DCASE 2026 winning system report](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Fujimura_17_t2.pdf)
- [Official Microsoft BEATs implementation](https://github.com/microsoft/unilm/tree/833df7e7832e5064a281131ee64a481afa8e5b95/beats)
- [Knowledge Transfer with Jacobian Matching](https://proceedings.mlr.press/v80/srinivas18a.html)
- [Improving ASD via LoRA Fine-Tuning](https://arxiv.org/abs/2409.07016)
- [DCASE 2025 pseudo-outlier exposure report](https://dcase.community/documents/challenge2025/technical_reports/DCASE2025_Emon_6_t2.pdf)

## Frozen V8 architecture

The official BEATs Iteration-3 checkpoint and its patch frontend remain frozen. The near and far
tap-0 sequences pass through the same twelve frozen Transformer blocks. After each block, a
trainable bandwise noise-aware residual layer refines only the near sequence using the matching far
sequence. The far path is never updated by a noise-aware layer. The output shape and the frozen
RDP(8)/BEAM backend remain unchanged.

The noise-aware layer inventory is identical for both comparators: RMS/layer normalization,
band-aligned multi-head cross-attention, a gated feed-forward residual, and a zero-initialized final
projection. No development anomaly label selects insertion depth, width, checkpoint, epoch, or
fusion weight. In particular, this is not the rejected V7 frozen-tap bypass: V8 is a trainable
target-conditioned encoder whose sole purpose is to repair the V6-localized observability failure.

## Capacity-matched comparators and schedule

For each seed, one common zero-initialized model is trained for ten epochs with normal MSE only and
saved as the immutable branch point. Two trajectories then receive the same ten additional epochs:

- `L1_layerwise_mse`: continues normal clean-teacher MSE only;
- `L2_layerwise_fault_transport`: uses the same normal MSE plus counterfactual tangent transport,
  lower-tail relative-error penalty, and an anchor to the common branch-point normal function.

Both trajectories therefore have byte-identical architecture, initialization, data order through
the branch point, parameter count, epoch count, and optimizer-step count. Only the post-branch loss
differs. Final epochs are used; anomaly-label checkpoint selection and hyperparameter sweeps are
prohibited.

The waveform protocol and frozen families remain identical to FP-NAA v1--v6. Periodic resonance,
amplitude modulation, and frequency modulation are optimizer-visible. Friction burst is never
used by either optimizer and remains the out-of-family diagnostic.

## Bounded execution order

V8 uses two compute gates to avoid another open-ended phase:

1. **V8-M mechanism preflight:** seed 2608, a stable hash split of normal training clips, and the
   frozen schedule above. It reads no development anomaly label. It must demonstrate a real
   optimizer update, finite deterministic gradients, normal-function stability, and better
   validation-set in-support median and q05 retention than `L1`.
2. **G2 performance screening:** only after V8-M passes, train seeds `[13711, 42, 2026]` and score
   the untouched development test partition with the existing exact DCASE metric.

V8-M passes only if all checks hold on its stable validation split:

- a one-clip real-GPU zero-adapter path reproduces pinned BEATs within relative error 1e-5 and a
  subsequent optimizer probe creates a finite non-zero adapter update;
- `L2` in-support retention median >= 0.90;
- `L2` in-support retention q05 >= 0.60;
- `L2 - L1` retention median >= 0.05;
- `L2 - L1` retention q05 >= 0.10;
- held-out friction median >= 0.85 and q05 >= 0.60;
- median normal-function relative drift from the branch point <= 0.10;
- every stored tensor, gradient norm, and optimizer update is finite.

The q05 preflight threshold is deliberately weaker than the unchanged paper gate of 0.75. It only
authorizes the expensive three-seed run; it is not scientific success.

## Unchanged scientific gates

The G2 and G3 performance requirements in `FP_NAA_EXECUTION_SPEC.md` remain unchanged. For V8,
`C1` means `L1_layerwise_mse` and `C2` means `L2_layerwise_fault_transport`. A positive mechanism
claim additionally requires `L2 - L1 >= 0.005` in mean three-seed official score. No LOMO run is
authorized unless all G2 core, in-support, held-out, and worst-machine checks pass. No confirmatory,
reference-safety, paper-freeze, or Jetson measurement is authorized until its preceding gate passes.

If V8-M fails, layerwise tangent restoration is closed without reading development anomaly labels.
If V8-M passes but G2 fails, V8 is closed without LOMO. Neither failure permits a loss-weight,
fault-family, tap, layer-count, seed, or threshold sweep.
