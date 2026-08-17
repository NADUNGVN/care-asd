# FP-NAA v4: reference-only perturbation-equivariant adapter

## Status and preregistration boundary

This mechanism was frozen after the valid V3 screening result
`server02_fp_naa_screening_20260817T145911Z` and before any V4 result existed. V1--V3 remain
immutable negative ablations. V4 changes the causal architecture, not a loss weight, projection
threshold, seed, backend, or gate.

## Evidence that closes V3

V3 preserved the strong C1 score but did not preserve the registered perturbations. Its mean score
was 63.4189%, only +0.0132 percentage point over C1, while in-support median/q05 retention was
0.7208/0.1019 and held-out retention was 0.8814/0.4932. V3 therefore passed the absolute score,
C0-gain, and worst-machine checks, but failed C2-over-C1 and three retention checks. LOMO is
prohibited.

The result is mechanistically informative. V3 constrained a correction vector after a
target-conditioned attention block. For a perturbation `delta`, both the identity path and the
correction changed, so the constraint did not control the Jacobian of the whole adapter with
respect to the target.

## V4 structural hypothesis

Let `x` be the near-microphone BEATs token grid and `r` the far-reference grid. C1 remains the
target-conditioned adapter. V4 C2 instead has the form

```text
F_theta(x, r) = x + G_theta(r).
```

For a fixed reference and any representable target-token perturbation `delta`, the architecture
satisfies

```text
F_theta(x + delta, r) - F_theta(x, r) = delta
```

independently of the learned parameters. Equivalently, its Jacobian with respect to `x` is the
identity wherever the floating-point operation is defined. This blocks the adapter correction from
erasing a perturbation that appears only in the near input.

`G_theta` retains the exact C1 module inventory and capacity. Both query and key/value views are
computed from `r` using the existing two LayerNorm/projection branches, followed by the same
bandwise multi-head attention and fusion MLP. The correction is added to the untouched `x`.
Consequently C1 and C2 each have 989,696 trainable parameters; V4 does not obtain safety by
shrinking C2.

C2 is trained independently with the same clean-representation MSE used by C1. All pseudo-fault
loss weights are zero, and the implementation skips the second fault forward pass. Pseudo faults
remain blinded diagnostics only. This prevents the 84--85% gradient conflict observed in V2 and
isolates the architectural constraint as the C1/C2 difference.

The exact identity concerns perturbations at the adapter input. The registered retention metric
compares the noisy-input BEATs delta with a clean-teacher BEATs delta, which need not be identical.
Retention must therefore still pass empirically; the algebra is not used as a substitute for G2.

## Closest-method and novelty boundary

- Noise-aware SSL refines a target representation with target-as-query cross-attention over a far
  representation. It motivates the two-channel setting but does not impose an identity target
  Jacobian: <https://arxiv.org/abs/2608.00447>.
- Deep active noise control already predicts a cancelling signal solely from a reference signal.
  Reference-only cancellation is therefore prior art and is not a novelty claim:
  <https://www.isca-archive.org/interspeech_2020/zhang20i_interspeech.html>.
- Kim's DCASE system uses a far-channel minimum-statistics noise transfer and floored spectral
  subtraction to limit machine-signal removal. It operates before the encoder and does not provide
  token-space perturbation equivariance:
  <https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Kim_91_t2.pdf>.
- Interference-aware echo-cancellation work explicitly documents target deletion under overlapping
  reference interference, supporting the safety risk but using a modified waveform training target
  rather than a structural identity path:
  <https://www.isca-archive.org/interspeech_2024/khanagha24_interspeech.html>.

Accordingly, the defensible candidate contribution is narrow: a capacity-matched, reference-only
BEATs correction whose target perturbation equivariance is exact by construction, evaluated under
frozen DCASE RDP/BEAM performance, held-out perturbation, LOMO, bootstrap, and reference-leakage
gates. Neither reference cancellation, residual connections, nor equivariance in general is
claimed as new. A final novelty claim still requires the complete literature audit and all G2/G3
evidence.

## Frozen V4 protocol

The authoritative configuration is `configs/experiment/fp_naa_v4.yaml`.

- BEATs checkpoint/cache, augmentation cache, C0 scores, RDP(8)/BEAM backend, three screening
  seeds, optimizer, 60 epochs, and every G2/G3 threshold are unchanged.
- C1 is the original target-conditioned MSE adapter.
- C2 is independently trained reference-only equivariant MSE adapter.
- No auxiliary fault loss, output projection, threshold grid, or development-label selection is
  allowed.
- SERVER-02 must pass `target_perturbation_equivariance_probe` before screening.
- LOMO remains blocked unless every G2 core check passes.

V4 succeeds at screening only if it reaches at least 62.00%, improves over C0 by at least 1.00
point and over C1 by at least 0.50 point, passes all four retention thresholds, and has no machine
drop larger than 2.00 points. A positive method-paper claim still requires the subsequent LOMO and
G3 confirmatory gates.
