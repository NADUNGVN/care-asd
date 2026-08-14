# Phase 7: baseline-aligned CARE residual test

Phase 6 reproduced the external official MSE baseline within its predeclared
alignment tolerance. Phase 7 therefore makes one controlled B01 comparison:

| Arm | Input waveform | Feature/model/training/scoring |
|---|---|---|
| B00 | Channel-0 near microphone | Locked official Phase 6 stack |
| B01 | Bounded causal CARE residual | The same locked Phase 6 stack |

## Locked comparison

- B01 does not use far-channel concatenation, diagnostics, score fusion, target
  labels, hyperparameter tuning, or a larger neural architecture.
- CARE operates causally over stereo STFT frames. Its residual STFT is converted
  back to a waveform using normalized overlap-add; unsupported window endpoints
  retain the original near sample. The resulting waveform alone is passed to
  the exact Phase 6 `librosa` centred log-Mel and five-frame vector stack.
- Both arms use the `640 → 128 → 128 → 128 → 128 → 8 → 128 → 128 → 128 → 128 → 640`
  autoencoder, Adam (`1e-3`), batch size 256, 100 epochs, seed 13711, normal-only
  training, and per-WAV mean reconstruction MSE.
- The primary outcome is mean development AUC. Mean pAUC at max FPR 0.1 is
  secondary. A paired, stratified 5,000-iteration bootstrap compares the
  committed B01 scores with the fixed Phase 6 B00 scores.

## Decision rule

B01 is evidence that CARE has useful signal only if its paired AUC confidence
interval excludes zero in the positive direction and no pre-specified machine
type has a material safety regression. Otherwise B01 is retained as a negative
control and no multi-view/capacity-changing follow-up is interpreted as a
replacement for this test.

The server wrapper writes every B01 report, score file, metrics file, bootstrap
result, log, cache provenance, and commit before pushing. It uses at most 16 CPU
workers while constructing the immutable cache, then runs the fixed GPU AE.
