# Official DCASE 2026 baseline reproduction

## Reproduced reference

| Item | Value |
|---|---|
| Baseline repository | `nttcslab/dcase2023_task2_baseline_ae` |
| Baseline commit | `f44242ec1f78f6cc34f53f43fb88be1ce5d13d47` |
| Evaluator repository | `nttcslab/dcase2026_task2_evaluator` |
| Evaluator commit | `f6a94a2b5e614a9626c9d1ccff6df0705e6aaa75` |
| Dataset | DCASE 2026 Task 2 development split |
| Manifest SHA-256 | `4cffc418e0a72a82e14b411125ee57ff36878f863fc90d577c276d52687f9882` |
| Seed | `13711` |
| Runtime | Python 3.11.15, PyTorch 2.6.0+cu118, CUDA on Quadro RTX 8000 |
| Official feature input | channel 0 (near); the official loader asserts stereo but returns `y[0]` |

## Method preserved from the official code

Each machine type has a separate autoencoder trained on its 1,000 normal
development clips. The feature is a 640-dimensional vector formed by stacking
five 128-bin log-Mel frames (`n_fft=1024`, `hop_length=512`). The encoder is
`640 → 128 → 128 → 128 → 128 → 8`; the decoder mirrors it. Hidden layers use
BatchNorm and ReLU. Training uses MSE, Adam (`lr=0.001`), batch size 256, and
100 epochs. We made no algorithmic changes.

## Score artifacts verified

Both normalized files contain exactly 1,400 unique development test clips:
350 clips for each source/target × normal/anomaly group.

| Mode | Score SHA-256 | Mean AUC (all) | Harmonic mean AUC (all) | Mean pAUC@0.1 (all) |
|---|---|---:|---:|---:|
| MSE | `8bf7f53adc1a21e6df025e08bc006cda5aa5f2610036df42a02bf65bdc012e78` | 0.609729 | 0.604764 | 0.542331 |
| Selective Mahalanobis | `a4f29afd23f29418816315cf803cc7b8458f966a458aaabc11374377af03aacb` | 0.604471 | 0.601027 | 0.536090 |

Full per-machine metrics and raw normalized scores are committed under
[`reports/baseline/`](baseline/). Development pAUC values were recomputed from
the raw scores with the same `max_fpr=0.1` definition used by the official code.

## Reproduction note

The initial MSE test process wrote all seven score and decision CSV files, then
failed only while rendering a diagnostic boxplot because Matplotlib 3.11 removed
the `labels` keyword used by the pinned official plotting helper. The score CSVs
were already written and were normalized successfully. The environment was then
pinned to the baseline-documented `matplotlib==3.10.1`; Selective Mahalanobis
test-only scoring completed successfully from the existing checkpoints. No model
was retrained during recovery.

These are development results, not a claim of challenge/evaluation performance.
