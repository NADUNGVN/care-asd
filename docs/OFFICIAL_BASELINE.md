# Official DCASE 2026 baseline reproduction

CARE-ASD Phase 2 reproduces the official baseline as an external, immutable
reference. It never copies or modifies official source code.

| Reference | Repository | Pinned commit |
|---|---|---|
| AE baseline | `nttcslab/dcase2023_task2_baseline_ae` | `f44242ec1f78f6cc34f53f43fb88be1ce5d13d47` |
| Evaluator | `nttcslab/dcase2026_task2_evaluator` | `f6a94a2b5e614a9626c9d1ccff6df0705e6aaa75` |

The pinned DCASE 2026 baseline invokes its `mono=False` path and asserts a
multi-channel WAV, but its official `file_load` function returns `y[0]` before
feature extraction. Thus the reproduced AE consumes the left/near channel only;
the far channel is present only to satisfy the stereo input assertion. This is
the correct near-only control for CARE-ASD comparisons.

## Reproduction contract

1. `baseline checkout` clones the exact commits under an ignored `external/`
   directory and rejects a checkout at a different revision.
2. `baseline stage-dev` makes directory symlinks only. It does not duplicate,
   rename, alter, or commit WAV files; it refuses a non-normal training split.
3. `baseline run-dev` invokes unmodified official shell scripts with an
   explicitly supplied Python executable from the isolated official environment.
4. `baseline normalize` maps every official two-column anomaly-score CSV to the
   CARE-ASD score schema and fails if a development test clip is missing or
   duplicated.
5. `baseline metrics` recomputes development AUC/pAUC from normalized scores.
   The separately pinned DCASE evaluator is reserved for post-freeze evaluation
   data because that tool carries evaluation ground truth.

Each score mode is recorded independently:

- `official_dcase2026_ae_mse`
- `official_dcase2026_ae_mahala`

Do not tune official parameters, seed, data, or score aggregation to improve
the reproduced result. Any CARE-ASD comparison must retain the raw official log,
the normalized score file, its metrics JSON, the baseline commit, and the CARE
ASD manifest hash.
