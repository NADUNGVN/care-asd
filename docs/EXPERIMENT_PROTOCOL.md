# Experiment protocol

## Priorities

1. Correctness and data-leakage prevention.
2. Reproducibility and provenance.
3. Modular DSP and scoring interfaces.
4. Tests and diagnostics.
5. Hardware optimization only after the CPU reference is validated.

## Every experiment must record

- Git commit SHA
- Config hash
- Random seed
- Dataset manifest hash
- Package versions
- Device / environment report

## Rules

- Do not use evaluation labels for train, calibration, thresholding, architecture choice, or hyperparameter tuning.
- Do not overwrite previous outputs; write new experiment IDs.
- Recompute metrics from raw score files (do not hand-edit tables).
- One shared encoder/scorer config when comparing front-ends.
- Prefer multi-seed reporting; never pick the best seed as the main result.

## Development vs evaluation

| Split | Allowed uses |
|-------|----------------|
| Development | Feature design, architecture, HPs, ablation, LOMO, calibration method comparison |
| Additional training | Normal training, source/target adaptation, prototypes/memory banks |
| Evaluation | Only after freeze of feature/model/scoring/calibration/aggregation/threshold policy |

## Freeze file

Official evaluation requires a freeze YAML (see plan §11.4). Commands that
target evaluation must refuse to run without a valid freeze file.
