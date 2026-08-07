# Data-leakage policy

This project treats leakage prevention as a first-class acceptance criterion.

## Forbidden

1. Using evaluation ground-truth for training, calibration, threshold selection,
   architecture selection, or hyperparameter tuning.
2. Machine-specific hyperparameter edits based on evaluation test outcomes.
3. Fitting scorers or calibrators on anomalous development/evaluation labels
   when claiming normal-only methods.
4. Peeking at evaluation metrics to choose which freeze config to submit, then
   presenting that as a single pre-registered run.

## Allowed on development

- Feature design and ablations
- Architecture and HP search
- Leave-one-machine-type-out validation
- Calibration method comparison
- Streaming synthetic stress suites (not for official task accuracy claims)

## Evaluation freeze

- Choose one config, one calibration procedure, one aggregation, locked seeds.
- Write freeze YAML with config hash, manifest hash, git commit, date.
- Any post-evaluation change requires a **new freeze ID** and must be labeled
  as post-evaluation analysis.

## Automated checks (target)

- Tests that calibrators reject anomaly-labeled scores.
- Evaluation CLI requires `--freeze-file`.
- Config mismatch between freeze file and runtime config fails hard.
