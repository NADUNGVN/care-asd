# Phase 8 CARE residual failure analysis

This is a post-hoc explanation of frozen B00/B01 development results; it is not tuning evidence.

## Global association

- Spearman rho between residual-minus-near mean log-Mel displacement and B01-B00 score delta: -0.5524 (two-sided p=1.154e-112).
- A more negative residual-minus-near value means CARE removed more log-Mel energy. It is a feature-displacement diagnostic, not a calibrated physical energy ratio.

## Strata with most negative mean score shift

| machine / section / domain / condition | clips | mean score delta | mean log-Mel displacement (dB) |
|---|---:|---:|---:|
| ToyCarEmu / section_00 / target / anomaly | 50 | -0.312510 | -1.0746 |
| valveEmu / section_00 / source / anomaly | 50 | -0.069393 | -1.4637 |
| ToyCar / section_00 / target / normal | 50 | -0.024776 | -0.4902 |
| ToyCar / section_00 / source / normal | 50 | -0.015093 | -0.2327 |
| ToyCar / section_00 / target / anomaly | 50 | -0.013510 | -0.4880 |

See `per_clip_analysis.csv`, `strata.csv`, and `correlations.csv` for machine-readable evidence.
