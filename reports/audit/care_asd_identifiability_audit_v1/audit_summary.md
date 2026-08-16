# CARE-ASD identifiability audit synthesis

## Frozen decision

The AP-CARE method route is stopped. AP-G2--G5, unseen evaluation access, and
board-kit claims are prohibited. The retained publication route is an
identifiability/audit study.

## Aligned development evidence

| System | Mean AUC | Mean pAUC | Bootstrap pAUC delta | Decision |
|---|---:|---:|---:|---|
| B00 | 0.60813 | 0.55391 | +0.00000 | reference |
| B01 | 0.60251 | 0.53459 | -0.01740 | rejected_harmful_pauc |
| B02 | 0.60627 | 0.54789 | -0.00540 | rejected_no_improvement |

Phase 8 associated stronger residual-induced log-Mel displacement with a lower
B01-minus-B00 anomaly score (Spearman rho
-0.5524).

## Identifiability evidence

- SAFE-REF false-safe rate: 0.9285; risk
  Spearman rho: 0.0400.
- AP-CARE leakage Spearman rho: 0.5326;
  uncertainty Spearman rho: 0.4060.
- AP-CARE eligible median attenuation:
  -0.0397 dB; holdout cases at or
  above 1 dB: 0/
  256.

## Claim boundary

Normal-only contaminated-reference heuristics did not identify a reliable safe cancellation regime under the tested assumptions; this is an empirical audit, not a distribution-free impossibility theorem.

The CSV files and SVG figures in this directory are generated directly from the
frozen source artifacts listed and hashed in `run.json`.
