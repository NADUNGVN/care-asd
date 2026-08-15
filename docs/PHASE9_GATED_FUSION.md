# Phase 9: preregistered near-primary gated fusion

B02 tests a new hypothesis after B01 was rejected: the original near signal is
the reconstruction target and primary branch; CARE residual is auxiliary only.

- Near and residual each project from 640 to 64. Their concatenation has 128
  dimensions, matching B00's first 640-to-128 projection exactly in parameter
  count. The remaining encoder/decoder is unchanged.
- The residual branch is multiplied by a frozen per-clip reliability value: the
  mean `path_confidence` already cached in Phase 5. It uses no condition/domain
  labels and is bounded to `[0,1]`.
- B00 comparison, 100 epochs, batch 256, seed 13711, normal-only train rows,
  and per-WAV mean MSE all remain locked.
- Primary criterion: paired B02-B00 development AUC bootstrap CI must be wholly
  positive. pAUC must not have a wholly negative CI. Otherwise B02 is rejected.

This is one prospective test. The Phase 8 output is explanatory only and does
not tune its gate or architecture.

The server wrapper defaults to 12 cache workers on the 28-logical-CPU server,
leaving at least 16 logical CPUs free. Set `CARE_ASD_RELIABILITY_WORKERS` only
when a different resource reservation is explicitly required.
