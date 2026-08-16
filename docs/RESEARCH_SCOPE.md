# CARE-ASD research scope

This document supersedes conflicting v1 research choices in
`CARE_ASD_CODEX_IMPLEMENTATION_PLAN.md`. On `research/ap-care-v2`, the detailed
execution authority is `docs/AP_CARE_V2_EXECUTION_SPEC.md`.

## Central claim

CARE-ASD now studies **AP-CARE: anomaly-preserving bounded reference
cancellation under contaminated acoustic references**. The candidate is a
deterministic, causal, normal-only controller that bounds time-frequency removal
while retaining the unchanged near signal. Its preservation claim is explicitly
conditional on acoustic path and normal-support assumptions.

Phases 7--10 are retained as negative evidence. Residual replacement and
capacity-matched reliability-gated fusion did not improve the aligned official
baseline, and corrected SAFE-REF synthetic calibration failed its identifiability
gate. They motivate bounded intervention but are not presented as successful
methods.

## Required evidence

1. Validate AP-CARE first on controlled synthetic cases with known machine,
   noise, path, leakage, and fault components under an independent holdout.
2. Demonstrate a non-trivial fault-preservation/noise-attenuation trade-off and
   validate each risk term against its known injected mechanism.
3. Treat DCASE development labels as a fixed go/no-go audit, not as inputs to
   controller thresholds; report 3-seed screening and 10-seed replication.
4. Freeze code, config, controller, seeds, and development evidence before scoring
   the five unseen DCASE evaluation machine types.
5. Produce evaluation scores without a ground-truth interface, seal their
   hashes, and only then call the separately pinned official evaluator.
6. Compare near-only, fixed/adaptive reference denoisers, and AP-CARE under the
   exact same official-compatible backend and seed contract.

## Claims that are out of scope

- A general “noise removal” claim beyond the calibrated RefSub conditions.
- A guarantee that normal-only statistics identify arbitrary unseen anomalies.
- Distribution-free calibration coverage under unseen machine types.
- Hailo latency, energy, or accuracy claims before Hailo runtime and smoke test.
- Field-microphone deployment claims; current edge evaluation is deterministic
  replay of public DCASE stereo audio and synthetic causal streams.

## Publication structure

The preferred DSP manuscript combines an identifiability boundary, one bounded
causal method contribution (AP-CARE), one mechanistic benchmark, and one frozen
unseen-machine evaluation. If the mechanism gate fails, the fallback is an
identifiability/audit paper and no expensive development replication is run.
Jetson AGX Xavier and Xavier NX board-kit measurements are added only after the
evaluation gate; deployment results do not substitute for method novelty.
