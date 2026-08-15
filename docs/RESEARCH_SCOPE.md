# CARE-ASD research scope

This document supersedes conflicting v1 research choices in
`CARE_ASD_CODEX_IMPLEMENTATION_PLAN.md`.

## Central claim

CARE-ASD now studies **SAFE-REF: normal-only risk-controlled use of a far
microphone** for two-channel anomalous sound detection. The system either uses a
fixed, training-normal reference denoiser or abstains to the unchanged near-only
baseline. It does not learn a continuous fusion weight from development or
evaluation anomalies.

Phases 7--9 are retained as negative evidence: residual replacement and
capacity-matched reliability-gated fusion did not improve the aligned official
baseline. They motivate the safety question but are not presented as successful
methods.

## Required evidence

1. Calibrate the SAFE-REF policy only on controlled semi-synthetic cases with
   known machine, noise, and fault components.
2. Treat DCASE development labels as a fixed go/no-go audit, not as inputs to
   the policy thresholds; report 3-seed screening and 10-seed replication.
3. Freeze code, config, policy, seeds, and development evidence before scoring
   the five unseen DCASE evaluation machine types.
4. Produce evaluation scores without a ground-truth interface, seal their
   hashes, and only then call the separately pinned official evaluator.
5. Compare near-only, unconditional fixed RefSub, and SAFE-REF under the exact
   same official-compatible AE capacity and seed contract.

## Claims that are out of scope

- A general “noise removal” claim beyond the calibrated RefSub conditions.
- Distribution-free calibration coverage under unseen machine types.
- Hailo latency, energy, or accuracy claims before Hailo runtime and smoke test.
- Field-microphone deployment claims; current edge evaluation is deterministic
  replay of public DCASE stereo audio and synthetic causal streams.

## Publication structure

The DSP manuscript has one method contribution (SAFE-REF), one mechanistic
safety benchmark, and one frozen unseen-machine evaluation. Jetson AGX Xavier
and Xavier NX board-kit measurements are added only after the evaluation gate;
deployment results do not substitute for method novelty.
