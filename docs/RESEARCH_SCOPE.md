# CARE-ASD research scope

This document supersedes conflicting v1 research choices in
`CARE_ASD_CODEX_IMPLEMENTATION_PLAN.md`. On `research/ap-care-v2`, the detailed
historical AP-CARE execution authority is `docs/AP_CARE_V2_EXECUTION_SPEC.md`.
After the preregistered AP-G1 failure, the active publication authority is
`docs/IDENTIFIABILITY_AUDIT_PAPER.md`.

## Central claim

CARE-ASD now audits the **identifiability limits of normal-only
contaminated-reference processing for anomalous sound detection**. It asks
whether stereo reference interventions can be justified without anomaly labels
when the far microphone contains an unknown mixture of machine and environmental
energy.

Phases 7--10 and AP-G1 form a sequential evidence chain. Residual replacement
significantly reduced aligned pAUC, capacity-matched reliability-gated fusion did
not improve the reference, SAFE-REF failed to identify safe reference use, and
AP-CARE preserved injected faults but failed five of six mechanism gates because
it did not achieve useful noise attenuation or reliable leakage/uncertainty
tracking. None is presented as a successful method.

## Required evidence

1. Preserve the exact B00/B01/B02 aligned scores, bootstraps, and Phase 8
   diagnostic association without post-failure model selection.
2. Preserve the corrected SAFE-REF calibration/holdout and AP-G1
   calibration/holdout, including every seed and failed criterion.
3. Report the complete chain, including null and harmful results, rather than
   selecting only favorable machine types or metrics.
4. Separate empirical non-identifiability under the tested assumptions from a
   general impossibility theorem.
5. Generate every paper table and figure from frozen, hashed artifacts through
   one side-effect-free/immutable synthesis contract.

## Claims that are out of scope

- A successful AP-CARE method or anomaly-preservation/noise-removal claim.
- A universal theorem that reference cancellation is impossible.
- AP-G2/AP-G3 GPU results or unseen-machine AP-G4 results.
- Jetson AGX Xavier, Jetson Xavier NX, Hailo, field-microphone, latency, power,
  or deployment claims as substitutes for method evidence.

## Publication structure

The retained DSP manuscript is an identifiability/audit paper. It combines an
aligned negative-control sequence, a frozen post-hoc mechanism association, and
two known-component synthetic holdouts. No expensive development replication,
unseen evaluation, or board-kit benchmark is run after the AP-G1 stop rule.
