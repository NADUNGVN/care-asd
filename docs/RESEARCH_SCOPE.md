# CARE-ASD research scope

This document supersedes conflicting v1 research choices in
`CARE_ASD_CODEX_IMPLEMENTATION_PLAN.md`.

## Central claim

CARE-ASD studies a **safe causal acoustic-path residual view** for normal-only,
two-channel anomalous sound detection. The near-channel spectral view is always
preserved; the residual is auxiliary and bounded by a per-frame removed-energy
budget. A path-confidence diagnostic makes conservative downstream abstention
possible but does not provide a distribution-free guarantee under unseen-domain
shift.

## Required evidence

1. Select front-end, encoder, scorer, calibration, and routing policy only on
   DCASE development data with multi-seed/LOMO analysis.
2. Run a frozen configuration once on the all-real evaluation dataset; every
   post-freeze change receives a new freeze ID.
3. Use controlled synthetic stereo simulation to quantify path estimation,
   energy removal, and retention of known fault-like components.
4. Compare against near-only, channel fusion, fixed/adaptive subtraction,
   coherence gating, and documented DCASE 2026 residual/fusion methods where
   independently reproducible.

## Claims that are out of scope

- “Noise removal” or anomaly preservation without controlled evidence.
- Distribution-free calibration coverage under unseen machine types.
- Hailo latency, energy, or accuracy claims before Hailo runtime and smoke test.
- Field-microphone deployment claims; current edge evaluation is deterministic
  replay of public DCASE stereo audio and synthetic causal streams.

## Publication structure

The DSP manuscript has one method contribution (Safe CARE), one reliability
audit, and one deployment study. Edge results demonstrate reproducible
trade-offs and do not substitute for method novelty.
