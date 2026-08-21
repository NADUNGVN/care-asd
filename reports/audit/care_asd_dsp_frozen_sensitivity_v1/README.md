# Frozen DSP sensitivity analysis

This directory contains a **derived post hoc analysis from frozen predictions/artifacts** at
Audit-A4 source snapshot `f1d5f7fadea74de6e9c7fdefcb172962b3298b63`. It performs no training, retuning, threshold change,
subset search, or evaluation-set access.

`official_metric_summary.csv` reports the exact DCASE 2026 Task 2 development harmonic score for
B00, B01, and B02. The historical paired AUC/pAUC deltas remain the frozen inferential estimands;
the harmonic score is secondary and descriptive.

The same deterministic calculation is stratified by the repository's dataset construction:
real synchronized machine types (ToyCar, fan) and emulated two-channel machine types (ToyCarEmu, bearingEmu, gearboxEmu, sliderEmu, valveEmu). With
only two real machine types and five emulated types, these strata are descriptive sensitivity
analyses, not preregistered subgroup inference.

`run.json` records the analysis label, controls, input hashes, code hashes, pairing checks, and
machine partition. `manifest.json` records input and generated-output hashes. The script refuses to
overwrite this directory.

Reproduce into a new path with:

```bash
uv run python scripts/run_dsp_frozen_sensitivity.py \
  --config configs/research/dsp_frozen_sensitivity_v1.yaml \
  --output-dir <new-output-directory>
```
