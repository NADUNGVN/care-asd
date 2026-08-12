# Phase 3 DSP baseline contract

All DSP controls consume the same ordered stereo WAV input: channel 0 is near
and channel 1 is far. They return `FeatureBatch`, which separates spectral
`views` from diagnostics. A later shared encoder/scorer will consume these
views; this prevents DSP comparisons from changing the learned model.

| Name | Spectral view / policy |
|---|---|
| `near` | Original near STFT |
| `far` | Original far STFT |
| `average` | `(near + far) / 2` |
| `difference` | `near - far` |
| `early_concat` | Separate near and far views for feature-level concatenation |
| `spectral_subtraction` | Fixed `max(|near| - 0.5|far|, 0)` with near phase |
| `wiener` | `|near|² / (|near|² + |far|² + eps)` gain on near |
| `coherence_mask` | Static clip-level `1 - coherence` gain on near |
| `adaptive_filter` | Causal normalized-LMS far-path residual |
| `late_score_fusion` | Separate near/far views; arithmetic mean only after a shared scorer |

All controls use the same STFT parameters in `SignalConfig`. The coherence-mask
control is explicitly static; the adaptive filter is frame-causal. Every output
also reports a causal coherence diagnostic and the per-bin
`view_to_near_energy_ratio`; those diagnostics are used to audit
over-cancellation, not to tune against evaluation data.

The original near view is not replaced by the Safe CARE method in Phase 4;
Safe CARE will treat its bounded residual as an auxiliary view. These Phase 3
controls establish the comparisons needed to support that claim.

## Development benchmark

`care-asd dsp benchmark-dev` runs the selected controls over the identical
DCASE development manifest. For each machine type, it fits only normal
`dev_train` clips with one fixed log-spectral mean/std representation and a
standardized-distance scorer. This is a controlled diagnostic scorer, not the
proposed CARE-ASD neural encoder. Thus a change in AUC/pAUC can be attributed
to the DSP front-end rather than a changed learned architecture.

The immutable output directory contains one normalized score CSV and metrics
JSON per front-end, `summary.csv`, and `overcancellation.csv`. The summary
separates `real` machines (`ToyCar`, `fan`) from the five `*Emu` machine types;
that labeling rule is recorded explicitly and must be revisited if DCASE
metadata changes. The energy audit is a table rather than a hand-selected plot
so figures can be generated reproducibly after the run.
