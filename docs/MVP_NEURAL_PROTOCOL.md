# Phase 5 MVP neural protocol

This protocol replaces broad Phase 5–10 development work until the Safe CARE
method has a go/no-go result on DCASE development data.

## Fixed screening comparison

All runs reconstruct normalized **near log-Mel** with the same compact
depthwise-separable autoencoder and clip-level reconstruction MSE. The only
variable is the input view set:

| ID | Input channels |
|---|---|
| `a00_near` | near |
| `a01_near_far` | near, far |
| `a02_care_multiview` | near, far, Safe CARE residual, coherence, log-ratio, phase sin/cos, path confidence |

The feature cache is built once from public stereo audio. It contains no fitted
normalization values and no learned parameters. During GPU screening, all cache
maps are preloaded once into server RAM (16 CPU workers on SERVER-02) and reused by the
three sequential ablations; this prevents repeated NPZ decompression and keeps
the single GPU supplied with batches. For each machine type, feature
normalization and model fitting use only `dev_train` normal clips, including
the allowed ten target-normal clips. Development test conditions are used only
by the deterministic post-hoc metric function.

## Runtime sequence

1. Run a ten-clip cache smoke test and a one-epoch CPU/GPU smoke run.
2. Build the full immutable cache once with CPU workers.
3. Screen A00, A01, A02 with seed `13711`, 30 epochs, batch size 128, and
   mixed precision on GPU. The views run sequentially because concurrent
   training would contend for the same GPU and make the comparison slower.
4. Select the two strongest configurations from their development summaries;
   rerun only those configurations with seeds `13711`, `42`, and `2026`.
5. Run paired stratified bootstrap on the final score files. The final model
   must improve AUC with a positive 95% delta CI over A00 and must not reduce
   pAUC by more than 0.005.

Checkpoints remain local under `outputs/`; report directories contain only
scores, metrics, model cards, resolved config, and hashes suitable for Git.
Hardware, calibration, and streaming stay outside the critical path until this
decision passes.
