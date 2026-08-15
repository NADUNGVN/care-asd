# Phase 10--12 SAFE-REF protocol

## Research question

Can normal training recordings identify when a synchronized far microphone is
safe enough to use as a noise reference, while falling back to the exact
near-only detector when reference use risks removing anomaly information?

SAFE-REF compares three capacity-matched systems:

| ID | Front end | Decision |
|---|---|---|
| B00 | Near channel | Always near |
| U00 | Fixed RefSub | Always RefSub |
| S00 | Near or fixed RefSub | Synthetic-calibrated normal-only policy |

RefSub uses `n_fft=1024`, `hop_length=512`, a 10th-percentile normal-training
noise floor, `alpha=1.5`, and floor `beta=0.10`. S00 makes one decision per
machine/section; it never interpolates scores or tunes on DCASE anomaly labels.

## Leakage boundary

`reference-safety eval` intentionally has no evaluator or ground-truth option.
It rejects evaluation manifests whose test condition is known, validates the
freeze hashes, writes the three score sets, and seals every official CSV in
`score_complete.json`. Only `reference-safety official-score` can receive the
pinned evaluator directory, and it first revalidates all sealed hashes.

## Server lifecycle

All commands below are one physical line. The start wrappers detach the job and
return the prompt immediately.

```bash
cd ~/Dung_TDTU/CARE_ASD && LD_LIBRARY_PATH="" git pull --ff-only origin main && bash scripts/server/start_phase10_reference_safety.sh
```

```bash
cd ~/Dung_TDTU/CARE_ASD && bash scripts/server/status_reference_safety.sh
```

After the Phase 10 report is reviewed, screening and replication are separate
jobs so a failed gate cannot consume ten-seed GPU time:

```bash
cd ~/Dung_TDTU/CARE_ASD && LD_LIBRARY_PATH="" git pull --ff-only origin main && bash scripts/server/start_phase11_reference_safety.sh screening
```

```bash
cd ~/Dung_TDTU/CARE_ASD && LD_LIBRARY_PATH="" git pull --ff-only origin main && bash scripts/server/start_phase11_reference_safety.sh replication
```

Evaluation is allowed only after the replication gate is committed and passed:

```bash
cd ~/Dung_TDTU/CARE_ASD && LD_LIBRARY_PATH="" git pull --ff-only origin main && bash scripts/server/start_phase12_reference_safety.sh
```

Cache work uses 12 processes with BLAS threads fixed to one. AE training uses
the GPU and the exact official 100-epoch, batch-256 contract. Low GPU utilization
during the cache stage is expected and is not a hang; `state.env` and the log
tail are the authoritative progress indicators.

Server wrappers use `uv run --no-sync`, so a research job never replaces the
verified GPU runtime while starting. Phase 11 and Phase 12 additionally require
exactly PyTorch `2.6.0+cu118`, CUDA runtime 11.8, and a visible GPU. If this
preflight fails, repair the runtime with
`bash scripts/server/setup_phase5_torch.sh` before continuing.
