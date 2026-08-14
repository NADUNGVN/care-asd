# CARE-ASD

> Workspace path: `Teacher_Vu/CARE_ASD/` (one research track under the multi-paper Teacher_Vu workspace).

**Causal Acoustic-Path and Reliability-Calibrated Anomalous Sound Detection
for Unseen Machines at the Edge**

Research codebase for unsupervised anomalous sound detection on
[DCASE 2026 Challenge Task 2](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
stereo (near/far) machine audio, with:

1. **Causal acoustic-path estimation** between near and far microphones  
2. **Reliability-calibrated** anomaly scores and selective `abstain`  
3. **Unseen-machine / target-domain** adaptation from normal samples only  
4. **Selective edge inference** (after hardware profile is fixed)

Target journal: *Digital Signal Processing* (Elsevier).

> Implementation follows
> [`CARE_ASD_CODEX_IMPLEMENTATION_PLAN.md`](CARE_ASD_CODEX_IMPLEMENTATION_PLAN.md)
> as the binding technical specification.
>
> Server collaboration follows
> [`docs/COLLABORATION_PROTOCOL.md`](docs/COLLABORATION_PROTOCOL.md). Commands
> supplied for a server are deliberately one physical line per task.

## Status

| Phase | Description | Status |
|------:|-------------|--------|
| 0 | Repository bootstrap | **complete** |
| 1 | Dataset download & audit | **complete** |
| 2 | Official baseline reproduction | **complete** |
| 3 | Signal-processing baselines | **complete** |
| 4 | CARE acoustic-path front-end | **implemented; deterministic control complete** |
| 5 | MVP embedding & anomaly scoring | **complete; not retained as the primary result** |
| 6 | Official baseline alignment | **complete; passed** |
| 7 | Baseline-aligned CARE residual test | **implemented; awaiting server run** |
| 7 | Causal streaming | planned |
| 8 | Experiment orchestration | planned |
| 9 | Evaluation freeze | planned |
| 10 | Hardware deployment | blocked on hardware profile |

## Quick start

### Requirements

- Python **3.11+** (3.12 supported)
- [uv](https://github.com/astral-sh/uv) (recommended)

### Install

```bash
# Core + dev tools + PyTorch
uv sync --extra dev --extra torch
```

Or with pip:

```bash
pip install -e ".[dev,torch]"
```

### Verify environment

```bash
uv run care-asd --help
uv run care-asd env-report
uv run care-asd config-show --config configs/experiment/default.yaml
uv run pytest
uv run ruff check .
uv run mypy src
```

### Configuration

Configs are YAML + OmegaConf, validated with Pydantic:

```bash
# Show defaults and config hash
uv run care-asd config-show

# Override seed without editing files
uv run care-asd config-show --override experiment.seed=7 --hash

# Write a fresh default config
uv run care-asd config-init -o configs/experiment/my_run.yaml
```

### Dataset (Phase 1)

Audio is **not** committed. On a server, follow the collaboration protocol, then
run one command per task:

```bash
uv run care-asd data download --split dev --data-root /path/on/server/to/care-asd-data
uv run care-asd data extract --split dev --data-root /path/on/server/to/care-asd-data
uv run care-asd data manifest --split dev --data-root /path/on/server/to/care-asd-data
uv run care-asd data validate --split dev --data-root /path/on/server/to/care-asd-data
```

See [`data/README.md`](data/README.md) and [`docs/DATASET.md`](docs/DATASET.md).

### Official baseline (Phase 2)

The DCASE 2026 baseline is an external pinned reference; see
[`docs/OFFICIAL_BASELINE.md`](docs/OFFICIAL_BASELINE.md). Its official loader
asserts stereo input but extracts features from channel 0 (near) only; it runs
unchanged, then has its scores normalized into CARE-ASD's schema.

## Project layout

```text
src/care_asd/     # Library code (all logic lives here)
configs/          # YAML configs (data, features, model, scoring, ...)
scripts/          # Thin CLI wrappers for long jobs
tests/            # Unit / integration / regression + synthetic fixtures
docs/             # Dataset, leakage, experiment, hardware, results schema
experiments/      # Registry + freeze files
data/             # Manifests only in git; raw audio gitignored
outputs/          # Generated (gitignored)
reports/          # Human-readable phase reports
```

## Research rules (summary)

- Normal-only training and calibration — **no evaluation labels for tuning**
- Stereo truth: channel 0 near, channel 1 far; fail if not 2 channels
- Every pipeline supports `--config` and `--dry-run`
- Provenance: git commit, config hash, seed, manifest hash, package versions
- No overwriting prior experiment outputs
- See [`docs/LEAKAGE_POLICY.md`](docs/LEAKAGE_POLICY.md)

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite DCASE 2026 Task 2 and the
Zenodo datasets you use.

## License

MIT — see [`LICENSE`](LICENSE).
