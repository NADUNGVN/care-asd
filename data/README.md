# Data directory

**Never commit raw, interim, or processed audio to Git.**

## Layout

| Path | Purpose | Git |
|------|---------|-----|
| `raw/` | Downloaded Zenodo archives and extracted WAVs | ignored |
| `interim/` | Intermediate extractions / temporary files | ignored |
| `processed/` | Cached features (optional) | ignored |
| `manifests/` | Parquet/CSV manifests (no audio) | tracked when non-empty |
| `checksums/` | Checksum sidecar files | tracked when non-empty |

## DCASE 2026 Task 2 sources

| Split | Zenodo record | Notes |
|-------|---------------|--------|
| Development | [19336329](https://zenodo.org/records/19336329) | Primary for Phases 1–8 |
| Additional training | [20151556](https://zenodo.org/records/20151556) | Unseen machines, normal only |
| Evaluation | [20437238](https://zenodo.org/records/20437238) | Requires policy flag; freeze first |

## Conventions

- Channel 0 = **near** microphone  
- Channel 1 = **far** microphone  
- Pipeline must assert `waveform.shape[0] == 2` for DCASE 2026 WAVs  

## Download (Phase 1)

```bash
care-asd data download --split dev --data-root /path/on/server/to/care-asd-data
care-asd data extract --split dev --data-root /path/on/server/to/care-asd-data
care-asd data manifest --split dev --data-root /path/on/server/to/care-asd-data
care-asd data validate --split dev --data-root /path/on/server/to/care-asd-data
```

Each download is verified against the checksum published in the official Zenodo
record. Extraction and manifest creation never overwrite an existing differing
file; retain the generated checksum sidecar and manifest with the experiment
record. Omit `--data-root` to use `data.root` from the selected config.

The evaluation split additionally requires `--accept-eval-policy`; do not pull
or inspect it before the experimental freeze policy permits doing so.

## License and citation

Follow the licenses and citation requirements on the Zenodo records and the
[DCASE 2026 Task 2 page](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring).
