# Results schema

## Score file (CSV / Parquet)

| Column | Description |
|--------|-------------|
| `file_id` | Unique clip identifier |
| `machine_type` | Machine type string |
| `section` | Section ID if present |
| `domain` | `source` / `target` / `unknown` |
| `condition` | `normal` / `anomaly` / `unknown` (unknown on eval) |
| `anomaly_score` | Higher = more anomalous (unless documented otherwise) |
| `model_id` | Model / scorer identifier |
| `experiment_id` | Registry experiment ID |

Optional columns (when available):

- `p_value`
- `path_confidence`
- `decision` (`NORMAL` / `ANOMALOUS` / `ABSTAIN`)
- `route_to_expert`

## Metrics JSON

Produced from raw score files only. Typical keys:

- Official: AUC, pAUC, precision, recall, F1 (per machine / aggregate)
- Calibration: risk-coverage, AURC, abstention rate, FPR
- Streaming: RTF, p50/p95 latency, false alarms/hour

## Experiment registry

`experiments/registry.csv` columns:

```text
experiment_id,date,git_commit,config_path,config_hash,manifest_hash,seed,status,notes
```

## Provenance

Each run should write:

- Environment report JSON
- Resolved config YAML copy
- Config hash
- Manifest hash
- Git commit (and dirty flag)
