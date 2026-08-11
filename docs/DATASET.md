# Dataset protocol

## Primary corpus

**DCASE 2026 Challenge Task 2** — Noise-aware Unsupervised Anomalous Sound
Detection for Machine Condition Monitoring.

| Resource | Location |
|----------|----------|
| Task page | https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring |
| Development | Zenodo `19336329` |
| Additional training | Zenodo `20151556` |
| Evaluation | Zenodo `20437238` |
| Official baseline | https://github.com/nttcslab/dcase2023_task2_baseline_ae |
| Official evaluator | https://github.com/nttcslab/dcase2026_task2_evaluator |

## Audio layout

- Stereo WAV clips (10 or 12 seconds).
- Channel 0 = near microphone.
- Channel 1 = far microphone.
- Development: 7 machine types; mixture of emulated acoustic paths and real dual-mic recordings.
- Training is **normal-only**.

## Source of truth for channel count

1. Metadata read from the WAV file itself.
2. Official task page.
3. Official baseline repository.

Zenodo text may still say “single-channel” in places; runtime code must assert:

```python
assert waveform.shape[0] == 2
```

## Manifest schema

See implementation plan §12. Required roles:

- `channel_0_role = near`
- `channel_1_role = far`
- `condition ∈ {normal, anomaly, unknown}`
- `domain ∈ {source, target, unknown}`
- `dataset_split ∈ {dev_train, dev_test, add_train, eval_test}`

The manifest contains both a stable `file_id` and a portable `relative_path`.
`relative_path` is POSIX-formatted relative to
`raw/dcase2026/<split>/extracted/`; consumers resolve it against their own
configured data root. Absolute server paths and parent traversal are forbidden
so a manifest can move safely between SERVER-02 and edge devices.

## Self-collected data

Not required for version 1. If added later, document license, mic geometry,
and sample rate here.
