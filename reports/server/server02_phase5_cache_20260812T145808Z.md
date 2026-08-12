# Phase 5 cache report

run_id=server02_phase5_cache_20260812T145808Z
cache_dir=/home/ubuntu/Dung_TDTU/data/CARE_ASD/neural_cache/server02_phase5_cache_20260812T145808Z
task_status=0

## Log

```text
Neural cache complete. clips=8400 
index=/home/ubuntu/Dung_TDTU/data/CARE_ASD/neural_cache/server02_phase5_cache_20
260812T145808Z/index.parquet
```

## Metadata

```json
{
  "base_channels": [
    "near",
    "far",
    "residual",
    "coherence",
    "log_ratio",
    "phase_sin",
    "phase_cos",
    "path_confidence"
  ],
  "clips": 8400,
  "features": {
    "coherence": true,
    "far_logmel": true,
    "fmax": null,
    "fmin": 0.0,
    "log_ratio": true,
    "n_mels": 128,
    "near_logmel": true,
    "phase_sin_cos": true,
    "residual_logmel": true
  },
  "frontend": {
    "gate": {
      "bias": 0.0,
      "bypass": false,
      "coherence_weight": 4.0,
      "max_value": 0.9,
      "min_value": 0.0,
      "mode": "semi_parametric",
      "snr_weight": -0.5
    },
    "name": "care",
    "residual": {
      "max_removed_energy_ratio": 0.8
    },
    "transfer": {
      "alpha": 0.95,
      "frequency_smoothing_bins": 1,
      "mode": "causal_ema",
      "reg_floor": 1e-05
    }
  },
  "manifest_sha256": "4cffc418e0a72a82e14b411125ee57ff36878f863fc90d577c276d52687f9882",
  "signal": {
    "center": false,
    "eps": 1e-08,
    "hop_length": 512,
    "n_fft": 1024,
    "win_length": 1024,
    "window": "hann"
  }
}

```

npz_files=8400
cache_size=9.5G	/home/ubuntu/Dung_TDTU/data/CARE_ASD/neural_cache/server02_phase5_cache_20260812T145808Z
