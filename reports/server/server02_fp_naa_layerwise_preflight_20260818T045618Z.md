# FP-NAA layerwise-preflight report

run_id=server02_fp_naa_layerwise_preflight_20260818T045618Z
source_git_sha=8ae0ebc147b383dabe17bb00f4b4277c7484f4ca
conda_environment=care-asd-fp-naa
workers=12
task_status=1
gate_passed=false

## Error

```json
{
  "error": {
    "code": "EXTERNAL_COMMAND_FAILED",
    "message": "An experiment subprocess failed",
    "details": {
      "command": [
        "/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python",
        "-m",
        "care_asd.cli",
        "fp-naa",
        "layerwise-preflight-dev",
        "--base-cache-dir",
        "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2",
        "--audio-root",
        "/home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted",
        "--cache-dir",
        "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/layerwise_preflight_cache/v8_seed2608",
        "--output-dir",
        "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_layerwise_preflight_20260818T045618Z/layerwise_preflight",
        "--checkpoint-dir",
        "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/checkpoints/server02_fp_naa_layerwise_preflight_20260818T045618Z",
        "--config",
        "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v8.yaml",
        "--beats-source",
        "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/unilm_833df7e7832e/beats",
        "--checkpoint",
        "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/BEATs_iter3.pt",
        "--workers",
        "12",
        "--device",
        "cuda"
      ],
      "returncode": 1
    }
  }
}
```

## Log tail

```text
{"event": "job_started", "run_id": "server02_fp_naa_layerwise_preflight_20260818T045618Z", "stage": "layerwise-preflight"}
{"event": "stage", "step": "assets"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "pytest", "tests/unit/test_layerwise_noise_aware.py", "tests/unit/test_fp_naa_layerwise_preflight.py", "tests/unit/test_fp_naa_config.py", "-q"]}
.............                                                            [100%]
13 passed in 4.05s
{"event": "stage", "step": "layerwise-preflight"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "layerwise-preflight-dev", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--audio-root", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted", "--cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/layerwise_preflight_cache/v8_seed2608", "--output-dir", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_layerwise_preflight_20260818T045618Z/layerwise_preflight", "--checkpoint-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/checkpoints/server02_fp_naa_layerwise_preflight_20260818T045618Z", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v8.yaml", "--beats-source", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/unilm_833df7e7832e/beats", "--checkpoint", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/BEATs_iter3.pt", "--workers", "12", "--device", "cuda"]}
/home/ubuntu/miniconda3/envs/care-asd-fp-naa/lib/python3.11/site-packages/torch/nn/utils/weight_norm.py:143: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
V8 cache ready. train=1024 validation=512 heldout=256
FP-NAA V8 layerwise preflight failed: Layerwise deterministic execution requires
BEATs layerdrop=0
{"error": {"code": "EXTERNAL_COMMAND_FAILED", "message": "An experiment subprocess failed", "details": {"command": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "layerwise-preflight-dev", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--audio-root", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted", "--cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/layerwise_preflight_cache/v8_seed2608", "--output-dir", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_layerwise_preflight_20260818T045618Z/layerwise_preflight", "--checkpoint-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/checkpoints/server02_fp_naa_layerwise_preflight_20260818T045618Z", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v8.yaml", "--beats-source", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/unilm_833df7e7832e/beats", "--checkpoint", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/BEATs_iter3.pt", "--workers", "12", "--device", "cuda"], "returncode": 1}}}
```
