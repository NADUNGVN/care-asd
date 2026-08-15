# Phase 10 SAFE-REF report

run_id=server02_phase10_reference_safety_20260815T102535Z
cache=/home/ubuntu/Dung_TDTU/data/CARE_ASD/reference_safety_cache/dev/server02_phase10_reference_safety_20260815T102535Z
workers=12
task_status=2

## Gate

```json
{
  "calibration": {
    "accepted": 0,
    "cases": 2048,
    "coverage": 0.0,
    "false_safe_count": 0,
    "false_safe_rate": 0.0,
    "false_safe_upper_ci": 1.0,
    "policy_loss_q95": 0.0,
    "risk_spearman": 0.20660679036834212,
    "tail_loss_reduction": 1.0,
    "unconditional_loss_q95": 0.6447413652551617
  },
  "criteria": {
    "false_safe_max": 0.05,
    "false_safe_upper_ci_max": 0.1,
    "minimum_coverage": 0.2,
    "minimum_risk_spearman": 0.6,
    "minimum_tail_loss_reduction": 0.5
  },
  "holdout": {
    "accepted": 0,
    "cases": 2048,
    "coverage": 0.0,
    "false_safe_count": 0,
    "false_safe_rate": 0.0,
    "false_safe_upper_ci": 1.0,
    "policy_loss_q95": 0.0,
    "risk_spearman": 0.21085671719285318,
    "tail_loss_reduction": 1.0,
    "unconditional_loss_q95": 0.6627202014550306
  },
  "passed": false,
  "schema_version": 1
}

```

## Log tail

```text
Phase 10 SAFE-REF started at 2026-08-15T10:25:35Z
   Building care-asd @ file:///home/ubuntu/Dung_TDTU/CARE_ASD
Downloading nvidia-cuda-nvrtc (86.0MiB)
Downloading pillow (6.6MiB)
Downloading nvidia-cudnn-cu13 (349.2MiB)
Downloading nvidia-cublas (403.5MiB)
Downloading triton (188.5MiB)
Downloading torch (502.2MiB)
Downloading torchaudio (1.7MiB)
Downloading nvidia-cuda-cupti (10.2MiB)
Downloading nvidia-nvshmem-cu13 (57.6MiB)
Downloading nvidia-curand (56.8MiB)
Downloading nvidia-cusparselt-cu13 (162.3MiB)
Downloading cuda-bindings (6.4MiB)
Downloading nvidia-cuda-runtime (2.1MiB)
Downloading nvidia-nccl-cu13 (196.4MiB)
Downloading nvidia-cusparse (139.2MiB)
Downloading nvidia-cufile (1.2MiB)
Downloading sympy (6.0MiB)
Downloading nvidia-cufft (204.2MiB)
Downloading nvidia-cusolver (191.6MiB)
Downloading nvidia-nvjitlink (38.9MiB)
      Built care-asd @ file:///home/ubuntu/Dung_TDTU/CARE_ASD
 Downloaded nvidia-cufile
 Downloaded nvidia-cuda-runtime
 Downloaded torchaudio
 Downloaded pillow
 Downloaded cuda-bindings
 Downloaded nvidia-cuda-cupti
 Downloaded nvidia-nvjitlink
 Downloaded nvidia-curand
 Downloaded nvidia-nvshmem-cu13
 Downloaded sympy
 Downloaded nvidia-cuda-nvrtc
 Downloaded nvidia-cusparse
 Downloaded nvidia-cusparselt-cu13
 Downloaded nvidia-cusolver
 Downloaded nvidia-nccl-cu13
 Downloaded nvidia-cufft
 Downloaded triton
 Downloaded nvidia-cudnn-cu13
 Downloaded nvidia-cublas
 Downloaded torch
Uninstalled 5 packages in 9.17s
Installed 33 packages in 14.02s
SAFE-REF vector cache complete. clips=8400 
profiles=/home/ubuntu/Dung_TDTU/data/CARE_ASD/reference_safety_cache/dev/server0
2_phase10_reference_safety_20260815T102535Z/profiles.parquet
SAFE-REF simulation completed. gate=failed 
summary=reports/reference_safety/server02_phase10_reference_safety_20260815T1025
35Z/simulation/summary.csv

```
