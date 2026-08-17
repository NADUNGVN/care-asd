# FP-NAA screening report

run_id=server02_fp_naa_screening_20260817T165229Z
source_git_sha=95c5434862d4644ee6e557e766cffb4fe0a832aa
conda_environment=care-asd-fp-naa
workers=12
task_status=0
gate_passed=false

## Gate

```json
{
  "checks": {
    "absolute_score": true,
    "core_screening": false,
    "gain_over_c0": true,
    "gain_over_c1": false,
    "heldout_retention_median": true,
    "heldout_retention_q05": false,
    "in_support_retention_median": false,
    "in_support_retention_q05": false,
    "lomo": null,
    "worst_machine_drop": true
  },
  "gate": "G2_screening_core",
  "machine_delta_c2_minus_c0": {
    "ToyCar": 0.0045997658140078945,
    "ToyCarEmu": 1.0731856183721078e-06,
    "bearingEmu": 0.042861654377222114,
    "fan": -0.007761383416034162,
    "gearboxEmu": 0.011990067134156623,
    "sliderEmu": 0.007690832129469194,
    "valveEmu": 0.06220143286819013
  },
  "note": "passed remains false until the preregistered LOMO gate is run",
  "passed": false,
  "retention": {
    "heldout_median_across_seeds": 0.9515587091445924,
    "heldout_worst_seed_q05": 0.5744868695735932,
    "in_support_median_across_seeds": 0.7731112241744995,
    "in_support_worst_seed_q05": 0.13401791378855704
  },
  "schema_version": 1,
  "scores": {
    "c0": 0.6197676455378447,
    "c1_mean": 0.63405655228682,
    "c2_mean": 0.6331053667290396,
    "c2_minus_c0": 0.013337721191194918,
    "c2_minus_c1": -0.0009511855577803985
  }
}

```

## Summary

```csv
seed,candidate,official_score,official_score_percent,in_support_retention_median,in_support_retention_q05,heldout_retention_median,heldout_retention_q05,trainable_parameters,score_path
13711,c1_mse,0.6344057638841025,63.44057638841025,0.7204845547676086,0.101599008962512,0.8826814889907837,0.5032059133052826,989696,seed13711/c1_mse/scores.csv
13711,c2_fault_preserving,0.635296971939492,63.5296971939492,0.7731121182441711,0.13401791378855704,0.9515516757965088,0.5744908511638641,989696,seed13711/c2_fault_preserving/scores.csv
42,c1_mse,0.6334202406721446,63.34202406721447,0.7210698127746582,0.1014079667627811,0.8769883513450623,0.4924710601568221,989696,seed42/c1_mse/scores.csv
42,c2_fault_preserving,0.6327098142580658,63.27098142580658,0.7731112241744995,0.1340220957994461,0.9515587091445924,0.5745032072067261,989696,seed42/c2_fault_preserving/scores.csv
2026,c1_mse,0.6343436523042127,63.43436523042128,0.720145583152771,0.1015052508562803,0.881390392780304,0.5057098925113678,989696,seed2026/c1_mse/scores.csv
2026,c2_fault_preserving,0.631309313989561,63.1309313989561,0.7731045782566071,0.13402089402079578,0.9515636563301086,0.5744868695735932,989696,seed2026/c2_fault_preserving/scores.csv

```

## Log tail

```text
{"event": "job_started", "run_id": "server02_fp_naa_screening_20260817T165229Z", "stage": "screening"}
{"event": "stage", "step": "assets"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "pytest", "tests/unit/test_fp_naa_adapter.py", "tests/unit/test_fp_naa_candidate.py", "-q"]}
................                                                         [100%]
16 passed in 5.89s
{"event": "stage", "step": "augmentation-cache"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "cache-augmentation", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--audio-root", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted", "--output-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counterfactual_fp32infer_v3", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v4.yaml", "--beats-source", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/unilm_833df7e7832e/beats", "--checkpoint", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/BEATs_iter3.pt", "--workers", "12", "--device", "cuda"]}
FP-NAA augmentation cache complete. clips=7000 heldout=643 
index=/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counter
factual_fp32infer_v3/index.parquet
{"event": "stage", "step": "screening"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "screen-dev", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--augmentation-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counterfactual_fp32infer_v3", "--c0-scores", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_c0_20260817T064722Z/c0_baseline/freq_rdp8_beam/scores.csv", "--output-dir", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_screening_20260817T165229Z/screening", "--checkpoint-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/checkpoints/server02_fp_naa_screening_20260817T165229Z", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v4.yaml", "--experiment-id", "server02_fp_naa_screening_20260817T165229Z", "--device", "cuda", "--preload-workers", "12"]}
FP-NAA C1/C2 screening complete. core_gate=failed 
summary=/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_screening
_20260817T165229Z/screening/screening_summary.csv
```
