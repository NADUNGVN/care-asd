# FP-NAA screening report

run_id=server02_fp_naa_screening_20260817T083029Z
source_git_sha=4ebfdb6c61eee226efdd7dd5de559572400ca776
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
    "gain_over_c0": false,
    "gain_over_c1": false,
    "heldout_retention_median": true,
    "heldout_retention_q05": true,
    "in_support_retention_median": true,
    "in_support_retention_q05": false,
    "lomo": null,
    "worst_machine_drop": true
  },
  "gate": "G2_screening_core",
  "machine_delta_c2_minus_c0": {
    "ToyCar": -0.0007023036046286402,
    "ToyCarEmu": -0.00417671120633567,
    "bearingEmu": 0.009982454085115666,
    "fan": 0.004586138776576543,
    "gearboxEmu": 0.01245845457931205,
    "sliderEmu": -0.007280378291648737,
    "valveEmu": 0.011185514624647452
  },
  "note": "passed remains false until the preregistered LOMO gate is run",
  "passed": false,
  "retention": {
    "heldout_median_across_seeds": 0.9681347012519836,
    "heldout_worst_seed_q05": 0.729199868440628,
    "in_support_median_across_seeds": 0.9542049467563629,
    "in_support_worst_seed_q05": 0.4654216602444651
  },
  "schema_version": 1,
  "scores": {
    "c0": 0.6197676455378447,
    "c1_mean": 0.63405655228682,
    "c2_mean": 0.623062129801012,
    "c2_minus_c0": 0.003294484263167363,
    "c2_minus_c1": -0.010994422485807953
  }
}

```

## Summary

```csv
seed,candidate,official_score,official_score_percent,in_support_retention_median,in_support_retention_q05,heldout_retention_median,heldout_retention_q05,trainable_parameters,score_path
13711,c1_mse,0.6344057638841025,63.44057638841025,0.7204845547676086,0.101599008962512,0.8826814889907837,0.5032059133052826,989696,seed13711/c1_mse/scores.csv
13711,c2_fault_preserving,0.6227509781223945,62.27509781223945,0.9539027810096741,0.4654216602444651,0.9694475531578064,0.7295292019844055,989696,seed13711/c2_fault_preserving/scores.csv
42,c1_mse,0.6334202406721446,63.34202406721447,0.7210698127746582,0.1014079667627811,0.8769883513450623,0.4924710601568221,989696,seed42/c1_mse/scores.csv
42,c2_fault_preserving,0.6259582718660417,62.59582718660417,0.9544197916984558,0.4659086748957634,0.9681347012519836,0.7333284497261048,989696,seed42/c2_fault_preserving/scores.csv
2026,c1_mse,0.6343436523042127,63.43436523042128,0.720145583152771,0.1015052508562803,0.881390392780304,0.5057098925113678,989696,seed2026/c1_mse/scores.csv
2026,c2_fault_preserving,0.6204771394145997,62.04771394145997,0.9542049467563629,0.46709434986114495,0.9674364924430848,0.729199868440628,989696,seed2026/c2_fault_preserving/scores.csv

```

## Log tail

```text
{"event": "job_started", "run_id": "server02_fp_naa_screening_20260817T083029Z", "stage": "screening"}
{"event": "stage", "step": "assets"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "pytest", "tests/unit/test_fp_naa_adapter.py", "tests/unit/test_fp_naa_candidate.py", "-q"]}
.....                                                                    [100%]
5 passed in 5.26s
{"event": "stage", "step": "augmentation-cache"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "cache-augmentation", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--audio-root", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted", "--output-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counterfactual_fp32infer_v3", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v1.yaml", "--beats-source", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/unilm_833df7e7832e/beats", "--checkpoint", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/BEATs_iter3.pt", "--workers", "12", "--device", "cuda"]}
FP-NAA augmentation cache complete. clips=7000 heldout=643 
index=/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counter
factual_fp32infer_v3/index.parquet
{"event": "stage", "step": "screening"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "screen-dev", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--augmentation-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counterfactual_fp32infer_v3", "--c0-scores", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_c0_20260817T064722Z/c0_baseline/freq_rdp8_beam/scores.csv", "--output-dir", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_screening_20260817T083029Z/screening", "--checkpoint-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/checkpoints/server02_fp_naa_screening_20260817T083029Z", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v1.yaml", "--experiment-id", "server02_fp_naa_screening_20260817T083029Z", "--device", "cuda", "--preload-workers", "12"]}
FP-NAA C1/C2 screening complete. core_gate=failed 
summary=/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_screening
_20260817T083029Z/screening/screening_summary.csv
```
