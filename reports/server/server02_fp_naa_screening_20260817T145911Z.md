# FP-NAA screening report

run_id=server02_fp_naa_screening_20260817T145911Z
source_git_sha=71710304cbc134691b2145452abbc3009197b329
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
    "ToyCar": -0.0002811361575070226,
    "ToyCarEmu": 0.007308168467013321,
    "bearingEmu": 0.022137586014403188,
    "fan": 0.0004610469511464599,
    "gearboxEmu": 0.024633322202730623,
    "sliderEmu": 0.0010146988299356563,
    "valveEmu": 0.07458044363617611
  },
  "note": "passed remains false until the preregistered LOMO gate is run",
  "passed": false,
  "retention": {
    "heldout_median_across_seeds": 0.881390392780304,
    "heldout_worst_seed_q05": 0.49315960705280304,
    "in_support_median_across_seeds": 0.7207969427108765,
    "in_support_worst_seed_q05": 0.10194057375192637
  },
  "schema_version": 1,
  "scores": {
    "c0": 0.6197676455378447,
    "c1_mean": 0.63405655228682,
    "c2_mean": 0.6341886673898856,
    "c2_minus_c0": 0.01442102185204086,
    "c2_minus_c1": 0.0001321151030655443
  }
}

```

## Summary

```csv
seed,candidate,official_score,official_score_percent,in_support_retention_median,in_support_retention_q05,heldout_retention_median,heldout_retention_q05,trainable_parameters,score_path
13711,c1_mse,0.6344057638841025,63.44057638841025,0.7204845547676086,0.101599008962512,0.8826814889907837,0.5032059133052826,989696,seed13711/c1_mse/scores.csv
13711,c2_fault_preserving,0.6345110093152224,63.45110093152224,0.7207046747207642,0.10282367058098311,0.883502185344696,0.5033363819122314,989696,seed13711/c2_fault_preserving/scores.csv
42,c1_mse,0.6334202406721446,63.34202406721447,0.7210698127746582,0.1014079667627811,0.8769883513450623,0.4924710601568221,989696,seed42/c1_mse/scores.csv
42,c2_fault_preserving,0.6336608526718583,63.36608526718584,0.7211659550666809,0.10194057375192637,0.8772093653678894,0.49315960705280304,989696,seed42/c2_fault_preserving/scores.csv
2026,c1_mse,0.6343436523042127,63.43436523042128,0.720145583152771,0.1015052508562803,0.881390392780304,0.5057098925113678,989696,seed2026/c1_mse/scores.csv
2026,c2_fault_preserving,0.6343941401825759,63.43941401825759,0.7207969427108765,0.10280066207051271,0.881390392780304,0.5069126784801483,989696,seed2026/c2_fault_preserving/scores.csv

```

## Log tail

```text
{"event": "job_started", "run_id": "server02_fp_naa_screening_20260817T145911Z", "stage": "screening"}
{"event": "stage", "step": "assets"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "pytest", "tests/unit/test_fp_naa_adapter.py", "tests/unit/test_fp_naa_candidate.py", "-q"]}
............                                                             [100%]
12 passed in 5.59s
{"event": "stage", "step": "augmentation-cache"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "cache-augmentation", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--audio-root", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted", "--output-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counterfactual_fp32infer_v3", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v3.yaml", "--beats-source", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/unilm_833df7e7832e/beats", "--checkpoint", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/BEATs_iter3.pt", "--workers", "12", "--device", "cuda"]}
FP-NAA augmentation cache complete. clips=7000 heldout=643 
index=/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counter
factual_fp32infer_v3/index.parquet
{"event": "stage", "step": "screening"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "screen-dev", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--augmentation-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counterfactual_fp32infer_v3", "--c0-scores", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_c0_20260817T064722Z/c0_baseline/freq_rdp8_beam/scores.csv", "--output-dir", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_screening_20260817T145911Z/screening", "--checkpoint-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/checkpoints/server02_fp_naa_screening_20260817T145911Z", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v3.yaml", "--experiment-id", "server02_fp_naa_screening_20260817T145911Z", "--device", "cuda", "--preload-workers", "12"]}
FP-NAA C1/C2 screening complete. core_gate=failed 
summary=/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_screening
_20260817T145911Z/screening/screening_summary.csv
```
