# FP-NAA screening report

run_id=server02_fp_naa_screening_20260817T234156Z
source_git_sha=bcabc4331e727557d679b3db4adb224a5f2368c2
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
    "bearingEmu": 0.022053872613279513,
    "fan": 0.0006098241294898443,
    "gearboxEmu": 0.024138371500327604,
    "sliderEmu": 0.0004709971053289408,
    "valveEmu": 0.07439158442787042
  },
  "note": "passed remains false until the preregistered LOMO gate is run",
  "observability_decomposition": {
    "adapter_retention_median": 0.9304538071155548,
    "adapter_retention_worst_seed_q05": 0.6794865846633912,
    "frontend_retention_median": 0.7731100022792816,
    "frontend_retention_worst_seed_q05": 0.1340193301439285,
    "note": "diagnostic only; the frozen combined retention checks remain authoritative",
    "transport_error_median": 0.694331169128418,
    "transport_error_worst_seed_q90": 0.9869839251041412
  },
  "passed": false,
  "retention": {
    "heldout_median_across_seeds": 0.881390392780304,
    "heldout_worst_seed_q05": 0.4924710601568221,
    "in_support_median_across_seeds": 0.7204845547676086,
    "in_support_worst_seed_q05": 0.1014079667627811
  },
  "schema_version": 1,
  "scores": {
    "c0": 0.6197676455378447,
    "c1_mean": 0.63405655228682,
    "c2_mean": 0.63405655228682,
    "c2_minus_c0": 0.014288906748975316,
    "c2_minus_c1": 0.0
  }
}

```

## Summary

```csv
seed,candidate,official_score,official_score_percent,in_support_retention_median,in_support_retention_q05,heldout_retention_median,heldout_retention_q05,in_support_frontend_retention_median,in_support_frontend_retention_q05,in_support_adapter_retention_median,in_support_adapter_retention_q05,in_support_transport_error_median,in_support_transport_error_q90,trainable_parameters,score_path
13711,c1_mse,0.6344057638841025,63.44057638841025,0.7204845547676086,0.101599008962512,0.8826814889907837,0.5032059133052826,,,,,,,989696,seed13711/c1_mse/scores.csv
13711,c2_fault_preserving,0.6344057638841025,63.44057638841025,0.7204845547676086,0.101599008962512,0.8826814889907837,0.5032059133052826,0.7731100022792816,0.1340193301439285,0.9310665130615234,0.6848273307085038,0.694331169128418,0.9869084596633911,989696,seed13711/c2_fault_preserving/scores.csv
42,c1_mse,0.6334202406721446,63.34202406721447,0.7210698127746582,0.1014079667627811,0.8769883513450623,0.4924710601568221,,,,,,,989696,seed42/c1_mse/scores.csv
42,c2_fault_preserving,0.6334202406721446,63.34202406721447,0.7210698127746582,0.1014079667627811,0.8769883513450623,0.4924710601568221,0.7731100022792816,0.1340193301439285,0.9300241470336914,0.6836366266012193,0.6942053139209747,0.9869839251041412,989696,seed42/c2_fault_preserving/scores.csv
2026,c1_mse,0.6343436523042127,63.43436523042128,0.720145583152771,0.1015052508562803,0.881390392780304,0.5057098925113678,,,,,,,989696,seed2026/c1_mse/scores.csv
2026,c2_fault_preserving,0.6343436523042127,63.43436523042128,0.720145583152771,0.1015052508562803,0.881390392780304,0.5057098925113678,0.7731100022792816,0.1340193301439285,0.9304538071155548,0.6794865846633912,0.6945322453975677,0.9867941498756408,989696,seed2026/c2_fault_preserving/scores.csv

```

## Log tail

```text
{"event": "job_started", "run_id": "server02_fp_naa_screening_20260817T234156Z", "stage": "screening"}
{"event": "stage", "step": "assets"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "pytest", "tests/unit/test_fp_naa_adapter.py", "tests/unit/test_fp_naa_candidate.py", "-q"]}
..................                                                       [100%]
18 passed in 5.41s
{"event": "stage", "step": "augmentation-cache"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "cache-augmentation", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--audio-root", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted", "--output-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counterfactual_fp32infer_v3", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v5.yaml", "--beats-source", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/unilm_833df7e7832e/beats", "--checkpoint", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/BEATs_iter3.pt", "--workers", "12", "--device", "cuda"]}
FP-NAA augmentation cache complete. clips=7000 heldout=643 
index=/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counter
factual_fp32infer_v3/index.parquet
{"event": "stage", "step": "screening"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "screen-dev", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--augmentation-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counterfactual_fp32infer_v3", "--c0-scores", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_c0_20260817T064722Z/c0_baseline/freq_rdp8_beam/scores.csv", "--output-dir", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_screening_20260817T234156Z/screening", "--checkpoint-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/checkpoints/server02_fp_naa_screening_20260817T234156Z", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v5.yaml", "--experiment-id", "server02_fp_naa_screening_20260817T234156Z", "--device", "cuda", "--preload-workers", "12"]}
/home/ubuntu/miniconda3/envs/care-asd-fp-naa/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:227: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn(
/home/ubuntu/miniconda3/envs/care-asd-fp-naa/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:227: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn(
/home/ubuntu/miniconda3/envs/care-asd-fp-naa/lib/python3.11/site-packages/torch/optim/lr_scheduler.py:227: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn(
FP-NAA C1/C2 screening complete. core_gate=failed 
summary=/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_screening
_20260817T234156Z/screening/screening_summary.csv
```
