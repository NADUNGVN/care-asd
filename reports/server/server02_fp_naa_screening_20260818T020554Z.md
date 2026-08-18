# FP-NAA screening report

run_id=server02_fp_naa_screening_20260818T020554Z
source_git_sha=c63e7be7f8eb55f5e55d16c66719b5b28afc6d4d
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
    "ToyCar": -0.00040962432140512206,
    "ToyCarEmu": 0.005352716062072216,
    "bearingEmu": 0.021014644809805305,
    "fan": 0.0034090148209654503,
    "gearboxEmu": 0.022078645395390062,
    "sliderEmu": -0.0013729545751095529,
    "valveEmu": 0.08034616309796483
  },
  "note": "passed remains false until the preregistered LOMO gate is run",
  "observability_decomposition": {
    "adapter_retention_median": 0.9350758194923401,
    "adapter_retention_worst_seed_q05": 0.664722529053688,
    "frontend_retention_median": 0.7731100022792816,
    "frontend_retention_worst_seed_q05": 0.1340193301439285,
    "note": "diagnostic only; the frozen combined retention checks remain authoritative",
    "transport_error_median": 0.682340681552887,
    "transport_error_worst_seed_q90": 0.9778082013130189
  },
  "passed": false,
  "retention": {
    "heldout_median_across_seeds": 0.8618637323379517,
    "heldout_worst_seed_q05": 0.4674616038799285,
    "in_support_median_across_seeds": 0.7224204540252686,
    "in_support_worst_seed_q05": 0.10110606588423246
  },
  "schema_version": 1,
  "scores": {
    "c0": 0.6197676455378447,
    "c1_mean": 0.63405655228682,
    "c2_mean": 0.6341575901941207,
    "c2_minus_c0": 0.014389944656275966,
    "c2_minus_c1": 0.00010103790730064954
  }
}

```

## Summary

```csv
seed,candidate,official_score,official_score_percent,in_support_retention_median,in_support_retention_q05,heldout_retention_median,heldout_retention_q05,in_support_frontend_retention_median,in_support_frontend_retention_q05,in_support_adapter_retention_median,in_support_adapter_retention_q05,in_support_transport_error_median,in_support_transport_error_q90,trainable_parameters,score_path
13711,c1_mse,0.6344057638841025,63.44057638841025,0.7204845547676086,0.101599008962512,0.8826814889907837,0.5032059133052826,,,,,,,989696,seed13711/c1_mse/scores.csv
13711,c2_fault_preserving,0.6344508521958595,63.44508521958595,0.7218388915061951,0.10122833251953121,0.8628453016281128,0.47716618180274956,0.7731100022792816,0.1340193301439285,0.9352988600730896,0.6660874366760254,0.682500422000885,0.9774319350719453,989696,seed13711/c2_fault_preserving/scores.csv
42,c1_mse,0.6334202406721446,63.34202406721447,0.7210698127746582,0.1014079667627811,0.8769883513450623,0.4924710601568221,,,,,,,989696,seed42/c1_mse/scores.csv
42,c2_fault_preserving,0.6339011725574971,63.390117255749715,0.7224204540252686,0.10110606588423246,0.8583921790122986,0.4674616038799285,0.7731100022792816,0.1340193301439285,0.9340775310993195,0.6670399814844131,0.682340681552887,0.9778082013130189,989696,seed42/c2_fault_preserving/scores.csv
2026,c1_mse,0.6343436523042127,63.43436523042128,0.720145583152771,0.1015052508562803,0.881390392780304,0.5057098925113678,,,,,,,989696,seed2026/c1_mse/scores.csv
2026,c2_fault_preserving,0.6341207458290054,63.41207458290054,0.7227644622325897,0.10113561116158958,0.8618637323379517,0.482210299372673,0.7731100022792816,0.1340193301439285,0.9350758194923401,0.664722529053688,0.6816520094871521,0.976802307367325,989696,seed2026/c2_fault_preserving/scores.csv

```

## Log tail

```text
{"event": "job_started", "run_id": "server02_fp_naa_screening_20260818T020554Z", "stage": "screening"}
{"event": "stage", "step": "assets"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "pytest", "tests/unit/test_fp_naa_adapter.py", "tests/unit/test_fp_naa_candidate.py", "-q"]}
....................                                                     [100%]
20 passed in 3.77s
{"event": "stage", "step": "augmentation-cache"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "cache-augmentation", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--audio-root", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted", "--output-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counterfactual_fp32infer_v3", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v5.yaml", "--beats-source", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/unilm_833df7e7832e/beats", "--checkpoint", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/BEATs_iter3.pt", "--workers", "12", "--device", "cuda"]}
FP-NAA augmentation cache complete. clips=7000 heldout=643 
index=/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counter
factual_fp32infer_v3/index.parquet
{"event": "stage", "step": "screening"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "screen-dev", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--augmentation-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/augmentation_cache/dev/counterfactual_fp32infer_v3", "--c0-scores", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_c0_20260817T064722Z/c0_baseline/freq_rdp8_beam/scores.csv", "--output-dir", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_screening_20260818T020554Z/screening", "--checkpoint-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/checkpoints/server02_fp_naa_screening_20260818T020554Z", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v5.yaml", "--experiment-id", "server02_fp_naa_screening_20260818T020554Z", "--device", "cuda", "--preload-workers", "12"]}
FP-NAA C1/C2 screening complete. core_gate=failed 
summary=/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_screening
_20260818T020554Z/screening/screening_summary.csv
```
