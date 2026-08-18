# FP-NAA layerwise-preflight report

run_id=server02_fp_naa_layerwise_preflight_20260818T060510Z
source_git_sha=25d42347332a71d7898d3e6bc7eca822dc048a37
conda_environment=care-asd-fp-naa
workers=12
task_status=0
gate_passed=false

## Gate

```json
{
  "authorization": "A pass authorizes V8 three-seed G2 implementation and execution only. It is not a development performance result and does not authorize LOMO.",
  "checks": {
    "actual_beats_runtime_probe": true,
    "finite_real_updates": true,
    "heldout_retention_median": false,
    "heldout_retention_q05": false,
    "in_support_retention_median": false,
    "in_support_retention_q05": false,
    "median_gain_over_l1": true,
    "normal_function_anchor": true,
    "q05_gain_over_l1": false
  },
  "criteria": {
    "heldout_retention_median_minimum": 0.85,
    "heldout_retention_q05_minimum": 0.6,
    "normal_function_drift_maximum": 0.1,
    "retention_median_gain_minimum": 0.05,
    "retention_median_minimum": 0.9,
    "retention_q05_gain_minimum": 0.1,
    "retention_q05_minimum": 0.6
  },
  "gate": "V8_M_layerwise_mechanism_preflight",
  "metrics": {
    "l1_in_support_retention_median": 0.7145631588768273,
    "l1_in_support_retention_q05": 0.08373789190076787,
    "l2_heldout_retention_median": 0.8310962627073277,
    "l2_heldout_retention_q05": 0.30016171199612285,
    "l2_in_support_retention_median": 0.7810628887341104,
    "l2_in_support_retention_q05": 0.10920489090030029,
    "l2_minus_l1_retention_median": 0.06649972985728314,
    "l2_minus_l1_retention_q05": 0.02546699899953242,
    "l2_normal_function_drift_median": 0.08185541358834326
  },
  "optimizer_update_norms": {
    "common": 15.068397317954739,
    "l1": 8.184771024333099,
    "l2": 5.671692947421786
  },
  "passed": false,
  "runtime_probe": {
    "frozen_path_relative_error": 5.869454184903589e-07,
    "frozen_path_relative_error_maximum": 1e-05,
    "optimizer_update_norm": 0.12832585127023874,
    "schema_version": 1,
    "status": "passed",
    "trainable_parameters": 11876352
  },
  "schema_version": 1,
  "trainable_parameters": 11876352
}

```

## Summary

```csv
candidate,fault_set,clips,retention_median,retention_q05,direction_median,transport_error_median,normal_function_drift_median
l1_layerwise_mse,heldout,256,0.8800444017652012,0.3594371197260544,0.8863250238175069,0.46661743033018604,0.08859170445649911
l1_layerwise_mse,in_support,512,0.7145631588768273,0.08373789190076787,0.7413873708136376,0.6843045075086985,0.08859170445649911
l2_layerwise_fault_transport,heldout,256,0.8310962627073277,0.30016171199612285,0.8830675665804583,0.47691633481980095,0.081565463229735
l2_layerwise_fault_transport,in_support,512,0.7810628887341104,0.10920489090030029,0.8032619124090021,0.6054658159254709,0.08185541358834326

```

## runtime_probe.json

```text
{
  "frozen_path_relative_error": 5.869454184903589e-07,
  "frozen_path_relative_error_maximum": 1e-05,
  "optimizer_update_norm": 0.12832585127023874,
  "schema_version": 1,
  "status": "passed",
  "trainable_parameters": 11876352
}

```

## Log tail

```text
{"event": "job_started", "run_id": "server02_fp_naa_layerwise_preflight_20260818T060510Z", "stage": "layerwise-preflight"}
{"event": "stage", "step": "assets"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "pytest", "tests/unit/test_layerwise_noise_aware.py", "tests/unit/test_fp_naa_layerwise_preflight.py", "tests/unit/test_fp_naa_config.py", "-q"]}
..............                                                           [100%]
14 passed in 3.58s
{"event": "stage", "step": "layerwise-preflight"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "layerwise-preflight-dev", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--audio-root", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted", "--cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/layerwise_preflight_cache/v8_seed2608", "--output-dir", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_layerwise_preflight_20260818T060510Z/layerwise_preflight", "--checkpoint-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/checkpoints/server02_fp_naa_layerwise_preflight_20260818T060510Z", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v8.yaml", "--beats-source", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/unilm_833df7e7832e/beats", "--checkpoint", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/BEATs_iter3.pt", "--workers", "12", "--device", "cuda"]}
/home/ubuntu/miniconda3/envs/care-asd-fp-naa/lib/python3.11/site-packages/torch/nn/utils/weight_norm.py:143: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
V8 actual-BEATs runtime probe passed. frozen_path_relative_error=5.869e-07 update_norm=1.283e-01
V8 cache ready. train=1024 validation=512 heldout=256
V8 common epoch=1/10 loss=0.0413038
V8 common epoch=2/10 loss=0.0345475
V8 common epoch=3/10 loss=0.0324552
V8 common epoch=4/10 loss=0.0310035
V8 common epoch=5/10 loss=0.0300401
V8 common epoch=6/10 loss=0.0292695
V8 common epoch=7/10 loss=0.0283890
V8 common epoch=8/10 loss=0.0279519
V8 common epoch=9/10 loss=0.0272969
V8 common epoch=10/10 loss=0.0269808
V8 l1_mse epoch=1/10 loss=0.0268868
V8 l1_mse epoch=2/10 loss=0.0261544
V8 l1_mse epoch=3/10 loss=0.0258373
V8 l1_mse epoch=4/10 loss=0.0258733
V8 l1_mse epoch=5/10 loss=0.0253968
V8 l1_mse epoch=6/10 loss=0.0251354
V8 l1_mse epoch=7/10 loss=0.0250556
V8 l1_mse epoch=8/10 loss=0.0246223
V8 l1_mse epoch=9/10 loss=0.0243482
V8 l1_mse epoch=10/10 loss=0.0241710
V8 l2_fault_transport epoch=1/10 loss=0.9201402
V8 l2_fault_transport epoch=2/10 loss=0.8874311
V8 l2_fault_transport epoch=3/10 loss=0.8570019
V8 l2_fault_transport epoch=4/10 loss=0.8554897
V8 l2_fault_transport epoch=5/10 loss=0.8458545
V8 l2_fault_transport epoch=6/10 loss=0.8290435
V8 l2_fault_transport epoch=7/10 loss=0.8163530
V8 l2_fault_transport epoch=8/10 loss=0.8134198
V8 l2_fault_transport epoch=9/10 loss=0.8084790
V8 l2_fault_transport epoch=10/10 loss=0.8072485
FP-NAA V8 mechanism preflight complete. gate=False
FP-NAA V8 layerwise preflight complete. gate=failed 
summary=/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_layerwise
_preflight_20260818T060510Z/layerwise_preflight/summary.csv
```
