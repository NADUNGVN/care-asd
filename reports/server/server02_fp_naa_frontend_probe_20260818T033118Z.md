# FP-NAA frontend-probe report

run_id=server02_fp_naa_frontend_probe_20260818T033118Z
source_git_sha=70d728275824f6f26a89c960a40ebfe72e9d123f
conda_environment=care-asd-fp-naa
workers=12
task_status=0
gate_passed=false

## Gate

```json
{
  "checks": {
    "tap_0": false,
    "tap_12": false,
    "tap_4": false,
    "tap_8": false
  },
  "criteria": {
    "in_support_retention_median_minimum": 0.9,
    "in_support_retention_q05_minimum": 0.75
  },
  "gate": "V6_frontend_observability_preflight",
  "heldout_family": {
    "name": "friction_burst",
    "role": "diagnostic only; never used for tap selection"
  },
  "note": "A pass only authorizes implementation of a pre-encoder candidate. It is not a G2 performance result and does not authorize LOMO.",
  "passed": false,
  "schema_version": 1,
  "selected_tap": null,
  "selection_rule": "deepest eligible tap using in-support pseudo-faults only"
}

```

## Summary

```csv
tap,in_support_clips,in_support_retention_median,in_support_retention_q05,in_support_direction_median,in_support_transport_error_median,heldout_clips,heldout_retention_median,heldout_retention_q05,eligible_in_support
0,7000,0.8465386404815177,0.4093625829094128,0.6429818323283221,0.796890033953509,643,0.9302852629306991,0.6209686442202004,False
4,7000,0.7930166268869006,0.2765919597880857,0.7093001090997118,0.7172676371459852,643,0.9355737312599972,0.6085902359343263,False
8,7000,0.7839335449307829,0.17562607079409295,0.7183795594737279,0.7072832310131933,643,0.9447596344055549,0.5967977652830165,False
12,7000,0.7731084317043231,0.13401815128420091,0.7259569989404011,0.6992422102345817,643,0.9515514452501791,0.5744937196155796,False

```

## family_summary.csv

```text
tap,fault_set,fault_family,clips,retention_median,retention_q05,direction_median,transport_error_median,transport_error_q90
0,heldout,friction_burst,643,0.9302852629306991,0.6209686442202004,0.794530299961912,0.6218671971036815,0.8753753905636065
0,in_support,amplitude_modulation,2411,0.787526210631889,0.3449074460192558,0.6893702226312676,0.7323614145680755,0.9366466889648835
0,in_support,frequency_modulation,2340,0.8808103251491404,0.5508659752515731,0.46312372357000636,1.0352023151803011,1.5046808099356181
0,in_support,periodic_resonance,2249,0.876834115846331,0.41574954826128724,0.7657198765974483,0.655322338216131,0.9757623223741666
4,heldout,friction_burst,643,0.9355737312599972,0.6085902359343263,0.859925183390936,0.5155630270778242,0.799855885605793
4,in_support,amplitude_modulation,2411,0.6776480038104797,0.192185126924516,0.7472189817334809,0.6736562114189097,0.9365551138968202
4,in_support,frequency_modulation,2340,0.8331673912033714,0.3891000949187572,0.564378203786879,0.883194602885027,1.1611065072776303
4,in_support,periodic_resonance,2249,0.8716077668290255,0.37358511790905535,0.8010592693224504,0.6050422115734674,0.9540800904924531
8,heldout,friction_burst,643,0.9447596344055549,0.5967977652830165,0.8598198308660874,0.5184923177451871,0.7976702360431768
8,in_support,amplitude_modulation,2411,0.6963521196117534,0.109151896752495,0.7274102840077747,0.6928008020220026,0.9705536270098194
8,in_support,frequency_modulation,2340,0.8051790282347716,0.2571756927390126,0.6260343895263478,0.8153133827409447,1.0530671757990435
8,in_support,periodic_resonance,2249,0.8671640468047416,0.31179832313711214,0.7942419891387522,0.613719103606237,0.9486199590507016
12,heldout,friction_burst,643,0.9515514452501791,0.5744937196155796,0.868475432612757,0.4999655227853597,0.7985519723024121
12,in_support,amplitude_modulation,2411,0.6971683585385476,0.08023257345921583,0.7336516284483948,0.6898830275298593,0.9847574387379628
12,in_support,frequency_modulation,2340,0.7730218519506489,0.1936420165870017,0.6409099336743895,0.8010331258272129,1.0311979096874528
12,in_support,periodic_resonance,2249,0.8612826253254673,0.27870905313319055,0.8029514968899898,0.6057580151061978,0.9437846190273598

```

## Log tail

```text
{"event": "job_started", "run_id": "server02_fp_naa_frontend_probe_20260818T033118Z", "stage": "frontend-probe"}
{"event": "stage", "step": "assets"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "pytest", "tests/unit/test_beats_frontend.py", "tests/unit/test_fp_naa_observability.py", "-q"]}
..........                                                               [100%]
10 passed in 2.53s
{"event": "stage", "step": "frontend-observability"}
{"event": "command", "argv": ["/home/ubuntu/miniconda3/envs/care-asd-fp-naa/bin/python", "-m", "care_asd.cli", "fp-naa", "observability-dev", "--base-cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/beats_cache/dev/beats_iter3_stereo_10s_fp32infer_v2", "--audio-root", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/raw/dcase2026/dev/extracted", "--cache-dir", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/fp_naa/observability_cache/dev/beats_depth_v6", "--output-dir", "/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_frontend_probe_20260818T033118Z/frontend_probe", "--config", "/home/ubuntu/Dung_TDTU/CARE_ASD/configs/experiment/fp_naa_v6.yaml", "--beats-source", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/unilm_833df7e7832e/beats", "--checkpoint", "/home/ubuntu/Dung_TDTU/data/CARE_ASD/external/fp_naa/BEATs_iter3.pt", "--workers", "12", "--device", "cuda"]}
/home/ubuntu/miniconda3/envs/care-asd-fp-naa/lib/python3.11/site-packages/torch/nn/utils/weight_norm.py:143: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
FP-NAA observability preflight complete. gate=failed 
summary=/home/ubuntu/Dung_TDTU/CARE_ASD/reports/fp_naa/server02_fp_naa_frontend_
probe_20260818T033118Z/frontend_probe/tap_summary.csv
```
