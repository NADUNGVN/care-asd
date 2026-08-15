# Phase 10 SAFE-REF report

run_id=server02_phase10_reference_safety_20260815T145112Z
cache=/home/ubuntu/Dung_TDTU/data/CARE_ASD/reference_safety_cache/dev/server02_phase10_reference_safety_20260815T102535Z
cache_reused=true
workers=12
task_status=2

## Gate

```json
{
  "calibration": {
    "accepted": 1767,
    "cases": 2048,
    "coverage": 0.86279296875,
    "false_safe_count": 1640,
    "false_safe_rate": 0.9281267685342388,
    "false_safe_upper_ci": 0.9392632796974412,
    "policy_loss_q95": 0.8086027863828539,
    "risk_spearman": 0.005933639840293943,
    "safe_cases": 163,
    "safe_prevalence": 0.07958984375,
    "tail_loss_reduction": 0.012747087021524606,
    "unconditional_loss_q95": 0.8190432013447839
  },
  "criteria": {
    "false_safe_max": 0.05,
    "false_safe_upper_ci_max": 0.1,
    "minimum_coverage": 0.2,
    "minimum_risk_spearman": 0.6,
    "minimum_tail_loss_reduction": 0.5
  },
  "holdout": {
    "accepted": 1776,
    "cases": 2048,
    "coverage": 0.8671875,
    "false_safe_count": 1649,
    "false_safe_rate": 0.928490990990991,
    "false_safe_upper_ci": 0.9395727389612808,
    "policy_loss_q95": 0.8043543651332069,
    "risk_spearman": 0.039958643563944324,
    "safe_cases": 172,
    "safe_prevalence": 0.083984375,
    "tail_loss_reduction": 0.017798982585734313,
    "unconditional_loss_q95": 0.8189304947481562
  },
  "passed": false,
  "schema_version": 1
}

```

## Log tail

```text
Phase 10 SAFE-REF started at 2026-08-15T14:51:12Z
Reusing immutable cache: /home/ubuntu/Dung_TDTU/data/CARE_ASD/reference_safety_cache/dev/server02_phase10_reference_safety_20260815T102535Z
SAFE-REF simulation completed. gate=failed 
summary=reports/reference_safety/server02_phase10_reference_safety_20260815T1451
12Z/simulation/summary.csv

```
