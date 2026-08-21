# fp_naa_post_v10_literature_delta

Literature cutoff: **2026-08-21**. Curated sources: **9** (0 peer-reviewed; 4 non-peer-reviewed DCASE technical reports).

## Positioning decision

The post-V10 literature gate rejects threshold-only tail rescue, another correlated BEATs score union, and proxy-fault model selection as immediate successors; it retains cross-channel predictive residuals and domain-aware calibration only as prior-art comparators inside a new safety or transfer study, not as novel components by themselves.

Known-component fault/noise safety evaluated by directly reviewed sources: **0**.

## Supported claims

- `backend_diversity_precedes_pooling`: For the next bounded study, backend and representation diversity deserves priority over another temporal-pooling or same-representation branch search. Sources: zhou2026_backend_systematic, jiang2026_aithu_fusion, dcase2026_task_results.
- `proxy_fault_selection_requires_external_validation`: Pseudo-fault or proxy-outlier selection cannot by itself authorize a performance run because proxy discrimination may fail to rank real-anomaly backends. Sources: zhou2026_backend_systematic, mkrtchian2026_daco.
- `calibration_is_a_control_not_a_power_certificate`: Domain-aware normal-score calibration is a necessary transfer control, but normal-distribution balance does not certify anomaly-detection power and requires a frozen viability veto. Sources: mkrtchian2026_daco.
- `orthogonal_stereo_residual_is_conditional`: Cross-channel predictive residuals remain mechanistically distinct from the failed V10 score branches, but only a new safety, identifiability, or transfer protocol could be a CARE-ASD contribution. Sources: jeong2026_cross_channel_residual, shokriazar2026_shiftall_ldknn.
- `no_immediate_v11_gpu_authorization`: The literature delta does not justify immediate V11 GPU training; a successor must first state a mechanism not reducible to tail-threshold tuning, ordinary fusion, or reconstruction of a published ensemble. Sources: kajita2026_train_normal_profile, jiang2026_aithu_fusion, dcase2026_task_results.

## Prohibited claims

- `top_tail_rescue_is_new`: Do not present sparse top-tail rescue or a relaxed median-gain rule as a novel V11 mechanism. Counterexamples: kajita2026_train_normal_profile.
- `cross_channel_residual_is_prior_art`: Do not claim that predicting one channel representation from the other and scoring the residual is itself novel. Counterexamples: jeong2026_cross_channel_residual.
- `dev_score_guarantees_unseen_transfer`: Do not use development-score ranking alone as evidence that a configuration will transfer to unseen evaluation machine types. Counterexamples: mkrtchian2026_daco, shokriazar2026_shiftall_ldknn.
- `proxy_fault_gate_proves_real_anomaly_power`: Do not claim that passing a synthetic or proxy-fault gate proves ranking power on real anomalies. Counterexamples: zhou2026_backend_systematic.
- `ordinary_score_fusion_is_novel`: Do not claim generic score calibration, rank fusion, maximum fusion, or multi-branch ensembling as a CARE-ASD novelty. Counterexamples: dcase2026_task_results, jiang2026_aithu_fusion, kajita2026_train_normal_profile.
- `technical_reports_are_peer_reviewed`: Do not describe DCASE challenge technical reports in this delta as peer-reviewed scientific evidence. Counterexamples: dcase2026_submission_policy.

## Source matrix

| ID | Cluster | Method | Review | Safety audit |
|---|---|---|---|---:|
| dcase2026_task_results | task_context | benchmark_definition | official_specification | false |
| dcase2026_submission_policy | task_context | benchmark_definition | official_specification | false |
| zhou2026_backend_systematic | direct_asd | backend_diversity_fusion | not_peer_reviewed | false |
| mkrtchian2026_daco | direct_asd | domain_aware_score_calibration | not_peer_reviewed | false |
| jeong2026_cross_channel_residual | direct_asd | cross_channel_predictive_residual | not_peer_reviewed | false |
| shokriazar2026_shiftall_ldknn | direct_asd | multi_encoder_multiview_fusion | not_peer_reviewed | false |
| kajita2026_train_normal_profile | direct_asd | train_normal_profile_ensemble | not_peer_reviewed | false |
| jiang2026_aithu_fusion | direct_asd | heterogeneous_score_fusion | not_peer_reviewed | false |
| leclei2022_n_minus_one | adjacent_model_selection | unsupervised_model_selection | not_peer_reviewed | false |
