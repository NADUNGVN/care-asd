# care_asd_literature_audit_v1

Literature cutoff: **2026-08-16**. Curated sources: **13** (2 peer-reviewed; 6 non-peer-reviewed DCASE technical reports).

## Positioning decision

An ASD-specific, frozen known-component safety audit that separates downstream anomaly-score effects from physical noise attenuation and injected-fault retention for deterministic normal-only contaminated-reference processing.

Known-component fault/noise safety evaluated by directly reviewed sources: **1**.

## Supported claims

- `contaminated_reference_is_real`: The far microphone is a mixed observation rather than a guaranteed noise-only reference, creating a plausible fault-suppression risk for direct subtraction. Sources: nishida2026_task_description, kim2026_reference_denoising, lei2026_task_adapted_dual_mic.
- `metric_gain_is_not_safety_evidence`: Published ASD score gains from reference processing do not by themselves establish component-level fault preservation. Sources: kim2026_reference_denoising, morita2026_stereo_spatial, ozeki2026_simple_noise_reduction, qian2026_spectral_subtraction, kim2026_residual_view.
- `audit_gap`: The reviewed direct ASD sources do not jointly report known-component noise attenuation, fault retention, aligned anomaly-score effects, and frozen stop decisions. Sources: fujimura2026_na_ssl, kim2026_reference_denoising, morita2026_stereo_spatial, ozeki2026_simple_noise_reduction, qian2026_spectral_subtraction, kim2026_residual_view, lei2026_task_adapted_dual_mic.
- `adjacent_precedent`: Desired-signal leakage and preservation-versus-attenuation trade-offs are established signal-processing problems outside anomalous sound detection. Sources: alkindi1989_signal_leakage, xiao2024_target_delay_anc, zhao2025_aux_reference_aec.

## Prohibited claims

- `no_dual_mic_benefit`: Do not claim that dual-microphone information is generally ineffective or harmful for anomalous sound detection. Counterexamples: fujimura2026_na_ssl, kim2026_reference_denoising, morita2026_stereo_spatial, ozeki2026_simple_noise_reduction, qian2026_spectral_subtraction, kim2026_residual_view.
- `contamination_is_novel`: Do not claim that contaminated-reference cancellation or desired-signal leakage is itself a new problem. Counterexamples: alkindi1989_signal_leakage, xiao2024_target_delay_anc.
- `universal_impossibility`: Do not claim a universal impossibility theorem for safe normal-only reference use or generalize beyond the tested controllers and controlled mixtures. Counterexamples: fujimura2026_na_ssl, zhao2025_aux_reference_aec.
- `peer_reviewed_dcase_reports`: Do not describe DCASE challenge technical reports or the current arXiv papers as peer-reviewed journal evidence. Counterexamples: dcase2026_submission_policy, kim2026_reference_denoising, morita2026_stereo_spatial, ozeki2026_simple_noise_reduction, qian2026_spectral_subtraction, kim2026_residual_view, lei2026_task_adapted_dual_mic, fujimura2026_na_ssl.

## Source matrix

| ID | Cluster | Method | Review | Safety audit |
|---|---|---|---|---:|
| dcase2026_task | task_context | benchmark_definition | official_specification | false |
| nishida2026_task_description | task_context | benchmark_definition | not_peer_reviewed | false |
| dcase2026_submission_policy | task_context | benchmark_definition | official_specification | false |
| fujimura2026_na_ssl | direct_asd | learned_dual_channel_representation | not_peer_reviewed | false |
| kim2026_reference_denoising | direct_asd | deterministic_signal_subtraction | not_peer_reviewed | false |
| morita2026_stereo_spatial | direct_asd | spatial_masking | not_peer_reviewed | false |
| ozeki2026_simple_noise_reduction | direct_asd | deterministic_signal_subtraction | not_peer_reviewed | false |
| qian2026_spectral_subtraction | direct_asd | deterministic_signal_subtraction | not_peer_reviewed | false |
| kim2026_residual_view | direct_asd | embedding_residual | not_peer_reviewed | false |
| lei2026_task_adapted_dual_mic | direct_asd | learned_dual_channel_representation | not_peer_reviewed | false |
| alkindi1989_signal_leakage | adjacent_signal_processing | adaptive_cancellation | peer_reviewed | false |
| xiao2024_target_delay_anc | adjacent_signal_processing | active_noise_control | peer_reviewed | true |
| zhao2025_aux_reference_aec | adjacent_signal_processing | learned_reference_purification | not_peer_reviewed | false |
