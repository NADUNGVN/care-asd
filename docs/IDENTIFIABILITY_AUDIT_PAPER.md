# CARE-ASD identifiability/audit paper execution plan

## Authority and frozen decision

This is the active publication plan after the preregistered AP-CARE G1 stop
rule. The immutable run `server02_ap_care_g1_20260816T105721Z` completed 512
controlled cases and failed five of six checks. AP-G2, AP-G3, AP-G4, and AP-G5
are stopped. The method may not be retuned on the frozen G1 holdout.

The working paper question is:

> When a far acoustic reference contains an unknown mixture of environmental
> interference and machine-origin energy, what can normal-only stereo
> statistics justify removing without erasing anomalous evidence?

The result is an empirical boundary under the tested DCASE-compatible and
controlled conditions. It is not a distribution-free impossibility theorem.

After Audit-A1, the working title is:

> **When the Noise Reference Contains the Machine: A Controlled Safety Audit
> for Anomalous Sound Detection**

The frozen literature matrix and claim boundary are documented in
`docs/LITERATURE_MATRIX_AUDIT_A1.md`. Positive DCASE 2026 dual-microphone
results explicitly rule out a generic claim against far-microphone use. The
retained scope is deterministic, normal-only, signal-level contaminated-reference
processing and its component-level safety evaluation.

Audit-A2 adds the frozen machine/domain sensitivity analysis documented in
`docs/ROBUSTNESS_APPENDIX_AUDIT_A2.md`. B01's pAUC harm survives every
one-machine deletion, while B02 remains inconclusive. Both interventions are
more favorable on source than target data on average, so the manuscript must
report domain fragility rather than a uniform per-machine failure.

## Frozen evidence chain

| Stage | Test | Result | Interpretation |
|---|---|---|---|
| Phase 6 | B00 aligned near-only reference | AUC 0.60813; pAUC 0.55391 | Locked comparator |
| Phase 7 | B01 residual replacement | pAUC delta -0.01740, 95% CI [-0.03271, -0.00286] | Significant harm |
| Phase 8 | Frozen displacement audit | rho -0.5524 over 1,400 clips | More removal associated with lower anomaly score |
| Phase 9 | B02 near-primary gated residual | pAUC delta -0.00540, CI [-0.01398, 0.00323] | No demonstrated improvement |
| Phase 10 | SAFE-REF known-component holdout | 92.85% false-safe; risk rho 0.0400 | Safe-use selector not identifiable |
| AP-G1 | Bounded AP-CARE known-component holdout | 5/6 checks failed; attenuation -0.0397 dB | Preservation without useful cancellation |

The synthesis contract is `configs/experiment/audit_paper_v1.yaml`. It names
only committed evidence and freezes the decisions `method_route=stopped`,
`gpu_replication=prohibited`, and `evaluation_access=prohibited`.

## Reproducible paper package

Run:

```bash
uv run care-asd audit synthesize --config configs/experiment/audit_paper_v1.yaml --repo-root . --output-dir reports/audit/care_asd_identifiability_audit_v1
```

The immutable directory contains:

- `performance_evidence.csv`: B00/B01/B02 aligned metrics and bootstrap deltas;
- `identifiability_evidence.csv`: SAFE-REF and AP-CARE gate quantities;
- `ap_care_holdout_diagnostics.csv`: support-by-mismatch diagnostics;
- `performance_deltas.svg`: aligned effect sizes and confidence intervals;
- `identifiability_gates.svg`: risk-proxy values versus preregistered gates;
- `ap_care_mechanism_frontier.svg`: fault retention versus noise attenuation;
- `decision.json`: machine-readable stop decision and prohibited actions;
- `audit_summary.md`: compact results narrative; and
- `run.json`: source paths, portable hashes, code commit, and output hashes.

Text hashes normalize line endings to LF so Windows checkout does not invalidate
Linux-generated provenance. Parquet and other binary files are hashed exactly.

## Manuscript claims

Supported claims:

1. Under a locked official-compatible detector, replacing the near signal with
   the tested residual significantly reduced pAUC.
2. Residual-induced feature removal was associated with reduced anomaly scores,
   but this frozen post-hoc association is explanatory rather than causal proof.
3. The tested normal-only SAFE-REF risk proxy did not identify safe reference
   use in known-component holdout mixtures.
4. The tested bounded AP-CARE controller retained injected fault energy but did
   not achieve the preregistered noise-attenuation or proxy-identifiability
   requirements.
5. These converging results motivate explicit known-component safety audits for
   future contaminated-reference ASD methods.

Unsupported claims:

- all reference microphones are harmful;
- normal-only safe cancellation is universally impossible;
- AP-CARE improves DCASE performance;
- results generalize to unseen evaluation machine types; or
- Jetson board-kit performance validates the failed method.

## Shortest remaining timeline

| Milestone | Work | Compute | Stop rule |
|---|---|---|---|
| Audit-A0 | Frozen evidence package and figures | Local CPU | Artifact/hash audit |
| Audit-A1 | Literature matrix and novelty positioning | None | **Complete; claim boundary frozen** |
| Audit-A2 | Machine/domain robustness appendix from frozen scores | Local CPU | **Complete; no new tuning** |
| Audit-A3 | Manuscript methods, results, limitations | None | **Complete; cross-checked against frozen artifacts** |
| Audit-A4 | Reproducibility bundle and submission checklist | None | **Technical gate passed; author metadata/formatting pending** |

No new training, evaluation-data access, or hardware benchmark is on the critical
path. Any exploratory successor controller must receive a new method ID, new
preregistration, and an independently generated holdout; it cannot revise this
paper's frozen result.

The Audit-A3 draft is
[`CARE_ASD_IDENTIFIABILITY_MANUSCRIPT_DRAFT.md`](CARE_ASD_IDENTIFIABILITY_MANUSCRIPT_DRAFT.md).
The FP-NAA V1-V10 sequence is isolated in
[`FP_NAA_NEGATIVE_EVIDENCE_LEDGER.md`](FP_NAA_NEGATIVE_EVIDENCE_LEDGER.md) as supplementary bounded
evidence and is not pooled into the paper's inferential estimates.

Audit-A4 verification and the remaining author-owned submission tasks are recorded in
[`IDENTIFIABILITY_AUDIT_SUBMISSION_CHECKLIST.md`](IDENTIFIABILITY_AUDIT_SUBMISSION_CHECKLIST.md).
