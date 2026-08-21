# CARE-ASD Audit-A4 reproducibility and submission checklist

Status date: 2026-08-21  
Target venue: *Digital Signal Processing* (Elsevier)  
Technical reproducibility gate: **passed**  
Scientific framing gate: **ready for internal review**
Submission gate: **pending formal review, bibliography verification, author metadata, venue
formatting, and final author approval**

This checklist closes the technical part of Audit-A4 without authorizing new training,
evaluation-label access, or post-failure tuning. The paper remains an empirical identifiability
audit under the tested assumptions, not a successful cancellation method or an impossibility
theorem.

## 1. Frozen package verification

All four packages were regenerated on 2026-08-21 from a clean `research/fp-naa` worktree at
commit `0d62f07` and sealed in the verified Audit-A4 repository state
`f1d5f7fadea74de6e9c7fdefcb172962b3298b63`. Regeneration used new output directories, so the
immutable committed packages were never overwritten. Every stable `run.json` field matched after
excluding only `git_commit`, which correctly records the commit at regeneration time.

| Package | Contract | Committed package | Frozen inputs | Output artifacts | Result |
|---|---|---|---:|---:|---|
| Audit-A0 paper synthesis | `configs/experiment/audit_paper_v1.yaml` | `reports/audit/care_asd_identifiability_audit_v1/` | 9 | 8 | Exact hash match |
| Audit-A1 literature boundary | `configs/research/audit_literature_v1.yaml` | `reports/audit/care_asd_literature_audit_v1/` | 13 cited primary sources | 3 | Exact hash match |
| Audit-A2 robustness appendix | `configs/experiment/audit_robustness_v1.yaml` | `reports/audit/care_asd_robustness_appendix_v1/` | 5 | 9 | Exact hash match |
| Post-V10 literature delta | `configs/research/fp_naa_post_v10_literature_delta.yaml` | `reports/audit/fp_naa_post_v10_literature_delta/` | 9 cited primary sources | 3 | Exact hash match |

Combined result: **23/23 generated artifact hashes matched**. The two quantitative packages also
matched all **14/14 frozen source hashes**. Text artifacts use LF-normalized UTF-8 hashes; binary
artifacts use raw-byte SHA-256.

The focused detached regression suite passed:

```text
29 passed in 7.60s
```

It covered audit synthesis, literature audit, robustness analysis, and their CLI surfaces. The
persisted local diagnostic directory is outside the repository and is not part of the scientific
record.

After the DSP manuscript reframing, the same four packages were regenerated again into new
temporary directories. The result remained **23/23 exact artifact hashes**, with quantitative
source hashes, coverage, and decision fields unchanged. A focused suite covering Audit-A4, the
exact DCASE metric, the new frozen sensitivity package, and CLI regression passed:

```text
32 passed in 10.23s
```

A path-scoped Git comparison against `f1d5f7f` confirmed that no frozen Audit-A4 source or output
artifact changed.

## 2. Reproduction commands

Each output path below must be a new, nonexistent directory because the generators deliberately
refuse to overwrite an earlier result.

```bash
uv sync --extra dev

uv run care-asd audit synthesize \
  --config configs/experiment/audit_paper_v1.yaml \
  --repo-root . \
  --output-dir <new-audit-a0-output>

uv run care-asd audit literature \
  --config configs/research/audit_literature_v1.yaml \
  --repo-root . \
  --output-dir <new-audit-a1-output>

uv run care-asd audit robustness \
  --config configs/experiment/audit_robustness_v1.yaml \
  --repo-root . \
  --output-dir <new-audit-a2-output>

uv run care-asd audit literature \
  --config configs/research/fp_naa_post_v10_literature_delta.yaml \
  --repo-root . \
  --output-dir <new-post-v10-output>

uv run python scripts/run_dsp_frozen_sensitivity.py \
  --config configs/research/dsp_frozen_sensitivity_v1.yaml \
  --output-dir <new-dsp-sensitivity-output>
```

Compare each regenerated `run.json` with its committed counterpart. `git_commit` may differ;
`artifacts`, source/config hashes, analysis settings, coverage, decisions, and claim counts must
match exactly.

## 3. Claim-to-artifact traceability

| Manuscript claim group | Authoritative artifact | Manuscript location |
|---|---|---|
| B00/B01/B02 AUC, pAUC, paired intervals | `care_asd_identifiability_audit_v1/performance_evidence.csv` | Abstract; Sections 4.3, 5.1–5.3 |
| Machine/domain sensitivity and LOMO ranges | `care_asd_robustness_appendix_v1/heterogeneity.json` and `leave_one_machine_out.csv` | Sections 5.2–5.3 |
| Phase 8 displacement/score association | `care_asd_identifiability_audit_v1/decision.json` and frozen Phase 8 source hash | Sections 4.4 and 5.4 |
| SAFE-REF false-safe behavior | `care_asd_identifiability_audit_v1/decision.json` and `identifiability_evidence.csv` | Sections 4.4–4.5 and 5.5 |
| AP-CARE retention/attenuation gate | `identifiability_evidence.csv`, `ap_care_holdout_diagnostics.csv`, and `decision.json` | Sections 4.4–4.5 and 5.6 |
| Exact DCASE and real/emulated sensitivity | `care_asd_dsp_frozen_sensitivity_v1/official_metric_summary.csv`, `run.json`, and `manifest.json` | Section 5.7; post hoc only |
| Literature novelty and prohibited claims | `care_asd_literature_audit_v1/claim_boundary.json` | Sections 1, 6, 7, and provisional references |
| FP-NAA V1-V10 bounded negative sequence | `docs/FP_NAA_NEGATIVE_EVIDENCE_LEDGER.md` and immutable server run IDs | Sections 4.5 and 6.5 only; not pooled |
| Stop and leakage decisions | `care_asd_identifiability_audit_v1/decision.json` | Sections 4.5, 6, 7, and reproducibility statement |

## 4. Environment and data boundary

- [x] Core dependency resolution is frozen in `uv.lock`.
- [x] The SERVER-02 FP-NAA environment is pinned by
  `environments/fp-naa-cu118.yml` and `requirements/fp-naa-cu118.lock.txt`.
- [x] The paper generators use committed derived evidence and do not require raw audio.
- [x] Restricted DCASE audio is not redistributed.
- [x] No unseen evaluation labels or scores were accessed.
- [x] The original SAFE-REF result and subsequent correction are both retained; the corrected study
  is labeled post-correction and is not retroactively described as the original preregistration.
- [x] The failed AP-CARE holdout was not reused for tuning.
- [x] GPU replication and successor stages remain prohibited by the frozen decision.
- [x] Code and generated-artifact text hashes are portable across LF/CRLF checkouts.
- [x] Repository code license is MIT.

## 5. Scientific-content checklist

- [x] B01 AUC is retained as the registered primary endpoint; harmful pAUC is labeled as a
  prospectively specified secondary result.
- [x] Null B02 result retained without interpreting interval overlap as equivalence.
- [x] Simple aggregate subtraction is distinguished from the paired-bootstrap estimand.
- [x] Source/target asymmetry and machine heterogeneity are reported.
- [x] Phase 8 is labeled post-hoc and non-causal.
- [x] Known-component safety endpoints are separated from ASD task metrics.
- [x] Positive dual-microphone systems are acknowledged as counterexamples to a generic negative
  claim.
- [x] DCASE technical reports and arXiv papers are labeled non-peer-reviewed where applicable.
- [x] FP-NAA V1-V10 is supplementary process evidence, not ten independent replications.
- [x] Limitations exclude unseen-machine, deployment, and universal-impossibility claims.
- [x] An instantaneous/local-bin latent-factor lemma and a conditional normal-only safety-
  certification proposition are separated from physical source semantics, empirical selector
  failure, and ASD degradation.
- [x] Paired-bootstrap inference is explicitly conditional on the single locked training seed
  13711 and does not claim training-initialization uncertainty.
- [x] Exact DCASE harmonic and real/emulated results are labeled “derived post hoc from frozen
  predictions/artifacts” and do not replace the frozen estimand.
- [x] The SAFE-REF 20% coverage criterion is reported as unchanged; the later 8.84% feasibility
  calculation is labeled a post-result diagnosis rather than a gate modification.

## 6. Items requiring author action before submission

- [ ] Confirm author order, affiliations, corresponding author, institutional emails, and ORCID
  identifiers.
- [ ] Replace the collective author placeholder in `CITATION.cff` and `pyproject.toml`.
- [ ] Confirm the final title with all authors.
- [ ] Obtain signal-processing review of the latent-factor lemma, the decision-theoretic Proposition
  1, their semantic/convolutional boundaries, and their relationship to established blind source
  separation identifiability results.
- [ ] Obtain explicit author sign-off on the SAFE-REF chronology, endpoint hierarchy, and
  single-seed claim boundary.
- [ ] Convert the Markdown draft into the current *Digital Signal Processing* manuscript template.
- [ ] Build a formal bibliography and verify every DOI, author list, year, and access date.
- [ ] Check the venue's current word/page limits, figure limits, graphical-abstract requirements,
  and double-anonymous policy at the time of submission.
- [ ] Export figures in the venue-required vector/raster formats and verify font embedding,
  grayscale readability, and color accessibility.
- [ ] Add author contributions using the venue's required CRediT taxonomy.
- [ ] Add funding, acknowledgments, conflicts of interest, ethics/data-use statements, and any
  institutional declarations supplied by the authors.
- [ ] Decide whether the code snapshot will receive a release tag and archival DOI; insert the
  immutable identifier into the manuscript if created.
- [ ] Run a final similarity/plagiarism review and language edit without changing scientific
  claims or frozen numbers.
- [ ] Obtain explicit approval of the final manuscript from every author.

## 7. Final release gate

The technical bundle is reproducible and the revised scientific story is **ready for internal
review**, not yet for journal upload. The manuscript remains blocked from external submission until
the critical review and author-action items are resolved. None of those actions permits changing
the frozen evidence, reopening V10, adding retrospective seeds, or accessing evaluation labels.
