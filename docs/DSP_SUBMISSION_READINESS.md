# Digital Signal Processing submission readiness: CARE-ASD

Assessment basis: manuscript and repository state prepared from frozen Audit-A4 commit
`f1d5f7fadea74de6e9c7fdefcb172962b3298b63`, plus the isolated post-hoc sensitivity package
`care_asd_dsp_frozen_sensitivity_v1`. Target journal: *Digital Signal Processing* (Elsevier).

## Executive decision

**READY FOR INTERNAL REVIEW**

The work now has a defensible journal contribution: a bounded signal-model argument, a reusable
controlled contaminated-reference safety-audit methodology, and a frozen empirical
safety/efficacy boundary. It is not ready for external submission until the formal proposition,
SAFE-REF chronology, statistical endpoint hierarchy, and bibliography have received author and
supervisor review and the Markdown draft has been converted into the journal manuscript format.
No additional model search is justified.

## Contribution assessment

| Dimension | Score (1–10) | Assessment |
|---|---:|---|
| Problem significance | 8 | Contaminated auxiliary references are a practical DSP risk, and ASD makes desired-signal preservation unusually hard to verify. |
| Methodological novelty | 7 | The joint fixed-comparator, known-component, selector-safety, and frozen-stop audit is a credible bounded contribution; the individual denoisers are not novel positive methods. |
| Theoretical grounding | 6 | The two-source/two-microphone model and constructive factorization ambiguity supply the missing layer, but the proposition still needs specialist mathematical review and careful placement relative to blind source separation theory. |
| Experimental validity | 7 | Pairing, frozen gates, controlled components, and untouched AP-CARE holdout are strengths. Synthetic fault proxies, SAFE-REF's post-correction status, and one detector family constrain the claim. |
| Statistical validity | 6 | Paired stratified bootstrap and frozen endpoints are appropriate for evaluation-observation uncertainty. One training realization and a four-case AP-CARE matched subset limit inference. |
| Reproducibility | 9 | Audit-A4 regeneration, hashes, immutable outputs, stop rules, and the new isolated deterministic sensitivity package provide unusually strong traceability. |
| External validity | 4 | Evidence is development-only, includes only two real synchronized machine types, uses synthetic fault proxies, and does not cover unseen evaluation machines or modern detector families. |
| DSP scope fit | 8 | The revised story centers signal-mixture non-uniqueness, contaminated-reference safety, and evaluation methodology rather than leaderboard optimization. |

## Critical remaining issues

1. **Formal review.** A signal-processing coauthor or supervisor should verify the bounded
   non-uniqueness proposition, its model class, and its relationship to established blind source
   separation/identifiability results. The paper must not let the proposition read as universal
   impossibility.
2. **Chronology sign-off.** The authors must confirm the SAFE-REF timeline from commits and run IDs,
   especially that the corrected protocol followed the first outcome and that the 20% criterion was
   never changed to rescue the gate. This disclosure must survive final editing.
3. **Endpoint and seed sign-off.** Abstract, Results, and conclusions must retain AUC as the
   registered primary endpoint, pAUC as a prospectively specified secondary endpoint, and all
   B00/B01/B02 inference as conditional on seed 13711.
4. **Bibliography verification.** Manually verify every title, author list, DOI, venue, year,
   pagination, URL, and publication status from publisher/official records; refresh the bounded
   search before submission and keep challenge reports/preprints labeled correctly.
5. **Journal manuscript assembly.** Convert the internal Markdown into Elsevier-compatible
   manuscript source, number equations/figures/tables/references, include complete citations and
   declarations, and perform an author-level consistency review against the frozen artifacts.

These issues require review and packaging, not retraining or outcome-driven analysis.

## Optional improvements

- Add a compact diagram of the contaminated two-source/two-microphone model and the four audit
  layers if it improves reviewer comprehension without implying extra evidence.
- Move detailed V1-V10 history and large robustness tables to supplementary material.
- Ask an independent internal reviewer to reproduce the official-score sensitivity directory from
  a clean clone and compare its manifest.
- Add a concise notation table and define every energy/attenuation convention in the final LaTeX
  source.
- Refresh the target-journal author checklist immediately before submission because requirements
  can change.

## Frozen-evidence integrity

Audit-A4 is the evidence boundary. Final verification after this revision found no path-scoped Git
diff against `f1d5f7f` in the frozen Audit-A4 source or output directories. All four packages were
regenerated into isolated temporary directories: **23/23 generated artifact hashes** matched, and
the quantitative source hashes, coverage fields, and decisions matched. The focused regression
suite, including the new DCASE sensitivity tests, passed **32/32 tests**. The new package's config,
code, three frozen score inputs, and four manifest-listed outputs also matched their recorded
SHA-256 hashes.

The only new empirical output directory is
`reports/audit/care_asd_dsp_frozen_sensitivity_v1/`; it does not overwrite or rename frozen
evidence.

## New analyses

| Analysis | Source inputs | Frozen inputs? | Chronological status | Tuning performed? |
|---|---|---:|---|---:|
| Bounded source-label non-uniqueness | Analytic two-source/two-observation model and explicit invertible source transformation | Not an empirical dataset | New theoretical formalization; scoped to stated weak assumptions | No |
| Exact DCASE 2026 harmonic metric | Frozen B00/B01/B02 1,400-row development score CSVs and existing exact scorer | Yes; input hashes recorded | Derived post hoc from frozen predictions/artifacts; secondary descriptive analysis | No |
| Real versus emulated machine sensitivity | Same frozen score CSVs; repository-fixed split ToyCar/fan versus five `*Emu` types | Yes; input hashes recorded | Post-hoc descriptive sensitivity; not preregistered subgroup inference | No |

No new training, seed search, threshold change, favorable-subset search, evaluation-label access, or
new V-series method was performed.

## Recommended title

1. *When the Noise Reference Contains the Machine: Identifiability and Safety Limits in Normal-Only Anomalous Sound Detection*
2. *Auditing Contaminated-Reference Preprocessing for Normal-Only Anomalous Sound Detection*
3. *A Controlled Safety and Identifiability Audit of Dual-Microphone Preprocessing for Anomalous Sound Detection*

## Contribution bullets

- We formulate dual-microphone ASD with a contaminated auxiliary reference and give a bounded
  non-uniqueness argument showing why reference-correlated energy is not automatically identifiable
  as safely removable noise without additional structural assumptions.
- We introduce a controlled audit protocol that jointly evaluates fixed-comparator ASD utility,
  known-component environmental attenuation and fault retention, selector false-safe behavior, and
  frozen prospective stop decisions.
- Using frozen DCASE-2026-aligned development evidence, we show that the tested broad suppression
  and conservative gating strategies did not establish an improved, reliably safe operating region;
  the conclusion is conditional on the locked comparator and bounded to the tested controlled
  component family and machine setting.
