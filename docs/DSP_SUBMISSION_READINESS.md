# Digital Signal Processing submission readiness: CARE-ASD

Assessment basis: frozen Audit-A4 evidence at
`f1d5f7fadea74de6e9c7fdefcb172962b3298b63`, the repository reviewed at
`0875bdadd30e37b3a053d4bf0e24ebea854a3326`, the isolated post-hoc frozen-sensitivity package, and
the subsequent Reviewer #2 scientific hardening pass. Target journal: *Digital Signal Processing*
(Elsevier).

## Executive decision

**READY FOR INTERNAL REVIEW**

The manuscript is ready for an independent BSS/multichannel DSP expert and author-team review. It
is not ready for journal submission. The former Proposition 1 did not survive unchanged: its
physical semantic implication was unsupported, and its convolutional scope was too broad. The
hardened version now separates a modest instantaneous/local-bin latent-factor lemma from a
conditional decision-theoretic fault-safety proposition. The empirical contribution remains intact
and bounded. No additional model search is justified.

## Contribution assessment

| Dimension | Score (1–10) | Assessment |
|---|---:|---|
| Problem significance | 8 | Contaminated auxiliary references are a practical DSP risk, and normal-only ASD makes preservation of unseen fault evidence difficult to certify. |
| Methodological novelty | 6 | The joint decision contract is a credible operational synthesis, but its individual comparator, component, selector, holdout, and frozen-rule elements are mostly established practice. |
| Theoretical grounding | 6 | The physical observation model, scoped latent lemma, and conditional non-certifiability proposition are defensible after weakening; they are not a general physical or convolutional BSS theorem and need expert sign-off. |
| Experimental validity | 7 | Pairing, frozen gates, controlled components, and the untouched AP-CARE internal holdout are strengths. One detector, synthetic proxies, and SAFE-REF's post-correction status constrain the result. |
| Statistical validity | 6 | Paired file-level stratified bootstrap matches the frozen estimand, but assumes within-stratum file exchangeability and omits training-initialization, acquisition-cluster, and machine-population uncertainty. |
| Reproducibility | 9 | Audit-A4 hashes, immutable outputs, frozen stop rules, and isolated deterministic post-hoc analysis provide unusually strong traceability. |
| External validity | 4 | Evidence is development-only, includes two real synchronized and five emulated types, uses one detector family and synthetic fault proxies, and does not cover unseen evaluation machines. |
| DSP scope fit | 8 | The paper now centers a contaminated-reference decision problem, bounded formal reasoning, and component-aware signal-processing evaluation rather than leaderboard optimization. |

## Critical remaining issues

1. **Independent BSS/theory sign-off.** A source-separation expert must verify that Lemma 1 remains
   an unstructured latent-coordinate result, Proposition 1 remains conditional on the declared
   future-fault class, and no phrase reintroduces physical semantic or general convolutional claims.
2. **Theory-literature and bibliography audit.** Audit-A1 is a bounded direct-ASD matrix, not an
   exhaustive ICA/BSS, multichannel cancellation, or decision-safety review. Refresh the search and
   manually verify all author/title/venue/year/pages/DOI/publication-status metadata against
   publisher or official records.
3. **Conditional-inference sign-off.** The authors must explicitly accept that the study does not
   quantify training-seed variability, unrecorded acquisition clustering, or population variation
   over machine types. These are submission risks and cannot be repaired by retrospective runs.
4. **External-validity sign-off.** The conclusions must remain limited to the locked detector,
   development machines, tested selectors/controllers, and controlled fault-proxy families.
5. **Journal manuscript assembly.** Convert the Markdown draft to Elsevier-compatible source,
   finalize equations, figures, tables, bibliography, declarations, and author-level consistency
   checks against frozen artifacts.
6. **Repository type-check repair.** The README command `uv run mypy src` does not currently pass:
   with the locked active environment it stops on a Python-3.12 NumPy stub while the mypy target is
   configured as 3.11; when explicitly aligned to Python 3.12 it exposes 43 existing source errors
   across 16 files. This hardening pass changed no Python, configuration, or dependency file, so the
   failure is baseline engineering debt rather than a manuscript-induced regression. It must be
   repaired or the validation claim revised before submission.

SAFE-REF chronology, endpoint hierarchy, DCASE metric status, real/emulated labeling, and V1–V10
epistemic role are now explicitly resolved in the manuscript; final editing must preserve them.

## Optional improvements

- Add a compact diagram separating the physical convolutional model, local-bin formal result, and
  three evidence levels, provided it does not imply additional proof.
- Move detailed V1–V10 history and large robustness tables to supplementary material.
- Ask an independent reviewer to regenerate Audit-A4 and the frozen-sensitivity package from a
  clean clone and compare manifests.
- Add a concise notation table and define every attenuation/retention sign convention in the final
  manuscript source.
- Refresh the target-journal author checklist immediately before submission because requirements
  can change.

## Frozen-evidence integrity

Audit-A4 is the evidence boundary. This hardening pass edits only manuscript/supporting analysis
documents and creates one reviewer-attack document. It does not alter the Audit-A4 source or output
packages and does not overwrite the isolated post-hoc sensitivity package.

Validation status for the completed hardening pass:

- Audit-A4 regenerated into four new temporary directories: **23/23 artifact hashes matched**;
- frozen quantitative sources: **14/14 source hashes matched**;
- path-scoped Git diff against `f1d5f7f` for all Audit-A4 configs/packages: **empty**;
- isolated DCASE sensitivity regeneration: **3/3 input and 4/4 output hashes matched**;
- all 50 test files: **240/240 tests passed** in four hidden batches (81 + 75 + 44 + 40); a
  single-process run was interrupted by the local session watchdog after 177 passes and is not
  counted as a successful run;
- lint: **passed** (`ruff check .`);
- type-check: **failed on pre-existing repository issues**, as described in Critical issue 6.

No frozen artifact was overwritten. Temporary regeneration directories were outside the committed
scientific packages, and the one repository-local sensitivity validation directory was removed
after its hashes were compared.

## New analyses

| Analysis | Source inputs | Frozen inputs? | Chronological status | Tuning performed? |
|---|---|---:|---|---:|
| Reviewer #2 mathematical attack | Existing formalization and manuscript; analytic model-class review | No empirical data | New manuscript-level hardening; found and corrected a semantic/convolutional overclaim | No |
| Latent-factor lemma and uniform-certification proposition | Analytic instantaneous/local-bin model plus two future-fault extensions identical on normal evidence | Not an empirical dataset | New theoretical clarification; not preregistered and not an empirical result | No |
| Exact DCASE 2026 harmonic metric | Frozen B00/B01/B02 1,400-row development score CSVs and existing exact scorer | Yes; hashes recorded | Derived post hoc from frozen predictions/artifacts; secondary descriptive analysis | No |
| Real versus emulated machine sensitivity | Same frozen score CSVs; repository-fixed ToyCar/fan versus five `*Emu` split | Yes; hashes recorded | Post-hoc descriptive sensitivity; not preregistered subgroup inference | No |

No new training, seed search, threshold change, favorable-subset search, evaluation-label access,
new B-series intervention, or V-series method was performed.

## Recommended title

1. *When the Noise Reference Contains the Machine: A Controlled Safety Audit for Normal-Only Anomalous Sound Detection*
2. *Auditing Contaminated-Reference Preprocessing for Normal-Only Anomalous Sound Detection*
3. *Utility Is Not Safe Removability: A Dual-Microphone Audit for Anomalous Sound Detection*

## Contribution bullets

- We formalize contaminated-reference ASD while separating exact latent-factor non-uniqueness in an unstructured local model from a conditional limitation on uniform normal-only fault-safety certification.

- We specify a decision-oriented audit that jointly evaluates controlled-comparator utility, known-component environmental attenuation and fault retention, selector false-safe behavior and coverage, and frozen holdout decisions.

- Using frozen DCASE-2026-aligned development evidence, we show that the tested broad and conservative interventions did not establish improved ASD or a reliable joint safe-use region, conditional on the locked comparator and tested component families.
