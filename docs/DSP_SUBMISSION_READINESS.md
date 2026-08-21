# Digital Signal Processing submission readiness: CARE-ASD

Assessment basis: frozen Audit-A4 evidence at
`f1d5f7fadea74de6e9c7fdefcb172962b3298b63`, the BSS Reviewer #2 hardening, and the expanded
prior-art/engineering pass based on repository commit
`8f45752cc4990d8f6a1eed364b33bbfb76e76873`. Target journal: *Digital Signal Processing*
(Elsevier).

## Executive decision

**READY FOR EXTERNAL EXPERT REVIEW**

The paper is not ready for journal assembly or submission. It is ready for an independent
BSS/multichannel-DSP expert to review the formal statements, residual novelty, and positioning.
The expanded audit found no contradiction with the frozen empirical result, but it rejected broad
audit-method novelty: desired-signal leakage, component-separated attenuation/distortion,
downstream utility, risk--coverage, and holdout evaluation are prior art. The strongest contribution
is now the frozen ASD-specific empirical component-safety/efficacy boundary; the operational audit
is a possible differentiator, and the formal result is supporting rationale.

No independent expert approval is claimed.

## Contribution assessment

| Dimension | Score (1–10) | Assessment |
|---|---:|---|
| Problem significance | 8 | A contaminated auxiliary reference is a real DSP problem, and unseen-fault preservation makes normal-only ASD unusually consequential. |
| Methodological novelty | 5 | Individual audit components and several combinations are prior art. Only their joint ASD-specific decision contract remains a possible differentiator. |
| Theoretical grounding | 5 | The physical model and bounded logic clarify scope, but Lemma 1 is elementary and Proposition 1 is conditional/no-free-lunch-like rather than a new BSS theorem. |
| Experimental validity | 7 | Frozen comparisons, controlled components, explicit selector outcomes, and untouched AP-CARE evidence are strengths; intervention and proxy scope is narrow. |
| Statistical validity | 6 | Paired file-level inference matches the frozen estimand but is conditional on one seed and omits acquisition-cluster and machine-population uncertainty. |
| Reproducibility | 9 | Immutable evidence, hashes, decision chronology, exact frozen sensitivity, and executable validation provide strong traceability. |
| External validity | 4 | Evidence is development-only, with two real synchronized types, one detector family, and synthetic fault proxies. |
| DSP scope fit | 8 | The paper now foregrounds a signal-processing safety/efficacy decision rather than a failed leaderboard method; fit depends on expert judgment of the integrated contribution. |

## Critical remaining issues

1. **Independent expert review.** A BSS/multichannel-DSP expert must assess Lemma 1, Proposition 1,
   missing assumptions/references, and whether the joint audit adds enough beyond good experimental
   practice. Use `BSS_EXPERT_REVIEW_PACKET.md`; no sign-off has occurred.
2. **Residual novelty risk.** The strongest rejection argument is that the theory is elementary or
   conditional while the audit ingredients are prior art. If a closer complete framework is found,
   the methodology claim must be removed and the submission reconsidered as an empirical case study.
3. **Conditional-inference author sign-off.** Authors must accept the one-seed, file-exchangeability,
   missing acquisition-cluster, seven-machine, proxy-fault, and development-only limits recorded in
   `DSP_STATISTICS_EXTERNAL_VALIDITY_SIGNOFF.md`.
4. **Bibliography completion.** Publisher/official metadata was checked for the core prior art, but
   final bibliography import, challenge-report status, and a human expert's missing-reference check
   remain open.
5. **Journal production.** Elsevier LaTeX assembly, final tables/figures, declarations, author
   contributions, data/code statement, and line-level copy editing must wait until expert review.

## Prior-art decision

The closest framework found is Ivry, Cohen, and Berdugo (Interspeech 2022), which separately
measures desired-speech maintenance and residual-echo suppression and analyzes their trade-off.
BSS Eval, P.835 evaluation, distortion-weighted multichannel filtering, enhancement-error
decomposition plus ASR, the AEC Challenge, and selective prediction further establish most audit
ingredients. Details and verified metadata are in `DSP_NOVELTY_PRIOR_ART_AUDIT.md`.

Permitted novelty wording:

> In the sources reviewed, we did not identify a normal-only ASD study that jointly operationalizes
> a locked downstream comparator, known nuisance and fault-proxy components, non-substitutable
> attenuation and retention, selector false-safe risk with coverage, frozen stop rules, and an
> untouched-holdout safe-use decision.

This is a bounded source result, not “first,” “no previous work,” or universal validation.

## Formal-claim status

- Lemma 1 remains an unstructured instantaneous/local-frequency-bin latent-coordinate result only.
- Proposition 1 remains a bounded uniform-certification limitation over the declared future-fault
  extension class.
- Proposition 1 is now explicitly independent of Lemma 1 and described as a conditional,
  application-specific no-free-lunch rationale.
- No physical source-label non-identifiability or general convolutional BSS impossibility is claimed.
- The formal results are not positioned as the primary novelty.

## Statistics and external validity

The required author table is `DSP_STATISTICS_EXTERNAL_VALIDITY_SIGNOFF.md`. It documents what is and
is not estimated under seed 13711, file-level paired resampling, missing acquisition-session
clusters, seven development types, two real synchronized types, one detector, synthetic proxies,
development-only analysis, post-hoc \(\Omega\), and the descriptive real/emulated split. No
retrospective seeds, models, thresholds, or subgroup searches are authorized.

## Type-check repair

The package metadata supports Python `>=3.11,<3.14`, README documents Python 3.11+ and 3.12 support,
and Ruff targets the lowest supported syntax (`py311`). The old mypy configuration forced Python
3.11 semantics while importing the active Python 3.12 NumPy stubs, causing an internal stub parse
failure. Aligning mypy with Python 3.12 exposed 43 genuine baseline typing errors across 16 files.

The repair removes the fixed mypy-version override so mypy checks against its running supported
interpreter, and applies targeted boundary types/casts, protocols for the dynamic BEATs interface,
public PyTorch AMP API typing, one exact Pydantic numeric constraint, and explicit nested-dictionary
typing. It does not weaken strict mode, add broad ignore suppressions, change dependencies, or alter
scientific algorithms. Final validation results are recorded below.

## Frozen-evidence integrity

Audit-A4 is immutable. This pass creates manuscript-supporting documents and targeted typing edits
only. It does not write to frozen predictions, Audit-A4 reports, or the frozen DCASE-sensitivity
package. No model training, additional seed, threshold tuning, evaluation-label access, new
selector, new intervention, or V11+ method occurred.

Final validation status:

- full tests: **240/240 passed on Python 3.11 and 240/240 passed again on Python 3.12**, each in
  five hidden batches (36 + 84 + 56 + 26 + 38) to avoid the local session watchdog;
- lint: **passed** (`ruff check .`, Python 3.12 environment);
- type-check: **passed on Python 3.11 and Python 3.12** (`mypy src`, 73 source files, strict mode);
- Audit-A4 regeneration: **23/23 artifact hashes and 14/14 frozen source hashes matched**;
- frozen DCASE sensitivity regeneration: **3/3 input hashes and 4/4 output hashes matched**;
- path-scoped Git diff from reviewed commit `8f45752` over the frozen configs/packages: **empty**.

All regenerations used new output paths. The repository-local temporary sensitivity output was
removed after comparison and is reproducible from the frozen generator; no committed artifact was
overwritten.

## New analyses

| Analysis | Source inputs | Frozen inputs? | Chronological status | Tuning performed? |
|---|---|---:|---|---:|
| Expanded novelty/prior-art audit | Publisher/DOI/official primary records plus existing audit matrix | Audit evidence not recomputed | New pre-submission positioning analysis; not exhaustive | No |
| Contribution/title red team | Hardened manuscript, formalization, and prior-art audit | No new empirical data | New editorial/scientific hardening | No |
| BSS expert review packet | Existing scoped Lemma 1 and Proposition 1 | No new empirical data | Review aid only; no approval claimed | No |
| Statistics/external-validity sign-off table | Existing frozen design and manuscript limitations | Yes, as documented design facts | Author-governance document; no new statistic | No |
| Type-check repair | Package metadata, active stubs, and existing source annotations | No scientific artifact | Engineering maintenance only | No |

The exact DCASE \(\Omega\) and real/emulated summaries remain derived post hoc from frozen
predictions as documented previously. This pass computes no new scientific endpoint.

## Recommended titles

1. *Auditing Safe Removability from a Contaminated Far-Microphone Reference in Normal-Only Anomalous Sound Detection*
2. *A Component-Aware Safety Audit of Contaminated-Reference Processing for Normal-Only Anomalous Sound Detection*
3. *Utility Is Not Safe Removability: A Decision-Oriented Audit for Far-Microphone ASD Preprocessing*
4. *A Frozen Safety--Efficacy Boundary for Far-Reference Preprocessing in DCASE-2026-Aligned Anomalous Sound Detection*
5. *When the Noise Reference Contains the Machine: Identifiability and Safety Limits in Normal-Only Anomalous Sound Detection* — defensible only with immediate scope qualifiers; not recommended because “Identifiability” over-promises.

## Contribution bullets

- Using frozen DCASE-2026-aligned development evidence, we establish a bounded empirical component-safety/efficacy result: the tested broad and conservative interventions did not establish improved ASD or a reliable joint safe-use region, conditional on the locked comparator and tested component families.

- We operationalize an ASD-specific decision audit that jointly requires fixed-comparator utility, known-component environmental attenuation and fault retention, selector false-safe risk and coverage, and prospective holdout decisions; its individual ingredients are acknowledged prior art.

- We provide a scoped supporting rationale that separates elementary local-bin latent-coordinate non-uniqueness from a conditional limitation on uniform normal-only fault-safety certification, without claiming physical source-label or general convolutional BSS impossibility.

## Next authorized step

Send the manuscript, `DSP_NOVELTY_PRIOR_ART_AUDIT.md`, and `BSS_EXPERT_REVIEW_PACKET.md` to an
independent BSS/multichannel-DSP expert. Do not assemble Elsevier LaTeX or reopen experiments before
that review returns.
