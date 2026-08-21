# Digital Signal Processing reviewer red team: current status

Review basis: frozen Audit-A4 evidence, the completed BSS Reviewer #2 hardening, and the expanded
prior-art audit completed 21 August 2026. Historical objections remain visible only where they help
prevent a regression in claim scope.

## Current decision

**READY FOR EXTERNAL EXPERT REVIEW**

This means the manuscript and concise review packet are ready to send to an independent
BSS/multichannel-DSP expert. It does not mean expert approval, journal assembly, or submission
readiness. The broad audit-method novelty claim was not sustained: only an ASD-specific operational
integration remains a possible differentiator. The formal result is supporting rationale rather
than primary novelty.

Status meanings:

- **RESOLVED:** the present manuscript/supporting document answers the objection at its stated scope.
- **PARTIALLY RESOLVED:** wording and scope are corrected, but current evidence cannot close the
  broader scientific question.
- **OPEN:** a material item still requires independent expert, author, or production action.

## Current reviewer-status table

| Claim or issue | Best objection | Is objection valid? | Required response/fix | Status |
|---|---|---|---|---|
| Novelty of the identifiability formulation | Lemma 1 is elementary change-of-coordinates algebra; Proposition 1 is a conditional no-free-lunch result, not new BSS identifiability theory. | Yes. | Treat both as scoped rationale, not primary novelty; state their logical independence and obtain expert review. Manuscript Secs. 1, 2.2, 6.1; formalization Secs. 5–6; `BSS_EXPERT_REVIEW_PACKET.md`. | **PARTIALLY RESOLVED** |
| Semantic admissibility | A transformed latent coordinate such as \(m'=m+\alpha e\) is not thereby a physically machine-origin source. | Yes; this invalidated the old physical-label reading. | Keep Lemma 1 inside the unstructured instantaneous/local-bin latent class. Manuscript Secs. 2.2 and 7; formalization Secs. 2 and 4. | **RESOLVED** |
| Convolutional scope | Abstract invertibility of \(T(z)\) does not ensure a causal stable inverse or physically admissible transformed filters. | Yes. | No general convolutional theorem; the time-domain model motivates the problem and the lemma is local-bin only. Manuscript Sec. 2.2; formalization Sec. 1. | **RESOLVED** |
| Proposition 1 validity | Its adverse extension is assumed into the fault class; the conclusion may be tautological and gives no probability of harm. | Yes. | Retain the bounded uniform-certificate conclusion and state exactly what is assumed/not proved. Independent expert must judge whether it belongs in the main paper. Manuscript Secs. 2.2 and 7; expert packet Sec. 4. | **OPEN** |
| Novelty of the audit protocol | Component metrics, attenuation--distortion trade-offs, downstream utility, risk--coverage, and holdouts are established practice. | Yes; the broad novelty claim failed. | Claim only the ASD-specific joint decision contract; cite close prior art and allow removal if expert review finds an equivalent complete framework. Manuscript Secs. 1 and 3; `DSP_NOVELTY_PRIOR_ART_AUDIT.md`. | **PARTIALLY RESOLVED** |
| Audit is merely good experimental practice | Combining familiar checks does not automatically create a publishable methodology. | Yes; unresolved editorial judgment. | Lead with frozen empirical boundary; present integration as a possible differentiator, not a universal framework. Novelty audit, “Recommended contribution hierarchy.” | **OPEN** |
| B01 is a straw man | Broad residual replacement is not modern BSS. | Yes if generalized; not valid against its declared stress-test role. | Keep B01 as one broad stress test only. Manuscript Secs. 4.2, 5.2, and 6.1. | **RESOLVED** |
| B02 represents conservative methods | One near-primary gated interface cannot represent all conservative processors. | Yes. | Call it one capacity-controlled intervention, not the class. Manuscript Secs. 4.2, 5.3, and 6.1. | **RESOLVED** |
| SAFE-REF generalizes to selectors | One synthetic generator and selector cannot rule out all normal-only selectors. | Yes. | Bound false-safe/coverage findings to the tested observables, selector, and controlled family. Manuscript Secs. 5.5, 6.1, and 7. | **RESOLVED** |
| AP-CARE supports the main conclusion | It is post-SAFE-REF, proxy-based, and has only four cases in one matched comparison. | Yes. | Use it as untouched internal mechanism evidence, not an independent replication or general proof. Manuscript Secs. 4.4, 5.6, and 7. | **RESOLVED** |
| Controlled components represent real faults | Generator-known provenance is not physical ground truth across real faults and rooms. | Yes. | Consistently say “fault proxy”; retain external-validity limit. Manuscript Secs. 4.4 and 7; sign-off table. | **RESOLVED** |
| One seed undermines inference | Evaluation bootstrap cannot quantify training-initialization or optimizer variability. | Yes. | Keep inference conditional on seed 13711 and require author sign-off; do not add retrospective seeds. Manuscript Secs. 4.3 and 7; `DSP_STATISTICS_EXTERNAL_VALIDITY_SIGNOFF.md`. | **OPEN** |
| Bootstrap creates pseudo-replication | Files may share acquisition sessions; machine types are not sampled populations. | Partly. Pairing is correct for the frozen file-level estimand, but broader CIs are unsupported. | State exchangeability unit and missing acquisition-cluster/population uncertainty. Manuscript Sec. 4.3 and Limitation 2; sign-off table. | **PARTIALLY RESOLVED** |
| Real/emulated composition limits conclusions | Two real synchronized types cannot support real-machine inference. | Yes. | Keep the split post-hoc descriptive and reject subgroup-benefit claims. Manuscript Sec. 5.7 and Limitation 5. | **RESOLVED** |
| pAUC is presented as primary | Highlighting harmful pAUC could switch endpoints after a null primary AUC. | Valid risk, not a current factual error. | Preserve mean AUC as registered primary and pAUC as prospective secondary. Abstract, Secs. 4.3 and 5.2. | **RESOLVED** |
| Post-hoc DCASE \(\Omega\) is outcome-driven | Adding the official metric after outcomes could be mistaken for confirmatory analysis. | Yes. | Label it derived post hoc from frozen predictions and descriptive; never replace primary inference. Manuscript Sec. 5.7 and Limitation 4. | **RESOLVED** |
| Successful dual-channel systems contradict CARE-ASD | Positive systems show far-channel processing can work. | Not against the bounded claim. | Preserve “auxiliary usefulness” versus “certified safe removability.” Introduction and Discussion Sec. 6.2. | **NOT A VALID OBJECTION** |
| V1–V10 are multiple replications | They are sequential outcome-informed searches and create researcher degrees of freedom. | Yes. | Treat only as research history, narrowing evidence, and stop support. Manuscript Secs. 4.5 and 6.5. | **RESOLVED** |
| SAFE-REF correction compromises prospective claims | First failure was inspected before the corrected split; the record is not pristine preregistration. | Yes. | Preserve chronology and call the result post-correction. Manuscript Sec. 4.5 and Limitation 8; `AP_CARE_V2_EXECUTION_SPEC.md`. | **RESOLVED** |
| Conclusion exceeds the data | Few interventions, one detector, proxy faults, and development-only observations cannot establish a general safety limit. | Yes against the old wording. | Title, Abstract, contributions, Discussion, and Conclusion now center a bounded empirical component-safety/efficacy result. | **RESOLVED** |
| Bibliography establishes novelty | A broad search may still miss a close hearing-aid, ANC, robust-control, or machinery framework. | Yes. | Send `DSP_NOVELTY_PRIOR_ART_AUDIT.md` to the expert; refresh metadata/status at assembly; avoid “first/no prior work.” | **OPEN** |
| Engineering validation is accurate | The earlier report claimed type-check debt. | The old claim was accurate then but is stale after targeted repair. | Mypy now targets its running supported interpreter; 43 errors received targeted typing fixes. Mypy passes on Python 3.11/3.12, 240 tests pass on both, lint passes, and frozen hashes match. `DSP_SUBMISSION_READINESS.md`, Type-check and integrity sections. | **RESOLVED** |

## Strongest remaining rejection argument

The theory is too elementary/conditional to be a primary DSP advance, while most elements of the
audit are established signal-processing and selective-decision practice. Without independent
confirmation that the *joint normal-only ASD decision contract plus frozen empirical boundary* is
a sufficiently coherent and useful contribution, the paper risks being judged a careful negative
case study rather than a journal-level methodological contribution.

## Open blockers before journal assembly

1. Independent BSS/multichannel-DSP review using `BSS_EXPERT_REVIEW_PACKET.md`; approval is not yet
   obtained.
2. Expert challenge to the residual novelty claim and identification of any closer prior framework.
3. Author acceptance of the conditional statistics/external-validity table; no boxes are currently
   signed.
4. Final bibliography import and metadata/status verification, especially challenge reports and
   recent non-peer-reviewed sources.

## Historical objections now closed

- “The manuscript has no formal model” is obsolete: see manuscript Sec. 2 and the formalization.
- “The paper proves physical machine/environment ambiguity” is explicitly withdrawn.
- “The lemma covers arbitrary convolutional filters” is explicitly withdrawn.
- “Exact DCASE metric, real/emulated composition, seed scope, SAFE-REF chronology, and V1–V10 role
  are hidden” is obsolete: see manuscript Secs. 4–7.

## Reviewer recommendation

Send the current manuscript, novelty audit, and expert packet for **external expert review**. Do not
assemble the journal submission until the open formal/novelty questions and author sign-offs are
returned. Do not respond to these risks with new models, seeds, thresholds, or subgroup searches.
