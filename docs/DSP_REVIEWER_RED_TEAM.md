# Digital Signal Processing reviewer red team: current status

Review basis: hardened manuscript following review of commit
`0875bdadd30e37b3a053d4bf0e24ebea854a3326`. Historical objections are retained below but are no
longer phrased as though the formal model, chronology, or endpoint hierarchy were absent.

## Current decision

**READY FOR INTERNAL REVIEW**

This means ready for an independent BSS/multichannel DSP expert and the author team, not ready for
journal submission. The theoretical overclaim found in the earlier proposition has been repaired by
weakening and separation of claims. Physical source-semantic non-identifiability is not proven, the
convolutional theorem has been withdrawn, and irreducible single-seed/external-validity limits remain.

Status meanings:

- **RESOLVED:** the present manuscript/supporting document answers the objection at the claimed scope.
- **PARTIALLY RESOLVED:** wording and scope are corrected, but current evidence cannot close the
  broader scientific question.
- **OPEN:** a material item still requires expert, author, bibliographic, or journal-production work.

## Current reviewer-status table

| Claim or issue | Evidence | Best current reviewer objection | Severity | Status | Where handled / required action |
|---|---|---|---|---|---|
| A formal model supports the paper. | Two-microphone physical observation model; local-bin latent class; Lemma 1; Proposition 1. | The old \(s'=Ts\) argument proved coordinate ambiguity, not that \(m'=m+\alpha e\) remained machine-origin. | Critical | RESOLVED | Manuscript Sec. 2.2 and `IDENTIFIABILITY_FORMALIZATION.md` Secs. 2, 4–6 explicitly concede the semantic limit and replace the original proposition. |
| Proposition 1 is defensible. | Two normal-observationally identical future-fault extensions require different retention decisions. | The result is conditional and nearly no-free-lunch; it does not prove physical normal-source ambiguity or a probability of harm. | Critical | PARTIALLY RESOLVED | Manuscript Secs. 2.2 and 7; formalization Sec. 5. Retain the bounded uniform-certification wording and obtain independent BSS expert sign-off. |
| The theory applies to convolutional acoustic mixing. | Time-domain model motivates the problem. | An arbitrary invertible \(T(z)\) need not have a stable causal inverse or preserve a physically admissible acoustic model. | Critical | RESOLVED | The formal result is narrowed to instantaneous/local-frequency-bin mixing in manuscript Sec. 2.2 and formalization Sec. 1. No general convolutional theorem remains. |
| The paper is consistent with ICA/BSS. | Explicit absence of independence, non-Gaussianity, geometry, paths, source models, and related structure. | Established ICA/BSS identifiability makes an unqualified “unidentifiable” claim false. | Critical | RESOLVED | Manuscript Secs. 2.2 and 6.3; formalization Secs. 2.3 and 8; `DSP_BSS_REVIEWER_ATTACK.md` BSS/ICA section. |
| Far-channel usefulness differs from safe removability. | Positive DCASE systems can fuse or learn from far audio without declaring shared energy nuisance-only. | The paper could otherwise appear to deny successful dual-channel systems. | Critical | RESOLVED | Manuscript Introduction questions 1–2 and Sec. 6.2 explicitly give “potentially yes” versus “not certified by usefulness/correlation.” |
| The audit protocol is a methodological contribution. | Joint decision contract covers comparator utility, known-component efficacy/retention, selector false-safe risk/coverage, and frozen decisions. | These are mostly standard good practices, and no cross-domain validation establishes a universal framework. | Major | PARTIALLY RESOLVED | Manuscript Sec. 3 and `CONTAMINATED_REFERENCE_SAFETY_AUDIT_PROTOCOL.md` Sec. 2 claim operational integration only. Avoid “first” or universal validation. |
| B01 is informative. | Exact locked backend; broad residual replacement; paired frozen effects. | It is a stress test and possible straw man, not representative modern BSS. | Major | RESOLVED | Manuscript Secs. 4.2, 5.2, and 6.1 label its exact role and prohibit algorithm-class generalization. |
| B02 is a conservative comparator. | Near-primary branch, frozen confidence, matched first-projection parameter count. | It changes the interface and is one gate, so “same detector” and representativeness are too strong. | Major | RESOLVED | Manuscript Secs. 3.1, 4.2, 5.3, and 6.1 call it capacity/protocol-controlled, not detector-identical or exhaustive. |
| SAFE-REF found no safe-use region. | Corrected holdout false-safe 0.92849, upper 0.93957, risk rho 0.03996, tail reduction 0.01780. | One selector/generator cannot show that every normal-only selector fails. | Major | RESOLVED | Manuscript Secs. 5.5, 6.1, and 7 restrict the result to the tested selector, observables, and controlled family. |
| SAFE-REF evidence is prospective. | Timestamped commits/runs; corrected split; unchanged 20% criterion. | The first failure was viewed before correction, and feasibility was diagnosed later. | Critical | RESOLVED | Manuscript Sec. 4.5 and `AP_CARE_V2_EXECUTION_SPEC.md` preserve chronology; the corrected result is called post-correction, not pristine preregistration. |
| AP-CARE corroborates the mechanism. | Frozen G1 internal holdout: retention passed; attenuation did not; five of six checks failed. | It followed SAFE-REF, uses proxies, and only four matched medium/high cases inform one comparison. | Major | RESOLVED | Manuscript Secs. 4.4, 5.6, and 7 report the chronology, proxy scope, separate endpoints, and subset count. It is not called an independent replication. |
| Known components establish physical fault safety. | Generator exposes environmental, machine, and injected fault-proxy inputs separately. | Synthetic provenance inside a generator is not physical ground truth across real faults, paths, and rooms. | Major | RESOLVED | Manuscript Secs. 4.4 and 7 use “fault proxy” and bound external validity. |
| Paired bootstrap supports uncertainty statements. | 5,000 replicates preserve file pairs and resample normal/anomaly files within machine-section groups. | File-level exchangeability may miss acquisition clustering and does not sample machine populations. | Major | PARTIALLY RESOLVED | Manuscript Sec. 4.3 and Limitation 2 now state the resampling unit and boundary. No stronger frozen cluster analysis is available. |
| Results are statistically robust to training. | Same locked seed 13711 for B00/B01/B02. | Evaluation bootstrap does not quantify initialization/optimizer uncertainty. | Major | OPEN | Manuscript Sec. 4.3 and Limitation 1 correctly make inference conditional. Do not add retrospective seeds; authors must accept this submission risk. |
| Endpoint claims respect the protocol. | Mean AUC primary; pAUC prospective secondary; exact \(\Omega\) post hoc. | Highlighting harmful pAUC or \(\Omega\) can become endpoint switching. | Major | RESOLVED | Abstract, Secs. 4.3, 5.2, 5.7, and 7 retain the hierarchy and descriptive label. |
| Real/emulated sensitivity supports external validity. | Frozen descriptive split: two real, five emulated types. | Tiny non-random groups cannot support inferential subgroup claims or evaluation-machine generalization. | Major | RESOLVED | Manuscript Sec. 5.7 and Limitation 5 keep it descriptive and reject a real-machine benefit claim. Broader validity remains OPEN below. |
| V1–V10 strengthens the stop decision. | Sequential frozen gates and negative-evidence ledger. | Outcome-informed variants create researcher degrees of freedom and are not replications. | Major | RESOLVED | Manuscript Secs. 4.5, 6.5, and 7; `FP_NAA_POST_V10_LITERATURE_GATE.md`. They are history/narrowing/stop support only. |
| The novelty claim is supported by literature. | Audit-A1 found no reviewed direct ASD source jointly reporting the four specified audit elements. | The matrix is bounded, includes preprints/reports, and is not an exhaustive ICA/BSS or safety-method review. | Major | PARTIALLY RESOLVED | Introduction uses “we did not identify ... among reviewed direct ASD sources”; theory sources are added in Sec. 2.2. A publisher-verified final search remains OPEN. |
| The conclusion matches the evidence. | Conditional B01/B02 effects plus controlled SAFE-REF/AP-CARE failures under frozen contracts. | Few interventions and proxy families cannot establish a general identifiability or safety limit. | Critical | RESOLVED | Abstract, Discussion Sec. 6.1, Limitations, and Conclusion now distinguish formal, interpretive, and empirical levels and bind the conclusion to tested regimes. |

## Open submission blockers

1. **Independent theoretical review.** A BSS/multichannel DSP expert must confirm the latent-versus-
   physical semantic boundary and the decision-theoretic Proposition 1.
2. **Bibliography and novelty audit.** Verify all author/title/venue/year/pages/DOI/status metadata
   against publisher or official records and refresh the bounded search at submission time. The
   existing Audit-A1 direct-ASD matrix is not an exhaustive theory review.
3. **Author acceptance of conditional inference.** The paper cannot claim training-seed robustness,
   acquisition-cluster robustness, or population inference over machines from the frozen design.
4. **External validity.** Only two real synchronized types, one detector family, and synthetic fault
   proxies remain. This is an honest limitation, not fixable by wording.
5. **Journal production.** Convert the Markdown draft to DSP format, finalize figures/tables,
   bibliography, ethics/data statements, author contributions, and line-by-line proofing.

## Historical objections now closed

- “The manuscript has no formal model” → **RESOLVED** by manuscript Sec. 2 and
  `IDENTIFIABILITY_FORMALIZATION.md`.
- “The exact DCASE metric is absent” → **RESOLVED** as a deterministic post-hoc descriptive metric
  in manuscript Sec. 5.7, without replacing the frozen estimand.
- “The real/emulated composition is hidden” → **RESOLVED** in manuscript Secs. 4.1, 5.7, and 7.
- “The one-seed bootstrap scope is hidden” → **RESOLVED** in manuscript Sec. 4.3 and Limitation 1.
- “SAFE-REF chronology and 20% feasibility are hidden” → **RESOLVED** in manuscript Sec. 4.5.
- “V1–V10 are treated as replications” → **RESOLVED** in manuscript Sec. 6.5.
- “Positive dual-channel systems contradict the paper” → **RESOLVED** by the two-question framing
  in Introduction and Discussion Sec. 6.2.

## Reviewer recommendation

Send the hardened draft for **internal expert review**. Do not submit to *Digital Signal Processing*
until the four scientific blockers above receive explicit sign-off. No new algorithm search or
reopening of frozen experiments is justified.
