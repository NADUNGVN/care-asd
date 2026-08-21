# Statistics and external-validity author sign-off

Status: required author review before journal assembly. Checking a row confirms acceptance of the
stated inferential boundary; it does not remove the limitation. No retrospective model search is
authorized.

| Author sign-off | Limitation | What is estimated | What is not estimated | Required manuscript wording | More retrospective experimentation? |
|---|---|---|---|---|---|
| [ ] | One locked training realization, seed 13711 | Conditional B00/B01/B02 differences for the frozen model realization | Training-initialization, optimizer, or model-selection variability | “Inference is conditional on the locked training realization (seed 13711).” | Additional seeds after outcome inspection would be retrospective and do not repair the frozen primary study; prohibited for this claim. |
| [ ] | File-level paired bootstrap exchangeability | Evaluation-observation uncertainty under paired resampling within the declared machine-section-condition strata | Dependence beyond the recorded strata or arbitrary deployment sampling | “Intervals assume file-level exchangeability within the frozen strata.” | A new resampling model cannot create missing cluster identifiers; clarify rather than post-hoc optimize. |
| [ ] | No acquisition-session cluster model | No separate session-level variance component | Session-to-session uncertainty or correlation among files from the same unrecorded acquisition | “The bootstrap does not account for unidentified acquisition-session clustering.” | Unhelpful without trustworthy session identifiers; do not fabricate clusters. |
| [ ] | Seven development machine types | Conditional aggregate and per-type descriptions across the seven recorded types | Population inference over machine types or unseen evaluation machines | “The study is conditional on the seven development machine types.” | Outcome-driven expansion is outside Audit-A4; future prospective work only. |
| [ ] | Two real synchronized machine types | Descriptive ToyCar/fan versus five-emulated summaries from frozen predictions | Reliable real-versus-emulated subgroup effects | “Only two real synchronized types are available; the split is descriptive and not an inferential subgroup result.” | Do not search or retune on favorable subgroups. |
| [ ] | One detector family | Effects for the locked official-compatible autoencoder comparator | Generalization to foundation models, raw-waveform systems, learned fusion, ensembles, or all ASD detectors | “The intervention results are conditional on one locked detector family.” | New detectors would be new research and are not required to report the present boundary honestly. |
| [ ] | Synthetic fault proxies | Retention/attenuation for components with known provenance inside the declared generator | Coverage of real fault physics, nonlinear propagation, rooms, or all fault signatures | “Controlled fault proxies provide generator-level component identity, not physical ground truth for all faults.” | New proxy search after outcomes cannot establish external validity; future prospective physical validation is needed. |
| [ ] | Development-only analysis | Performance and component behavior on the permitted DCASE-aligned development observations/holdouts | Challenge evaluation performance or deployment generalization | “All empirical conclusions are development-only.” | Evaluation-label access is prohibited; do not extrapolate. |
| [ ] | Post-hoc DCASE \(\Omega\) | Exact descriptive harmonic metric derived deterministically from frozen B00/B01/B02 scores | A preregistered endpoint, new inferential test, or threshold-optimized result | “Derived post hoc from frozen predictions/artifacts; secondary descriptive analysis.” | No additional analysis is needed; it must not replace the frozen AUC primary estimand. |
| [ ] | Descriptive real/emulated split | Frozen descriptive \(\Omega\) summaries under a construction-defined partition | Confirmatory interaction, subgroup benefit, or population difference | “The real/emulated split is post-hoc descriptive; B02's small real-only delta is not evidence of benefit.” | Additional subgroup testing would be outcome-driven and is prohibited. |

## Collective author confirmation

- [ ] We accept that the paired bootstrap does not quantify training-initialization, acquisition-
  cluster, or machine-population uncertainty.
- [ ] We accept that controlled proxy provenance is not equivalent to physical-fault coverage.
- [ ] We will not describe post-hoc \(\Omega\) or real/emulated results as confirmatory.
- [ ] We will not add retrospective seeds, models, thresholds, or favorable subgroups to strengthen
  the known result.
- [ ] We will keep the conclusion conditional on the locked comparator, declared interventions,
  component families, development machines, and frozen decision rules.

Author names/signatures and date:

________________________________________________________________________________

________________________________________________________________________________
