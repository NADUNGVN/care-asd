# Controlled contaminated-reference safety-audit protocol

Status: methodological synthesis of the frozen CARE-ASD study. This is a protocol for auditing
preprocessing when an auxiliary microphone may contain both nuisance and desired-source energy. It
is not claimed to be universally validated and does not define a new CARE-ASD model.

## 1. Intended use and claim boundary

The protocol applies when a signal-processing intervention uses a synchronized auxiliary channel
as a reference and can suppress information from a primary observation. Its purpose is to keep four
questions separate:

1. Does the intervention change downstream ASD utility under a controlled comparator?
2. Does it attenuate a known environmental component?
3. Does it retain known machine- or fault-relevant components?
4. Can a label-free selector identify when both efficacy and safety hold?

Passing only one question is insufficient. A detector score can improve while some fault evidence
is removed; retention can be high because the processor abstains or has negligible attenuation;
and a controller can achieve high coverage by accepting unsafe cases.

The protocol audits a declared processor, controller, component family, detector, and data regime.
It does not prove safety for untested faults, propagation paths, detectors, or deployment domains.

## 2. Layer A: downstream utility under a controlled comparator

### 2.1 Freeze the comparison contract

Before reading candidate outcomes, record:

- development files and their pairing;
- training data and normal-only restriction;
- feature and score definitions;
- detector architecture or explicit capacity constraint;
- optimizer, training realization, and random seed;
- primary and secondary endpoints;
- bootstrap unit, stratification, repetitions, and interval rule; and
- pass, fail, and stop decisions.

The strongest comparison changes only the preprocessing and preserves the entire downstream
detector. If an intervention necessarily changes the detector interface, the altered components and
capacity controls must be reported rather than described as an identical detector.

### 2.2 Use paired observations

Each candidate and reference should score the same evaluation observations. Candidate-minus-
reference effects are paired within the prespecified sampling strata. The resampling target must be
stated precisely. Resampling frozen evaluation clips quantifies evaluation-observation uncertainty;
it does not quantify training-initialization uncertainty when only one training realization exists.

### 2.3 Preserve the endpoint hierarchy

The registered primary endpoint determines the confirmatory decision. Prospectively specified
secondary endpoints may show harm or a useful diagnostic pattern, but they must remain secondary.
Post-hoc benchmark metrics or subgroup summaries are descriptive additions and cannot replace the
frozen estimand.

**CARE-ASD realization.** B01 changes the near waveform while retaining the locked B00 backend.
B02 retains the near view but introduces a capacity-controlled two-branch input projection, so it
is protocol- and capacity-controlled rather than literally architecture-identical. B00, B01, and
B02 use the same 1,400 development files and seed 13711.

## 3. Layer B: controlled known-component decomposition

### 3.1 Construct or obtain separately known components

The audit requires observations for which the intervention's environmental and machine/fault
inputs can be evaluated separately. Components may be measured, simulated, or mixed from recorded
carriers, but their provenance and limitations must be explicit. Synthetic fault proxies are not a
substitute for all physical failures.

### 3.2 Freeze the realized controller

If the processor is nonlinear or adaptive, first realize its controller state on the complete
mixture. Then hold that state fixed while applying the processor to the known components. Allowing
each counterfactual component to re-estimate the controller would answer a different question and
can obscure what the realized mixture decision removed.

### 3.3 Report both efficacy and safety

At minimum, define:

- environmental attenuation, with sign convention and aggregation;
- machine/fault retention, including lower-tail behavior;
- a no-processing reference;
- eligible-case rules, if any; and
- a joint safe-and-useful criterion fixed before holdout evaluation.

A generic waveform, SNR, or log-Mel change is not sufficient evidence of safe denoising. It does
not identify which source was changed.

**CARE-ASD realization.** AP-CARE G1 applies one realized controller separately to known
environmental, normal-machine, and injected fault-proxy components. The frozen endpoints include
environmental attenuation and the median and lower-tail fault-retention ratios.

## 4. Layer C: selector and controller safety

Let the hidden controlled label (S=1) denote a case that meets the prespecified joint safety and
efficacy target, and let the normal-only selector decision (D=1) denote acceptance. Report:

\[
\text{coverage}=P(D=1),
\]

\[
\text{false-safe}=P(S=0\mid D=1),
\]

along with an uncertainty bound, safe-use prevalence (P(S=1)), and useful attenuation among
accepted or eligible cases. Risk scores should also be tested against known component risk on an
untouched controlled holdout.

Coverage is not safety. Zero coverage can trivially suppress false-safe decisions, while broad
coverage can conceal an unacceptable false-safe rate. If safe-use prevalence is low, a nominal
coverage target may be mathematically incompatible with a strict false-safe target; that
feasibility relationship should be diagnosed explicitly without retroactively changing a failed
gate.

**CARE-ASD realization.** Corrected SAFE-REF reports coverage, false-safe rate and its upper bound,
risk tracking, and tail-loss reduction. AP-CARE adds joint retention/attenuation checks and an
untouched internal holdout. The studies have different chronology and should not be counted as
independent preregistered replications.

## 5. Layer D: frozen prospective decisions

The record must distinguish four chronological states:

| State | Permitted activity | Evidential role |
|---|---|---|
| Hypothesis generation | Inspect prior literature and earlier experiments; define a mechanism question. | Motivates a candidate; not confirmatory evidence. |
| Method search | Develop on explicitly designated calibration or development information. | Produces a frozen candidate and gate. |
| Frozen evaluation | Run the fixed candidate once on the declared untouched observations. | Supports the registered decision under its exact contract. |
| Stop decision | Apply pass/fail/stop rules without outcome-driven rescue. | Preserves interpretability of null and harmful results. |

Any protocol correction after an outcome has been inspected must retain the earlier record, explain
the defect, state what was changed, and label the corrected study's chronological status. An
internal calibration/holdout split remains useful after a correction, but it does not retroactively
make the whole corrected protocol preregistered.

Sequential mechanism studies should not be presented as independent replications. They can provide
research history, narrowing evidence, evidence against reporting one cherry-picked failed variant,
and a justified stopping boundary.

## 6. Minimum audit report

A complete report should contain:

1. the signal model and assumptions that make the reference potentially contaminated;
2. the fixed-comparator contract and endpoint hierarchy;
3. paired downstream effects with uncertainty and training-seed scope;
4. known-component attenuation and retention, including lower tails;
5. selector coverage, false-safe behavior, and uncertainty bounds;
6. an untouched prospective gate or a transparent explanation of why one is unavailable;
7. all protocol changes in chronological order;
8. separate labels for confirmatory, sensitivity, and post-hoc results;
9. source and output hashes sufficient to regenerate every reported value; and
10. an explicit list of untested generalizations.

## 7. CARE-ASD evidence mapping

| Protocol layer | Frozen CARE-ASD evidence | Proper role |
|---|---|---|
| Downstream utility | B00/B01/B02 aligned scores and paired bootstrap | Conditional comparison under locked seed 13711 |
| Feature mechanism | Phase 8 displacement/score association | Frozen post-hoc, non-causal diagnostic |
| Controlled selector safety | Corrected SAFE-REF calibration and holdout | Post-correction controlled study with disclosed chronology |
| Controlled components | AP-CARE G1 calibration and untouched holdout | Prospective component-level mechanism gate |
| Robustness | Machine/domain bootstrap and leave-one-machine-out | Descriptive sensitivity; not unseen-machine validation |
| Benchmark alignment | Exact DCASE harmonic score from frozen score files | Derived post hoc; secondary to frozen paired estimands |
| Research history | FP-NAA V1-V10 ledger | Sequential narrowing and stop support; not pooled replication |
