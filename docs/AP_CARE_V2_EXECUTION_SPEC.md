# AP-CARE v2 research execution specification

## Authority and status

This document is the authoritative research-to-code contract for the isolated
`research/ap-care-v2` branch. It supersedes SAFE-REF as the active method
direction on this branch. `docs/PHASE10_REFERENCE_SAFETY.md` and Phases 7--10
remain reproducible historical evidence; they are not instructions to continue
SAFE-REF screening or evaluation.

The branch starts from clean `origin/main` commit `31286a4`, which contains the
corrected Phase 10 report. Work on this branch must not be merged or
cherry-picked into `main` before the AP-CARE replication gate passes.

## Decision motivating the pivot

The corrected Phase 10 SAFE-REF holdout contained 2,048 controlled cases. Only
172 cases (8.40%) met the synthetic definition of safe reference use. The
normal-only policy nevertheless accepted 1,776 cases, producing 92.85%
false-safe decisions, a 93.96% upper confidence bound, risk Spearman rho of
0.040, and only 1.78% tail-loss reduction.

This is a scientific gate failure, not a runtime failure. More importantly, at
8.40% safe prevalence, any selector constrained to at most 5% false-safe rate
has maximum possible coverage of approximately `0.08398 / 0.95 = 8.84%`.
Therefore the former 20% coverage requirement is infeasible even for an oracle.
Phases 11 and 12 are stopped.

Prior aligned evidence also establishes the cancellation risk:

| System | Mean AUC | Mean pAUC | Decision |
|---|---:|---:|---|
| B00 aligned near-only | 0.60813 | 0.55391 | Exact reference |
| B01 CARE residual replacement | 0.60251 | 0.53459 | Rejected |
| B02 near-primary gated residual | 0.60627 | 0.54789 | Rejected/tied |

B01 minus B00 had mean pAUC delta -0.01740 with 95% bootstrap interval
[-0.03271, -0.00286]. B02 minus B00 had mean pAUC delta -0.00540 with interval
[-0.01398, 0.00323]. Phase 8 found Spearman rho -0.5524 between residual-induced
log-Mel displacement and anomaly-score change. Stronger cancellation therefore
correlates with reduced anomaly evidence in the current pipeline.

## Revised research question

Under explicit acoustic-support assumptions, can a deterministic, causal,
normal-only controller bound contaminated-reference cancellation so that it
improves the fault-preservation/noise-attenuation frontier and development pAUC
relative to an unchanged near-only detector and capacity-matched reference
denoisers?

AP-CARE does **not** claim to identify every unseen anomaly from normal-only
recordings. It guarantees only an observable intervention bound: the algorithm
cannot remove more time-frequency energy than its pre-registered budget. Any
claim about anomaly preservation is conditional on the assumptions below and
must be measured, not asserted.

## Scope assumptions

- **A1 -- path relevance:** machine-originating faults reach the far microphone
  through a path related to the normal machine path over the analysis window.
- **A2 -- support relevance:** fault energy overlaps normal machine support or
  is protected by the conservative per-band removal budget.
- **A3 -- reference utility:** environmental interference has enough observable
  near/far structure to provide non-trivial cancellation opportunity.
- **A4 -- causal availability:** only past and current frames and training-normal
  statistics are available at inference.

Outside A1--A3, AP-CARE offers a bounded perturbation, not a semantic anomaly
preservation guarantee. Results must be stratified into in-support and
out-of-support faults so this boundary is visible.

## Method contract

For near STFT `X_n(f,t)`, far STFT `X_f(f,t)`, and a causal transfer estimate
`H(f,t)`, the ordinary reference cancellation proposal is

```text
C_ref(f,t) = H(f,t) X_f(f,t)
```

AP-CARE applies a bounded controller:

```text
B(f,t) = clip(r_noise(f,t) * (1-r_leak(f,t)) * (1-u_H(f,t)), 0, 1)
C_ap(f,t) = G_ref(f,t) * B(f,t) * H(f,t) X_f(f,t)
Y_ap(f,t) = X_n(f,t) - bounded(C_ap(f,t))
```

The terms have separate, testable meanings:

- `r_noise`: evidence that the far observation contains removable environmental
  interference rather than merely machine-correlated energy.
- `r_leak`: normal-support evidence that cancellation may remove machine-origin
  content. It is a risk proxy, not an anomaly probability.
- `u_H`: instability of the causal transfer estimate across independent normal
  profiles or adjacent causal windows.
- `G_ref`: the fixed reference-denoising proposal shared with the matched
  baseline.

`bounded()` enforces a configurable removed-energy ceiling independently in
each frequency band and frame. It must expose the proposed energy, permitted
budget, actual removed energy, active-bound indicator, and final gain. The
unchanged near view is always retained; the AP residual is an auxiliary view or
front-end candidate and never silently replaces the reference signal.

The first implementation is deterministic. A neural gate, anomaly-labelled
threshold, test-set statistic, or learned score fusion is out of scope until
the deterministic method passes replication.

## Controlled metric definitions

Synthetic mixtures retain the known near/far normal-machine, fault, and
environmental-noise components. Compute the controller once from the complete
mixture, freeze its realized transfer/gain/budget tensors, and apply that same
linearized operation separately to each known component. This prevents a
counterfactual component from changing the controller being measured.

For component energy `E(.)` over the pre-registered active frames/bands:

```text
fault_retention        = E(T_ap(fault_near)) / E(fault_near)
machine_retention      = E(T_ap(machine_near)) / E(machine_near)
noise_attenuation_db   = 10 log10(E(noise_near) / E(T_ap(noise_near)))
true_reference_leakage = E(machine_far) / (E(machine_far) + E(noise_far) + eps)
```

Reports include median, fifth percentile, bootstrap interval, and the full
trade-off curve rather than only a selected operating point. An eligible
cancellation case is fixed before execution as one with non-zero environmental
reference energy and near-channel environmental-to-machine energy ratio of at
least -10 dB. Matched-attenuation comparisons use the closest pre-registered
D00 operating point within +/-0.25 dB; unmatched cases are reported and never
silently discarded.

## Comparator contract

All DCASE comparisons use the exact same official-compatible backend, data,
features, training schedule, seeds, aggregation, and score normalization.

| ID | Front end | Role |
|---|---|---|
| B00 | Unchanged near channel | Primary reference |
| B01 | Historical CARE residual replacement | Negative control, no rerun unless required |
| B02 | Historical near-primary gated residual | Negative control, no rerun unless required |
| D00 | Fixed floored RefSub | Capacity-matched cancellation baseline |
| D01 | Train-normal adaptive per-band RefSub | Strong prior-art comparator |
| D02 | Causal power-ratio/coherence mask with floor | Strong prior-art comparator |
| S00 | SAFE-REF | Failed identifiability control; no Phase 11/12 continuation |
| A00 | AP-CARE v2 | Candidate method |

D01 and D02 must be reproduced from cited equations and frozen globally; they
must not be tuned per development anomaly or machine. If exact reproduction is
not possible, the report must name the deviation rather than claim equivalence.

## Leakage and tuning boundary

- Training-normal audio may estimate transfer, support, noise utility,
  uncertainty, and normalization statistics.
- Synthetic ground truth may define mechanism gates and global controller
  constants.
- DCASE development anomaly labels may only perform the frozen G2/G3 audit.
- No AP-CARE threshold may be selected from DCASE development anomalies.
- Evaluation machine types and labels are unavailable until code, config,
  seeds, manifests, and development decisions are frozen and hashed.
- No cross-test-clip reduction or test-set normalization is allowed.

## Pre-registered gates

### G0 -- contract and regression safety

Pass only when the AP-CARE config schema, CLI dry run, deterministic synthetic
fixtures, causal-prefix invariance, stereo validation, budget invariants,
baseline equivalence, provenance, and leakage tests all pass. With AP-CARE gain
zero, output must equal the near view. With the budget disabled, the proposal
must reproduce its declared reference baseline within numerical tolerance.

### G1 -- controlled mechanism gate

Use an independent calibration/holdout split and sweep at least: far-channel
machine leakage, environmental-noise gain, path gain, delay, transfer mismatch,
fault amplitude, fault band, support relation, and multiple fixed seeds.

G1 passes only if every condition below holds on the untouched holdout:

1. `r_leak` versus true machine leakage has Spearman rho >= 0.60 and bootstrap
   95% lower bound > 0.40.
2. `u_H` versus injected path mismatch has Spearman rho >= 0.60 and bootstrap
   95% lower bound > 0.40.
3. In medium/high-contamination cases, AP-CARE improves median fault retention
   by at least 0.10 absolute over D00 at matched noise attenuation within
   +/-0.25 dB; the paired bootstrap 95% lower bound must be > 0.
4. AP-CARE median fault retention is >= 0.90 and its fifth percentile is >=
   0.75 in the pre-registered in-support stratum.
5. Median environmental-noise attenuation is >= 1.0 dB in eligible cases.
6. Cancellation is non-trivial: at least 20% of eligible cases have non-zero
   cancellation and median attenuation among those cases is >= 1.0 dB.
7. Results report in-support and out-of-support faults separately; failure on
   out-of-support faults restricts the claim and may not be hidden by averaging.

If G1 fails, stop A00 development. Do not alter thresholds and rerun the same
holdout. The publication route becomes an identifiability/audit study based on
Phases 7--10 and the failed controlled mechanism.

### G2 -- three-seed development screening

Use seeds `[13711, 42, 2026]`. Primary metric is mean pAUC at max FPR 0.1.
G2 passes only when:

- A00 pAUC minus B00 is positive on the three-seed score ensemble;
- at least two of three individual seeds have positive pAUC direction;
- A00 mean AUC minus B00 is >= -0.005;
- no machine-level official component falls by more than 0.020; and
- A00 improves the fault-preservation/noise-attenuation diagnostics over D00
  without changing the frozen controller.

Failure stops GPU replication. D01/D02 may show that reference denoising is
useful, but they do not rescue the AP-CARE claim.

### G3 -- ten-seed replication

Use seeds `[13711, 42, 2026, 3407, 777, 11, 23, 101, 314, 2718]`. G3 passes
only when the stratified paired bootstrap gives:

- pAUC delta A00 minus B00 with 95% lower bound > 0;
- mean-AUC delta with 95% lower bound > -0.005;
- no machine-level official component drop greater than 0.010; and
- the positive result remains when the weakest and strongest seed are removed.

No evaluation access, deployment benchmark, or merge to `main` is allowed if
G3 fails.

### G4 -- frozen unseen-machine evaluation

Freeze the commit, config hashes, comparator implementations, seeds, manifest
hashes, checkpoint hashes, and development report before evaluation scoring.
Generate score files through a ground-truth-free interface, seal them, and only
then call a separately pinned official evaluator. Evaluation results are never
used to revise A00.

### G5 -- board-kit study

Jetson AGX Xavier and Jetson Xavier NX are both board kits. Measure causal
single-stream latency, real-time factor, peak memory, sustained power, thermal
state, and numerical agreement only after G4. Hardware results support the
method; they do not substitute for G1--G3 novelty.

## Shortest execution path

| Milestone | Work | Server use | Stop rule |
|---|---|---|---|
| M0 | Freeze this specification and branch metadata | None | Contract review |
| M1 | Implement synthetic sweep, A00 controller, diagnostics, and tests | None or CPU smoke test | G0 |
| M2 | Run one immutable synthetic calibration/holdout benchmark | CPU, bounded job | G1 |
| M3 | Cache A00/D00/D01/D02 vectors only after G1 | CPU workers capped to preserve host capacity | Cache audit |
| M4 | Run three-seed capacity-matched screening | GPU | G2 |
| M5 | Run ten-seed replication and paired bootstrap | GPU | G3 |
| M6 | Freeze and score unseen machines | GPU | G4 |
| M7 | Benchmark both Jetson board kits | Jetson devices | G5 |

There is no Phase 11 SAFE-REF job in this path. M2 is the earliest scientific
kill point and must finish before any long DCASE training.

## Implementation checkpoint

M0 and M1 are complete on `research/ap-care-v2`. The additive public contracts
are:

- `APCAREExperimentConfig` in `src/care_asd/ap_care_config.py`;
- `APCAREController.fit`, `transform`, and `apply_frozen` in
  `src/care_asd/signal/ap_care.py`;
- the controlled generator and immutable artifact runner in
  `src/care_asd/signal/ap_care_simulation.py`; and
- `care-asd ap-care simulate`, including a side-effect-free `--dry-run`.

The controller exposes its unchanged near STFT, residual candidate, realized
cancellation filter, `r_noise`, `r_leak`, `u_H`, coherence, gain/delay mismatch,
per-band proposed/permitted/actual removed energy, and active-bound flags. A
diagnostic bypass proves equivalence to the declared unbounded complex
reference proposal; it is not enabled by the frozen experiment config.

Local smoke sweeps used while implementing M1 are software diagnostics only.
They are ignored, uncommitted, and are not G1 evidence. The first decision-grade
G1 run must start from the committed config and produce a new immutable server
artifact after the branch commit is reviewed.

The M2 runtime contract is now implemented by
`scripts/server/start_ap_care_g1.sh`, `run_ap_care_g1.sh`, and
`status_ap_care_g1.sh`. It freezes 512 cases, records every deterministic case
seed, hashes the config and each result artifact, writes atomic case progress,
and defaults to 16 single-threaded worker processes. On SERVER-02 this reserves
12 of 28 logical CPUs, exceeding the requested 25% host reserve. Exit status 2
from the scientific CLI is normalized to a completed job only when both
`gate.json` and `run.json` exist; it means G1 failed and AP-G2 must stop. Runtime
or Git-push failures remain operational failures.

## Required artifacts

Each gate writes a versioned directory containing config, config hash, git SHA,
manifest hash, seed list, environment summary, machine-readable metrics,
human-readable decision, and exact paths to large server-only caches. No raw
audio, token, checkpoint, or large log is committed.

Long server work must use checked-in start/status wrappers. Commands supplied to
the researcher remain one physical line. Reports from this branch are pushed to
`research/ap-care-v2`, not `main`.

## Paper decision matrix

| G1 | G2/G3 | Publication interpretation |
|---|---|---|
| Pass | Pass | AP-CARE method paper with conditional preservation claim |
| Pass | Tie/fail | Mechanism and limitation paper; no unseen evaluation claim |
| Fail | Not run | Identifiability/audit paper using Phases 7--10 and G1 |

The intended journal contribution is the combination of an explicit
identifiability boundary, a causal bounded-intervention method, and controlled
mechanistic evidence. A product of existing coherence/SNR heuristics without
these components is incremental and is not the target contribution.

## Prior-art anchors

- KATECH, “Noise-Aware Reference Denoising for First-Shot Anomalous Sound
  Detection,” DCASE 2026 Task 2 technical report:
  <https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Kim_91_t2.pdf>
- Morita, “Leveraging Stereo Spatial Information for Noise-Aware Anomalous
  Sound Detection,” DCASE 2026 Task 2 technical report:
  <https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Morita_50_t2.pdf>
- LUDO Lab, “Residual View and Prototype Selection for Noise-Aware Anomalous
  Sound Detection,” DCASE 2026 Task 2 technical report:
  <https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Kim_27_t2.pdf>
