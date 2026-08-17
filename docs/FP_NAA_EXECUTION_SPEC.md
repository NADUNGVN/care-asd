# FP-NAA successor execution specification

Status: preregistered working protocol on branch `research/fp-naa`.

This track succeeds the frozen CARE-ASD/AP-CARE audit. It does not reinterpret or overwrite the
negative identifiability and cancellation results. The new objective is to obtain a strong,
statistically credible DCASE 2026 result and a contribution that remains distinct from the current
noise-aware ASD frontier.

## 0. Frozen server environment

SERVER-02 experiments run only in the dedicated Conda environment `care-asd-fp-naa`. Conda owns
Python 3.11 and the environment lifecycle; the exact Python dependency set is frozen in
`requirements/fp-naa-cu118.lock.txt`. The Linux runtime is pinned to PyTorch and torchaudio
`2.6.0+cu118`, CUDA runtime 11.8, and cuDNN 9.1. No FP-NAA wrapper invokes the repository `.venv`
or `uv run`.

The environment definition is `environments/fp-naa-cu118.yml`. Setup must finish with a real GPU
convolution probe, not only `torch.cuda.is_available()`. Jobs execute through the Python CLI with
`conda run`, so they select the named environment even when the interactive SSH prompt displays
`(base)`. FP-NAA has no server shell wrappers.

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda run -n care-asd-fp-naa care-asd fp-naa runtime-check
```

```bash
env -u LD_LIBRARY_PATH -u LD_PRELOAD conda run -n care-asd-fp-naa care-asd fp-naa job status
```

## 1. Research question

Can a dual-microphone noise-aware representation adapter improve first-shot anomalous sound
detection while explicitly preserving the representation change caused by machine faults?

The motivating gap is concrete. Noise-aware BEATs systems distil a clean teacher representation
with representation MSE, yet the NABEATs study reports that a representation with better MSE/SNR
can perform worse on downstream tasks and identifies the training objective as unresolved. The
proposed contribution is therefore the objective and its safety validation, not a new claim for
cross-attention, pseudo anomalies, BEAM, RDP, or local-density score normalization.

## 2. Method and claim boundary

Working name: **FP-NAA — Fault-Preserving Noise-Aware Adapter**.

For a frozen BEATs encoder `E`, normal target waveform `s`, reference/background waveform `n`, a
physically parameterized pseudo-fault transform `P`, and a lightweight reference-conditioned
adapter `F`, define:

```text
teacher fault delta = E(P(s)) - E(s)
student fault delta = F(P(s) + n, n) - F(s + n, n)
```

Training combines:

1. normal representation distillation from the uncorrupted teacher;
2. cosine alignment of teacher and student fault-delta directions;
3. a robust penalty on fault-delta magnitude retention;
4. reference dropout/corruption consistency so a mismatched far microphone cannot silently erase
   anomaly evidence.

The adapter is deliberately small and operates on cached BEATs token grids. The frozen encoder,
BEAM, relative-deviation pooling (RDP), and variance-minimizing score rescaling are common to every
comparator. Any gain attributed to FP-NAA must therefore survive a capacity-matched MSE-only
adapter comparison.

### 2.1 Frozen counterfactual augmentation protocol

Only normal `dev_train` recordings are augmentation sources. For every target clip, a deterministic
different normal clip from the same machine/section supplies the far-channel noise reference. One
shared, scaled reference waveform is added to the clean and pseudo-fault target, so their waveform
difference is exactly the injected fault perturbation before the frozen BEATs encoder. The noise SNR
is sampled uniformly from -10 to +10 dB and the fault perturbation RMS is sampled uniformly from
-24 to -12 dB relative to the clean target RMS. Shared peak limiting is used; independent clipping
of the pair is prohibited because it would invalidate the counterfactual identity.

The training fault families are frozen before observing C1/C2 results:

- periodic impacts exciting a decaying resonance (localized bearing/gear impact proxy);
- amplitude modulation (load, mesh, or rotating-envelope sideband proxy);
- small periodic time warp/frequency modulation (speed-fluctuation sideband proxy).

Ten percent of normal training clips, chosen by a stable hash, additionally receive a broadband
friction-burst fault. This fourth family is never used by the C2 optimizer and is the pre-registered
out-of-family retention test. These signals are diagnostic probes, not claims of a complete physical
fault simulator. Development anomalies are never used to create, select, or scale pseudo-faults.

Not novel and not claimed:

- BEATs or frequency-preserving BEATs tokens;
- cross-attention, FiLM, LoRA, or dual-microphone conditioning;
- pseudo-fault/pseudo-anomaly generation by itself;
- BEAM, RDP, nearest-neighbour memories, or score calibration;
- waveform cancellation or source separation.

Potentially novel claim, subject to the literature audit and all gates below:

> Counterfactual fault-delta preservation prevents a reference-conditioned noise adapter from
> improving normal-signal reconstruction at the cost of erasing anomaly-relevant representation
> changes.

## 3. Frozen comparators

All systems use BEATs Iteration 3, eight frequency patches, RDP with `gamma=8`, band-aligned BEAM,
cosine distance, and variance-minimizing normal-only score rescaling.

| ID | Frontend | Training objective | Purpose |
| --- | --- | --- | --- |
| C0 | frozen near-channel BEATs | none | strong training-free baseline |
| C1 | dual-channel adapter | normal representation MSE only | capacity-matched noise-aware control |
| C2 | same adapter | MSE + counterfactual fault-delta loss | main FP-NAA candidate |
| C3 | same as C2 | C2 + reference dropout/corruption | reference-safety ablation |

The first implementation milestone is C0. Candidate results are uninterpretable until C0 is close
to the published DCASE 2026 development reference of 62.02 official-score points for original
BEATs + frequency RDP(8) + BEAM.

## 4. Exact evaluation contract

For every machine/section:

- source AUC compares source-domain normal test clips against **all** anomalous clips;
- target AUC compares target-domain normal test clips against **all** anomalous clips;
- pAUC at maximum FPR 0.1 uses normal and anomalous clips from both domains;
- the official score is the harmonic mean of all source AUC, target AUC, and pAUC cells.

The existing legacy CARE-ASD metric remains unchanged for reproducibility of old artifacts. FP-NAA
uses a new exact scorer and writes both cell metrics and the official harmonic mean.

Development labels may be used only for method selection on the seven development machine types.
All candidate decisions additionally use leave-one-machine-out (LOMO) analysis. Evaluation-machine
labels are never used.

For each LOMO fold, the adapter optimizer sees counterfactual pairs from six machine types only.
The held-out machine still supplies its normal `dev_train` clips to the frozen RDP8+BEAM memory,
because normal-only machine onboarding is part of the DCASE task rather than supervised adapter
training. C1 and C2 are retrained from the same initialization for all three screening seeds; the
fold decision uses their mean official score and counts a fold as positive only when C2 > C1.

The confirmatory confidence interval is a paired percentile bootstrap of the **official harmonic
score**, not a confidence interval for legacy mean AUC. Sampling is with replacement inside each
machine/section/domain/condition stratum, uses identical sampled clip indices for C1 and C2, and
recomputes every source AUC, target AUC, pooled pAUC, and the final harmonic mean in each of 10,000
replicates.

Confirmatory execution is staged and immutable. The three screening seeds are reused byte-for-byte;
only seeds 3407 and 777 are newly trained after both the G2 core and three-seed LOMO gates pass.
Their five score files are averaged without labels for each capacity-matched candidate. The exact
paired bootstrap then compares the five-seed C2 and C1 ensembles. Run contracts hash the C0 score,
screening gate, LOMO gate, caches, and frozen config so artifacts from unrelated runs cannot be
mixed. This staging changes compute cost, not the registered G3 decision rule.

The confirmatory LOMO stage likewise reuses all 42 screening-fold artifacts (seven machines, three
seeds, two candidates) and trains only the 28 missing seed/fold/candidate models. Fold deltas are
then recomputed over all five seeds; the gate remains positive C2 - C1 change on at least six of
seven held-out machines.

### 4.1 Frozen reference-safety protocol

The far microphone is not assumed to be noise-only. DCASE 2026 work explicitly notes that it can
contain weaker machine sound and that using its full spectrum as a noise estimate can remove the
machine signal. Reference safety is therefore evaluated on only the deterministically reserved 10%
of normal training clips carrying the never-trained friction-burst pseudo-fault. No development
anomaly is used by this protocol.

Reference leakage is constructed at waveform level before the pinned BEATs encoder. The same far
noise waveform is mixed with the clean or faulty target at machine-to-noise ratios -20, -10, and
0 dB (low, medium, high). Each clean/fault reference pair uses one gain and one shared peak scale,
so the reference-side counterfactual difference is not distorted by independent clipping. The five
frozen C2 seeds are tested in this order:

1. matched donor reference;
2. deterministic reference from a different machine type;
3. zero-waveform microphone dropout, encoded by BEATs rather than represented by an artificial
   zero token;
4. channel-swap stress, precisely defined as the reference socket receiving the primary clean or
   faulty channel while the primary input remains available for measuring output fault retention;
5. low, medium, and high target leakage in the far reference.

For each condition, median retention must be at least 0.85 and the minimum across the five seeds of
the clip-level 5th percentile must be at least 0.65. If matched retention passes but a corrupted
reference condition fails, reference reliability is the active registered failure mode and one C3
revision is permitted. If matched retention itself fails, C2 is closed without a C3 rescue. This
distinction prevents a general fault-preservation failure from being relabeled as reference risk.

## 5. Pass/fail gates

### G0 — implementation and provenance

- Microsoft BEATs source is pinned to commit
  `833df7e7832e5064a281131ee64a481afa8e5b95`.
- `BEATs_iter3.pt` SHA-256 is
  `8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34`.
- cached feature rows cover the manifest exactly, are immutable, and record source/checkpoint hashes.
- unit tests verify token-grid reconstruction, RDP, BEAM, score rescaling, and the exact DCASE score.

### G1 — strong baseline reproduction

- C0 official development score >= 60.50.
- A result from 59.50 to 60.49 triggers one bounded implementation audit.
- A result < 59.50 blocks candidate claims until the baseline mismatch is explained.

### G2 — screening contribution

Across seeds 13711, 42, and 2026:

- C2 mean official score >= 62.00;
- C2 - C0 >= +1.00 percentage point;
- C2 - C1 >= +0.50 percentage point;
- median in-support fault-delta retention >= 0.90 and 5th percentile >= 0.75;
- held-out friction-burst retention median >= 0.85 and 5th percentile >= 0.65;
- no machine loses more than 2.00 points versus C0;
- at least five of seven LOMO folds have positive C2 - C1 change.

Failure closes the current formulation. At most one causally motivated revision is admitted per
registered diagnostic; unstructured hyperparameter search is prohibited.

### G3 — confirmatory contribution

On seeds 13711, 42, 2026, 3407, and 777:

- ensemble official score >= 63.00;
- stratified paired-bootstrap 95% CI lower bound for C2 - C1 is > 0;
- mean C2 - C1 >= +0.75 percentage point;
- worst machine drop versus C0 <= 1.00 point;
- fault-delta retention gates remain satisfied under unmatched-reference, channel-swap, dropout,
  and low/medium/high reference-leakage strata;
- the C2 advantage is positive on at least six of seven LOMO folds.

Only a G3 pass supports a positive method paper. Otherwise the track reports a bounded negative
result or moves to a new preregistered mechanism.

## 6. Iteration policy

Experiments proceed in this order:

1. reproduce C0 and audit any discrepancy;
2. train C1 to establish what normal MSE distillation alone achieves;
3. screen C2 and inspect only preregistered diagnostics;
4. add C3 only if reference-risk diagnostics show that reliability, rather than delta preservation,
   is the active failure mode;
5. run confirmatory seeds and bootstrap only after a screening pass;
6. freeze the paper tables and claims before any evaluation-set scoring.

Each server job has a short checked-in start command, a separate status command, an atomic state
file, a persistent log, and an immutable report committed to `research/fp-naa`. Server jobs use at
most 12 preprocessing workers by default so at least 25% of SERVER-02 CPU capacity remains free.

## 7. Primary prior art used to set the boundary

- [DCASE 2026 Task 2 setup and exact metric](https://arxiv.org/abs/2606.01578)
- [Anomalous Sound Detection Meets Noise-Aware SSL](https://arxiv.org/abs/2608.00447)
- [Noise-Aware Reference Denoising for First-Shot ASD](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Kim_91_t2.pdf)
- [NABEATs: Noise-Aware Audio Representation Learning](https://arxiv.org/abs/2607.16688)
- [BEAM sub-band matching](https://arxiv.org/abs/2603.13749)
- [Relative-deviation pooling](https://arxiv.org/abs/2603.04605)
- [Variance-minimizing anomaly-score rescaling](https://dcase.community/documents/workshop2025/proceedings/DCASE2025Workshop_Matsumoto_12.pdf)
- [OS-SCL feature perturbation](https://arxiv.org/abs/2509.13853)
- [Official Microsoft BEATs implementation](https://github.com/microsoft/unilm/tree/master/beats)
