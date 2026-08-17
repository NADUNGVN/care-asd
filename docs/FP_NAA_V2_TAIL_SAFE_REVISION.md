# FP-NAA v2: tail-safe, primary-preserving revision

## Status and decision boundary

This is a methodological amendment made after the completed deterministic FP-NAA v1 screening run
`server02_fp_naa_screening_20260817T083029Z`. The v1 artifacts remain immutable and are retained as
a negative ablation. No v2 result existed when the choices below were frozen.

The v1 screening was numerically and deterministically valid but failed the preregistered core gate:

- C0: 61.9768%;
- C1 mean: 63.4057%;
- C2 mean: 62.3062%;
- C2 - C0: +0.3294 percentage point;
- C2 - C1: -1.0994 percentage points;
- in-support retention median: 0.9542;
- worst-seed in-support retention q05: 0.4654.

The failure is mechanistic rather than infrastructural. At epoch 60, v1 C2's normal MSE was about
0.0564 versus 0.0348 for C1, while the direction term remained about 0.2965. The fixed auxiliary
penalty therefore dominated the primary denoising objective. Amplitude modulation was the principal
lower-tail failure (seed 13711 q05 approximately 0.295), even though median retention passed.

## v2 hypothesis

FP-NAA v2 treats fault preservation as a bounded risk constraint, not an everywhere-exact matching
objective. It combines three mechanisms:

1. **Dead-zone fault constraints.** Direction is penalized only below cosine 0.50. Delta gain is
   constrained to [1.05, 1.20], encouraging modest fault amplification while avoiding uncontrolled
   distortion. Once a sample is safe, it contributes no magnitude or direction penalty.
2. **Tail-risk optimization.** Direction and gain violations are aggregated over the largest 10% of
   per-example violations (empirical CVaR). This aligns training with the preregistered q05 safety
   gate instead of allowing the mean to hide a failing tail.
3. **Primary-safe gradient projection.** The auxiliary gradient is projected away from the component
   that conflicts with the normal-reconstruction gradient before the update is applied. The model
   first receives 20 MSE-only epochs, followed by a 10-epoch auxiliary ramp. Gradient cosine and
   conflict frequency are recorded per epoch.

A backend-aligned separation term additionally operates on the 20% of time-frequency patches where
the frozen teacher responds most strongly to the pseudo fault. It requires at least 1.10 times the
teacher's clean-to-fault cosine distance. This replaces the v1 assumption that matching a single
flattened representation delta necessarily improves downstream BEAM ranking.

Every retention row also records delta gain, direction cosine, teacher/student delta norms, and
salient-patch distance gain. These diagnostics are observational only and cannot change a gate.

The architecture, parameter count, data, pseudo-fault families, held-out friction-burst family,
BEATs checkpoint, seeds, optimizer, epochs, backend, and all contribution gates are unchanged. The
authoritative configuration is `configs/experiment/fp_naa_v2.yaml`.

## Why this revision is evidence-based

Multi-objective training can degrade a primary task when auxiliary gradients conflict; PCGrad
projects conflicting components rather than relying only on fixed scalar weights ([Yu et al.,
NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)).
Superquantile/CVaR constraints explicitly target a distribution tail rather than its average
([Roth and Cui, JMLR 2025](https://www.jmlr.org/papers/v26/24-0752.html)). These tools address the
two failures observed in v1 directly: primary-objective interference and poor q05 despite a passing
median.

The DCASE 2026 evidence also supports evaluating synthetic faults through detector geometry rather
than treating every generated sample as automatically useful: the SATLab system screens synthetic
audio before BEATs fine-tuning and reports that sample fidelity/diversity control is important
([DCASE 2026 technical report](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Zhang_111_t2.pdf)).

## Frozen v2 decision rule

The original G2/G3 gates are not relaxed. A v2 screening run may advance only if all core checks
pass, including C2 - C1 >= +0.50 percentage point and in-support retention q05 >= 0.75. Failure does
not permit LOMO or confirmatory execution. Any subsequent methodological change requires another
versioned amendment and a fresh screening run.
