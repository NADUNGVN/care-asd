# FP-NAA post-V10 literature gate

Cutoff: 2026-08-21. Source policy: primary sources only. This is an incremental update to
[`FP_NAA_LITERATURE_UPDATE_20260817.md`](FP_NAA_LITERATURE_UPDATE_20260817.md) and the frozen
Audit-A1 matrix; it does not repeat their signal-level safety search.

## Trigger and decision

The authoritative V10-M run `server02_fp_naa_evidence_preflight_20260821T131202Z` completed
normally but authorized no supplementary expert. All 21 machine/expert certificates respected the
clean activation budget, while every in-support and held-out median evidence gain was zero. This
closes V10 and creates a research-selection question, not permission to relax its gate.

The literature delta yields a **no-go for immediate V11 GPU training**. Three tempting successors
are already prior art or directly weakened by published negative evidence:

1. sparse top-tail rescue is already a submitted train-normal aggregation family and did not beat
   domain consensus in its reported development replay;
2. ordinary calibrated multi-branch fusion is crowded DCASE prior art, including substantially
   more heterogeneous systems than V10;
3. proxy-outlier or pseudo-fault model selection can saturate without ranking real-anomaly
   backends.

Two components remain useful only conditionally: cross-channel predictive residuals as an
orthogonal comparator, and domain-aware normal-score calibration as a transfer control. Neither is
novel by itself. A future contribution would have to be a new safety, identifiability, or
prospective-transfer protocol around them.

## Delta evidence matrix

| Primary source | Result relevant after V10 | CARE-ASD consequence |
|---|---|---|
| [DCASE 2026 official results](https://dcase.community/challenge2026/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring-results) | Current systems combine heterogeneous encoders, backends, stereo views, rank/calibration rules, and ensembles | Generic complementarity, stereo processing, and score fusion are not contribution candidates |
| [Zhou and Wang, 2026](https://arxiv.org/abs/2606.19269) | Backend choice dominates pooling; train-normal fusion is useful; proxy-outlier backend selection saturates and fails | Stop pooling search; do not treat pseudo-fault ranking as evidence of real-anomaly power |
| [Mkrtchian, 2026](https://arxiv.org/abs/2607.04526) | Normal-only domain balance can transfer better than development ranking, but can select degenerate scores without a viability veto | Calibration is a mandatory control, not a power certificate |
| [Jeong and Kim, 2026](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Jeong_66_t2.pdf) | Near-to-far embedding prediction residual and raw-spectral density add orthogonal evidence | Predictive residual is prior art; only a new evaluation/safety claim remains open |
| [Shokriazar et al., 2026](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Moradi_97_t2.pdf) | Frozen BEATs/EAT, Wiener and difference views, Shiftall-LDKNN, fixed fusion, and strict LOMO form a strong existing route | Do not approximately rebuild an expensive published ensemble without an isolated new mechanism |
| [Kajita, 2026](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Kajita_4_t2.pdf) | A 54-component train-normal profile bank includes top-tail rescue; reported development replay favors consensus | Do not rename relaxed V10 tail selection as V11 novelty |
| [Jiang et al., 2026](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Jiang_125_t2.pdf) | Heterogeneous BEATs systems, seeds, augmentation, and score selection report 68.20% development score | A same-backbone fusion claim needs more diversity and a novelty other than fusion |
| [Le Clei et al., 2022](https://openreview.net/pdf?id=7rHwie6nQos) | Agreement among diverse anomaly detectors can support unsupervised model selection | The premise requires genuinely diverse errors; V10's uniformly inactive experts do not satisfy it |

DCASE challenge technical reports are current primary system descriptions, not peer-reviewed
journal or conference evidence. The two 2026 arXiv studies are preprints at this cutoff.

## Candidate-family gate

| Candidate after V10 | Novelty | Leakage/transfer | Compute | Decision |
|---|---|---|---|---|
| Relax V10 median gain or tail penalty | Fails: result-contingent retuning | Reuses the observed gate | Low | **Reject** |
| Sparse top-tail rescue | Fails: direct 2026 prior art | Would be chosen after observing V10 sparsity | Low | **Reject** |
| More final/tap BEATs pooling branches | Weak and crowded | Proxy-fault selection has negative evidence | Medium | **Reject** |
| Rebuild a multi-encoder DCASE ensemble | Components and protocol are published | LOMO can control leakage, but no isolated CARE-ASD claim | High | **Reject as a research contribution** |
| Domain-aware calibration | Established prior art | Useful only with frozen viability and transfer gates | Low | **Retain as control** |
| Near/far predictive-consistency residual | Established component prior art | A new independent safety/transfer endpoint could be defensible | Low to medium | **Conditional comparator** |
| Consolidate the controlled negative safety study | Existing CARE-ASD novelty boundary remains | Uses frozen Audit-A1/A2 and V1-V10 evidence | Low | **Proceed** |

## Requirements before any successor experiment

A successor may be specified only if all conditions are met before implementation:

1. **Distinct mechanism:** it cannot be a threshold, penalty, expert-list, layer, pooling, or loss
   change selected in response to V10.
2. **Bounded novelty:** any cross-channel residual or calibration component is explicitly prior art;
   the contribution must be a new causal, safety, identifiability, or prospective-transfer claim.
3. **Independent evidence:** the three V10 in-support pseudo-fault families cannot select the new
   rule. Any new proxy family and its held-out counterpart must be generated and hash-frozen before
   outcome inspection.
4. **Viability separation:** normal-only calibration may rank only candidates that first pass a
   preregistered coarse viability veto. It cannot establish anomaly power alone.
5. **No evaluation leakage:** evaluation labels remain unread until code, scores, policy, and hashes
   are frozen. Development-machine LOMO is a robustness estimate, not proof of evaluation transfer.
6. **Compute gate:** no GPU training is authorized until a cached or normal-only mechanism probe
   demonstrates a non-degenerate signal and a manuscript claim not already covered by the matrix.

## Reproducible artifact

The curated contract is
`configs/research/fp_naa_post_v10_literature_delta.yaml`. After its source commit is frozen, run:

```bash
uv run care-asd audit literature \
  --config configs/research/fp_naa_post_v10_literature_delta.yaml \
  --repo-root . \
  --output-dir reports/audit/fp_naa_post_v10_literature_delta
```

The generated matrix and claim boundary are evidence for the go/no-go decision. They do not
authorize V11 automatically.
