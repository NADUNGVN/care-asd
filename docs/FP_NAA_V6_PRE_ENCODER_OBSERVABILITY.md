# FP-NAA v6 pre-encoder observability preflight

## Frozen status and purpose

This preflight was specified after the valid V5 result
`server02_fp_naa_screening_20260818T020554Z` and before any V6 encoder-tap result existed. It is
not a new performance experiment and cannot authorize LOMO. It answers one bounded question:

> Does a frozen shallow or intermediate BEATs representation retain enough normal-only
> counterfactual fault evidence to justify moving the noise-aware adapter before the final encoder
> output?

V5 remains an immutable negative result. Its C2 mean official score was 0.6341576, only +0.0001010
over the capacity-matched C1 mean instead of the registered +0.005 minimum. Its in-support
retention median/q05 was 0.7224/0.1011. Decomposition localized the larger limitation upstream:
the final BEATs representation retained 0.7731 median and 0.1340 q05 of the clean-teacher fault
delta, while the trained adapter retained 0.9351 median and 0.6647 q05 of what reached it.

## Probe contract

The official BEATs Iteration-3 checkpoint and the existing normal-only waveform counterfactuals
are unchanged. Four frozen depths are inspected:

- tap 0: normalized patch projection before BEATs positional convolution;
- tap 4: output after Transformer block 4;
- tap 8: output after Transformer block 8;
- tap 12: final Transformer output.

For every normal training clip, frozen pseudo-fault family, and tap, the probe computes

```text
d_clean = E_k(clean + fault) - E_k(clean)
d_noisy = E_k(noisy + fault) - E_k(noisy)
retention = exp(-abs(log(||d_noisy|| / ||d_clean||))).
```

The three registered in-support families alone determine eligibility. A tap is eligible only when
median retention is at least 0.90 and q05 retention is at least 0.75. The deepest eligible tap is
selected because it has traversed the greatest amount of frozen semantic processing. The held-out
friction-burst family is reported only after this rule is applied and cannot select a tap.

If no tap is eligible, pre-encoder ACTT is rejected before implementation or development-label
scoring. If a tap is eligible, the result merely authorizes a separately preregistered pre-encoder
candidate. All original G2/G3 performance, retention, LOMO, bootstrap, and safety thresholds remain
unchanged.

## Literature boundary

This direction is motivated by, but does not claim, multi-layer representations or internal
noise-aware adapters as new. The official BEATs implementation exposes twelve Transformer blocks
and returns layer results internally. DCASE 2026 systems already use intermediate frozen features,
multi-branch scoring, and cross-attention noise-aware modules inserted inside BEATs. The potential
contribution would have to be the counterfactual observability gate and a later fault-preserving
pre-encoder mechanism, validated without anomaly-label selection.

Primary sources:

- [Official Microsoft BEATs implementation](https://github.com/microsoft/unilm/tree/833df7e7832e5064a281131ee64a481afa8e5b95/beats)
- [BEATs: Audio Pre-Training with Acoustic Tokenizers](https://proceedings.mlr.press/v202/chen23ag.html)
- [DCASE 2026 Task 2 overview](https://arxiv.org/abs/2606.01578)
- [MERL DCASE 2026 noise-aware internal adapters](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Fujimura_17_t2.pdf)
- [GISP@HEU adaptive layer selection](https://dcase.community/documents/challenge2026/technical_reports/DCASE2026_Guan_93_t2.pdf)

