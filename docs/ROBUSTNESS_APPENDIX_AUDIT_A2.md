# Audit-A2 machine/domain robustness appendix

## Scope and immutable inputs

Audit-A2 is a descriptive robustness analysis of three already committed DCASE
2026 development score tables:

- B00: aligned near-only comparator;
- B01: residual replacement; and
- B02: near-primary gated residual.

All tables contain the same 1,400 clip identifiers. Every one of the 28
machine×domain×condition strata contains 50 clips. The analysis reads no audio,
trains no model, changes no threshold, and accesses no hidden evaluation data.
Its strict contract is `configs/experiment/audit_robustness_v1.yaml`.

## Frozen estimands

For each of seven machine sections and each domain (`all`, `source`, `target`),
the appendix reports:

1. AUC and standardized pAUC at maximum FPR 0.1 for B00, B01, and B02;
2. candidate-minus-B00 paired deltas;
3. 2,000 paired-bootstrap replicates, resampled inside
   machine×domain×condition strata; and
4. leave-one-machine-out mean deltas over the retained machine sections.

The original 5,000-replicate frozen global intervals remain the inferential
headline. The new per-machine intervals and leave-one-machine-out ranges test
heterogeneity and sensitivity; they do not replace or retune the original test.

## Headline findings

| Comparison | Mean all-domain pAUC delta | Frozen 95% CI | Machines improved | Leave-one-machine-out range |
|---|---:|---:|---:|---:|
| B01 − B00 | -1.93 pp | [-3.27, -0.29] pp | 2/7 | [-2.49, -1.11] pp |
| B02 − B00 | -0.60 pp | [-1.40, +0.32] pp | 4/7 | [-0.89, +0.02] pp |

B01's pAUC harm is not solely caused by one machine: deleting any single
machine leaves the mean delta negative. Its effect is nevertheless
heterogeneous, including improvements on two machine sections and a tie on one.
B02 is smaller and inconclusive; its leave-one-machine-out range crosses zero.

The domain split exposes the main fragility:

| Comparison | Mean source pAUC delta | Mean target pAUC delta |
|---|---:|---:|
| B01 − B00 | +0.75 pp | -2.14 pp |
| B02 − B00 | +0.33 pp | -1.02 pp |

Both interventions therefore look more favorable in the source domain while
degrading the under-represented target domain on average. This supports a
domain-fragility interpretation. It does not prove a causal mechanism and does
not generalize to hidden evaluation machines.

## Manuscript claim boundary

Supported:

- B01 pAUC harm remains negative under every one-machine deletion on the frozen
  development machines.
- B01 and B02 exhibit heterogeneous machine effects and a source/target
  asymmetry.
- B02 does not demonstrate a stable improvement over B00.

Unsupported:

- every machine is harmed;
- AUC and pAUC respond identically;
- target-domain degradation will recur on unseen evaluation machines;
- the observed domain association is causal; or
- the conclusion applies to learned dual-microphone representations.

## Reproduction

```bash
uv run care-asd audit robustness --config configs/experiment/audit_robustness_v1.yaml --repo-root . --output-dir reports/audit/care_asd_robustness_appendix_v1
```

The output is immutable and includes coverage validation, machine/domain
metrics, paired intervals, frozen global inference, leave-one-machine-out
results, heterogeneity records, SVG figures, a narrative appendix, and hashes.
