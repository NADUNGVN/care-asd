# FP-NAA V1-V10 bounded negative evidence ledger

This ledger is a supplementary research-process record for the CARE-ASD identifiability manuscript.
It does not combine sequential experiments as independent evidence, alter any frozen gate, or
authorize a successor run. Development anomaly labels, pseudo-fault families, and held-out families
retain the roles declared by each version's preregistration.

| Version | Frozen hypothesis | Decisive result | Decision |
|---|---|---|---|
| V1 | Fixed-penalty fault-preserving adapter can retain injected deltas and beat the MSE C1 adapter | C2 score 62.3062% versus C1 63.4057%; in-support q05 retention 0.4654 versus required 0.75 | Closed without LOMO |
| V2 | Tail-risk objective and primary-safe gradient rule repair V1's weak tail | C2 score 61.0112%, 2.3945 points below C1; 84-85% auxiliary gradient conflict; in-support q05 0.3037 | Closed without LOMO |
| V3 | Exact reference-contraction projection prevents direct target suppression | C2 score 63.4189% versus C1 63.4057%, only +0.0132 point; in-support q05 0.1019 | Closed without LOMO |
| V4 | Reference-only correction with exact target equivariance preserves anomaly deltas | Score 63.3105%, -0.0951 point versus C1; in-support retention 0.7731/0.1340 median/q05 | Closed without LOMO |
| V5 | C1-initialized anchored counterfactual tangent transport restores preservation | Score 63.4158%, only +0.0101 point over C1; in-support retention 0.7224/0.1011 | Closed without LOMO |
| V6 | An earlier frozen BEATs tap retains enough injected-fault evidence to bypass the final frontend | No eligible tap; tap 0 reached only 0.8465/0.4094 in-support median/q05 | Mechanism gate failed |
| V7 | Conditional selected-tap evidence union complements immutable C1 | Implementation condition was not met because V6 selected no tap | Closed without implementation |
| V8 | Capacity-matched layerwise tangent restoration repairs frontend observability | L2 in-support retention 0.7811/0.1092 and held-out retention 0.8311/0.3002 | Mechanism gate failed; no G2 |
| V9 | Localized tap-0 ACTT repair avoids the remaining Transformer blocks | P2 retention 0.6674/0.2433 in-support and 0.6739/0.3695 held-out | Mechanism gate failed; no G2 |
| V10 | Normal-tail-penalized monotone evidence union admits only complementary fixed experts | All 21 machine/expert median gains were zero; authorized experts 0; machine coverage 0 | Mechanism gate failed; no screening |

## Cross-version interpretation

The sequence narrows the observed failure boundary:

1. Preserving synthetic feature deltas in an adapter did not produce a performance advantage over
   normal-MSE adaptation (V1-V5).
2. Moving to earlier representation depth did not expose a tap meeting the registered lower-tail
   preservation requirement (V6-V9).
3. Preserving immutable C1 and adding only certified score evidence did not reveal robust
   complementarity among the three fixed experts (V10).

These results do **not** prove that learned dual-microphone representations, all anomaly-preserving
objectives, or all score ensembles fail. They show that ten bounded formulations did not cross their
own prospective gates. The post-V10 primary-source audit is
[`FP_NAA_POST_V10_LITERATURE_GATE.md`](FP_NAA_POST_V10_LITERATURE_GATE.md); it prohibits a
threshold-relaxed V10 successor and ordinary fusion novelty claims.

## Immutable run index

| Evidence | Run ID |
|---|---|
| V1 screening | `server02_fp_naa_screening_20260817T083029Z` |
| V2 screening | `server02_fp_naa_screening_20260817T120203Z` |
| V3 screening | `server02_fp_naa_screening_20260817T145911Z` |
| V4 screening | `server02_fp_naa_screening_20260817T165229Z` |
| V5 valid rerun | `server02_fp_naa_screening_20260818T020554Z` |
| V6 frontend probe | `server02_fp_naa_frontend_probe_20260818T033118Z` |
| V8 layerwise preflight | `server02_fp_naa_layerwise_preflight_20260818T060510Z` |
| V9 tap-repair preflight | `server02_fp_naa_tap_repair_preflight_20260818T072423Z` |
| V10 evidence preflight | `server02_fp_naa_evidence_preflight_20260821T131202Z` |

V7 has no run because its preregistered implementation condition failed.
