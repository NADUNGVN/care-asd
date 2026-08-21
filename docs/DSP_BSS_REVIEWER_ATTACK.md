# CARE-ASD: BSS Reviewer #2 attack

Review basis: repository state reviewed from commit
`0875bdadd30e37b3a053d4bf0e24ebea854a3326`, followed by a manuscript-only scientific hardening
pass. No frozen experiment was reopened.

# Executive verdict

The empirical record is unusually transparent for a negative signal-processing result, but the
pre-hardening theoretical centerpiece was rejectable. Its change-of-basis construction established
non-unique latent coordinates while speaking as though the transformed variables retained physical
machine/environment provenance. That is not generally valid. It also moved from convolutional
mixing to an arbitrary invertible filter transformation without causal, stable, or physical
admissibility conditions.

The hardened version repairs the logical overreach by splitting the claim into (i) an exact but
modest instantaneous/local-bin latent-factor lemma and (ii) a separate decision-theoretic
non-certifiability proposition over a declared future-fault class containing both retention-benign
and adverse extensions along the suppressed direction. This
is defensible, but its theoretical novelty is limited and must not be sold as a new BSS
identifiability theorem. The strongest contribution remains the frozen, decision-oriented safety
audit and its ASD-specific empirical boundary.

# Strongest rejection argument

The single strongest rejection argument is that the original Proposition 1 confused **coordinate
non-uniqueness with physical semantic non-identifiability**. From
\(m'=m+\alpha e\), it does not follow that \(m'\) is a physically admissible machine-origin source.
If machine and environment labels denote provenance or location, the transformed pair may leave the
physical model class. A generic \(s'=Ts\) proof therefore cannot support the paper's central claim
that identical observations admit two physically valid source explanations requiring different
removal decisions.

This objection is valid. The manuscript now concedes it rather than hiding it. Physical source-label
non-identifiability is no longer claimed as proven. If an editor or reviewer requires such a theorem
as the paper's main novelty, the current evidence is insufficient and the paper should be rejected
rather than rescued by stronger wording.

# Proposition attack

## Pre-hardening logical chain

The earlier argument implicitly used the following chain:

1. \(x=Hs=HT^{-1}Ts\);
2. a non-diagonal \(T\) mixes the named coordinates;
3. both decompositions are physically admissible machine/environment explanations;
4. different semantic explanations imply different unseen-fault safety decisions.

Only step 1 follows from linear algebra without further assumptions. Step 2 is true for latent
coordinates. Step 3 requires a physical model class closed under the chosen transformation, which
was not defined and is implausible when labels encode source provenance. Step 4 additionally needs a
declared class of future faults and a retention rule.

## Hardened result

The revised formalization defines an unstructured latent-factor class
\(\mathcal{M}_{\mathrm{lat}}\). In that class, non-diagonal \(T\) is admissible and exact sample-wise
observation equivalence follows. The result is now **Lemma 1**, and it explicitly does not attach
physical source labels to transformed coordinates.

The revised **Proposition 1** fixes the normal physical model and compares two future-fault
extensions that are identical on all normal observations. One extension avoids the suppressed
direction; the other contains a fault increment whose retention would fail. A normal-only rule must
make the same decision in both. Uniform fault-safety certification is therefore unavailable over
that declared extension class. This is a valid decision-theoretic result, but it is conditional
and close to a no-free-lunch observation: if future faults are unconstrained, normal data cannot
certify their preservation under nontrivial removal.

## What Proposition 1 does not establish

- It does not prove two physically labelled normal source decompositions exist.
- It does not assign a probability to the adverse future-fault world.
- It does not show that every processor suppresses a fault direction.
- It does not prevent an identity or always-abstain rule from being retention-safe.
- It does not show that ICA, BSS, beamforming, supervised separation, or learned fusion fails.
- It does not prove a statement for arbitrary convolutional filters.

# Semantic admissibility

An admissible **latent factorization** is now defined as a nonsingular instantaneous/local-bin
factorization with finite-power components and no semantic constraints. An admissible **physical
model** must additionally respect asserted source provenance, geometry, propagation, support, and
source-family restrictions. The repository does not contain measurements that characterize the
latter class fully.

The semantic objection is therefore resolved by scope, not by proving the stronger claim. The
numeric \(T\) example remains useful to show ambiguity beyond scale/permutation in the unstructured
factor model. It is no longer offered as a constructive physical counterexample. A fully physical
two-world non-uniqueness theorem remains **OPEN** and is not needed for the revised, weaker
certification claim.

# Convolutional scope

The original phrase “or more generally an invertible matrix of admissible filters” was unsupported.
Algebraic invertibility of \(T(z)\) does not ensure that \(T^{-1}(z)\) is causal and stable, that FIR
structure is preserved, or that \(H(z)T^{-1}(z)\) remains compatible with acoustic propagation.

The formal lemma is now limited to an instantaneous mixture or a local frequency-bin
representation with a nonsingular constant complex matrix. The time-domain convolutional model is
retained only as physical motivation. This is scientifically narrower and correct. A general
convolutional theorem remains **OPEN** and is not claimed.

# BSS/ICA assumptions

A BSS reviewer is correct that source identifiability can be restored under stronger assumptions.
Classical ICA uses statistical independence and suitable non-Gaussianity and recovers components up
to conventional ambiguities; see [Comon (1994)](https://doi.org/10.1016/0165-1684(94)90029-9) and
[Hyvärinen and Oja (2000)](https://doi.org/10.1016/S0893-6080(00)00026-5). Convolutive separation
work states explicit source and mixing conditions; relevant primary examples include
[Nguyen Thi and Jutten (1995)](https://doi.org/10.1016/0165-1684(95)00052-F) and
[joint-diagonalization conditions for convolutive mixtures](https://doi.org/10.1016/j.sigpro.2008.02.003).

CARE-ASD now claims only a weak model without guaranteed independence, non-Gaussianity, sparsity,
temporal/nonstationary identifiability, known geometry or paths, source-specific generative models,
supervised labels, reference-only intervals, sufficiently diverse mixtures, additional spatial
observations, or a validated fault-support model. Successful BSS under such added structure is not a
counterexample to the revised proposition.

The existing Audit-A1 literature matrix is a bounded direct-ASD audit, not an exhaustive theoretical
BSS/ICA review. The four theoretical sources above were verified against publisher or official
records during hardening, but all metadata still requires final author-side bibliography checking.

# Unseen-fault logic

The algebraic lemma itself says nothing about an unseen fault \(q\). The application logic is now
separated:

1. unstructured normal latent coordinates are not uniquely determined without added assumptions;
2. correlation alone therefore supplies no physical nuisance label;
3. independently, the declared future-fault class contains both retention-benign and adverse
   extensions along a suppressed direction;
4. a nontrivial normal-only removal decision cannot be uniformly retention-certified over that
   class; and
5. controlled CARE-ASD experiments test whether particular proxies and controllers satisfy the
   empirical joint safety/efficacy rules.

Step 4 is the formal proposition. Step 5 is empirical. The paper no longer presents the controlled
failures as proof of the proposition or the proposition as a prediction that B01/B02 must fail.

# Audit-method novelty attack

The six audit elements—controlled comparator, known components, separate attenuation/retention,
selector safety, frozen stop rules, and an untouched holdout—are individually standard or close to
standard good practice. Calling the collection a fundamentally new signal-processing theory would
be marketing. The defensible contribution is narrower: a **decision-oriented audit specification
for contaminated-reference preprocessing in which downstream utility and component safety are
non-substitutable estimands**.

The revised protocol defines processor, selector, controlled safe-use label, downstream utility,
environmental attenuation, fault retention, false-safe event, coverage, and frozen audit decision.
That definition makes the procedure reproducible by another researcher and prevents four common
category errors: equating signal modification with denoising, equating utility with fault safety,
equating abstention with a useful safe region, and equating retention without attenuation with
successful cancellation.

The methodology is reusable in principle for contaminated reference channels in acoustic,
vibration, or biomedical settings, but validation outside ASD is untested. Methodological novelty is
therefore **PARTIALLY RESOLVED**: the integration is a credible contribution, while priority over all
similar safety frameworks has not been established.

# Statistical attack

The paired bootstrap resamples development files separately by condition within each
machine-section stratum and preserves B00/candidate pairing. This matches the frozen conditional
file-level estimand. It does not establish independence among clips originating from a common
recording process, does not resample acquisition sessions, and does not sample a population of new
machine types. Calling the resulting intervals general confidence intervals would be
pseudo-replicative overreach. The manuscript now states the resampling unit, exchangeability
condition, and population limits.

All main models use one locked training realization, seed 13711. Evaluation bootstrap does not
measure optimizer or initialization variability. The main comparison remains conditional on that
realization; this is an **OPEN** generalization limitation, not something retrospective seeds should
repair.

Mean AUC is the registered primary endpoint. B01 did not improve it; its interval includes zero.
The harmful pAUC interval is a prospectively specified secondary result. The manuscript must not
compress these into “B01 significantly failed.” The exact DCASE harmonic metric \(\Omega\) is a
post-hoc descriptive derivation from frozen predictions and cannot replace the paired estimand.
Real/emulated results are descriptive. V1–V10 are sequential and outcome-informed, so they do not
provide ten tests, ten replications, or multiplicity-adjusted confirmatory evidence.

# External-validity attack

Only ToyCar and fan are real synchronized development machine types; five machine types are
emulated. Seven development types are not a random population of machines. The detector is one
official-compatible autoencoder and does not represent modern learned stereo systems. Controlled
faults are injected proxies with known components, not real mechanical fault provenance across
rooms, devices, nonlinear propagation, or sensor mismatch. SAFE-REF evaluates one post-correction
selector family; AP-CARE evaluates one later frozen controller and proxy generator.

These limitations prevent claims about DCASE evaluation machines, real industrial faults, all
selectors, all reference cancellers, or dual-channel ASD generally. They do not erase the reported
conditional effects or the value of the audit protocol.

# Claims that survived the attack

- A far microphone can be useful without being a noise-only reference.
- Cross-channel correlation alone is not a component-origin or fault-safety certificate.
- B01 did not improve registered primary mean AUC under the locked comparator and harmed the frozen
  secondary pAUC endpoint.
- B02 did not demonstrate improvement; its intervals do not establish equivalence or universal
  safety.
- The corrected SAFE-REF selector had high false-safe behavior within its controlled family.
- AP-CARE G1 did not jointly establish useful attenuation and the declared retention/mechanism
  checks on its untouched internal holdout.
- The tested observables/controllers did not establish a reliable safe-removal region under the
  declared DCASE-2026-aligned development regime.
- The audit keeps downstream utility, component efficacy, retention, selector safety, and chronology
  as separate, non-substitutable evidence.

# Claims that required weakening

- “Physical source-label non-uniqueness” became “unstructured latent-factor non-uniqueness.”
- The arbitrary convolutional change-of-basis claim became an instantaneous/local-bin lemma.
- “Normal observations cannot identify safe removal” became a uniform-certification limitation
  conditional on a declared future-fault class that includes the suppressed direction.
- The audit “methodological novelty” became an operational integration of largely established
  practices, not invention of each component.
- B01/B02 became two tested interventions rather than representatives of all BSS/cancellation
  methods.
- SAFE-REF/AP-CARE failure became selector- and generator-specific rather than evidence against all
  normal-only selectors.
- Bootstrap intervals became conditional file-level uncertainty, not deployment-wide or
  training-seed uncertainty.

# Claim-by-claim Reviewer #2 ledger

| Claim | Best objection | Is objection valid? | Required response/fix | Status |
|---|---|---|---|---|
| 1. The identifiability formulation is novel. | A generic \(x=Hs=HT^{-1}Ts\) ambiguity is textbook factor-model algebra, not a new source-identifiability result. | Yes. | Claim novelty in the contaminated-reference safety formulation and bounded decision consequence, not the algebra itself; position against ICA/BSS. | PARTIALLY RESOLVED |
| 2. The audit protocol is novel. | Its ingredients are standard good experimental practice. | Yes in part. | Define a reusable decision contract and claim integration/operationalization rather than invention of each element. | PARTIALLY RESOLVED |
| 3. The audit is more than good practice. | No independent-domain validation or comparison to an existing safety framework is provided. | Yes. | Present it as a reusable ASD-tested specification; avoid universal methodological priority. | PARTIALLY RESOLVED |
| 4. B01 establishes the danger of suppression. | B01 is a broad residual-replacement stress test and can be attacked as a straw man against modern BSS. | Yes. | Label it a stress test; do not use it as a proxy for all denoisers; rely on component evidence only for tested mechanisms. | RESOLVED |
| 5. B02 represents conservative processing. | It is one architecture with a changed two-branch interface, not the class of conservative methods. | Yes. | State capacity/protocol control, architectural change, and non-representativeness. | RESOLVED |
| 6. SAFE-REF shows normal-only selectors fail. | One corrected selector and one synthetic family cannot generalize to other features or supervision. | Yes. | Restrict to the tested selector/observables/generator and list structures that could succeed. | RESOLVED |
| 7. AP-CARE supports the main conclusion. | It was designed after SAFE-REF and high retention may merely reflect negligible attenuation. | Yes. | Treat it as a distinct later prospective mechanism gate; report retention and attenuation separately and avoid replication language. | RESOLVED |
| 8. Controlled components represent faults. | Injected proxies have generator-known identity but are not physical ground truth for real mechanical failures. | Yes. | Use “fault proxy,” disclose the generator scope, and retain external-validity limitation. | RESOLVED |
| 9. Inference is statistically robust. | One training seed leaves initialization uncertainty unmeasured. | Yes. | Make inference conditional on seed 13711; do not add retrospective seeds. | OPEN |
| 10. Clip bootstrap gives valid confidence intervals. | File-level resampling can be pseudo-replication if clips share unmodelled acquisition dependence and does not sample machine populations. | Potentially yes. | State the unit and exchangeability assumption; limit intervals to frozen evaluation observations/strata. | PARTIALLY RESOLVED |
| 11. Results characterize real dual-mic ASD. | Five of seven development machine types are emulated and only two are real synchronized types. | Yes. | Keep real/emulated results descriptive and prohibit population generalization. | OPEN |
| 12. pAUC establishes B01 failure. | Mean AUC was primary and its interval includes zero. | Yes if pAUC is presented as primary. | Report AUC non-improvement first; label pAUC harm prospectively specified but secondary. | RESOLVED |
| 13. Post-hoc \(\Omega\) strengthens inference. | Computing the official metric after outcome inspection can become endpoint switching. | Yes. | Label it derived post hoc from frozen predictions, descriptive, and secondary to frozen paired estimands. | RESOLVED |
| 14. Successful DCASE dual-channel systems contradict CARE-ASD. | Positive systems show that far-channel information can help. | Not against the bounded claim. | Distinguish auxiliary usefulness from certification of safely removable environmental energy; describe fusion/representation uses neutrally. | NOT A VALID OBJECTION |
| 15. V1–V10 are robustness evidence. | They are sequential, outcome-informed adaptations with large researcher degrees of freedom. | Yes. | Use only as transparent research history, narrowing evidence, and support for stopping; never pool as replications. | RESOLVED |
| 16. SAFE-REF remained prospective. | The corrected protocol followed inspection of a failed run, and the 20% criterion was later diagnosed as infeasible. | Yes. | Preserve the timestamped chronology, post-correction label, unchanged gate, and conclusions independent of coverage alone. | RESOLVED |
| 17. The conclusion establishes an identifiability limit. | The theorem did not prove physical semantic ambiguity; empirical tests cover few methods and synthetic families. | Yes. | Replace the physical theorem with the latent lemma plus conditional decision proposition; bind empirical claims to tested regimes. | PARTIALLY RESOLVED |

# Remaining blockers

1. **BSS expert sign-off:** an independent source-separation theorist should verify that the revised
   lemma/proposition boundary is stated correctly and that the manuscript never slides back from
   latent coordinates to physical provenance.
2. **Theoretical novelty positioning:** the direct-ASD literature matrix does not exhaust ICA/BSS,
   decision-theoretic safety, or multichannel cancellation literature. A final publisher-verified
   bibliography and novelty search is required.
3. **File-level dependence:** no frozen acquisition-cluster variable supports a stronger bootstrap.
   The limitation is disclosed but cannot be eliminated from current artifacts.
4. **Training and external validity:** one seed, one detector family, two real machine types, and
   synthetic fault proxies remain irreducible limits of this frozen study.
5. **Journal presentation:** figures, bibliography, formatting, author approvals, and independent
   line-by-line scientific review remain necessary before submission.

# Recommendation

**INTERNAL EXPERT REVIEW REQUIRED**

The hardened manuscript is appropriate to send to an independent BSS/multichannel DSP expert for a
pre-submission review. It is not yet ready for journal submission. The recommendation would move
toward external pre-submission review only after the expert confirms the semantic/model-class
boundary and the authors complete the theoretical literature and bibliography audit.
