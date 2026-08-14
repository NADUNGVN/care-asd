# Phase 8: CARE residual failure analysis

Phase 7 rejected residual-only CARE (B01) as a replacement for the aligned
near-channel baseline (B00). Phase 8 is a frozen, post-hoc diagnostic analysis,
not a further development-set selection loop.

For every labelled development-test clip, the analysis joins:

- the committed B00 and B01 anomaly scores;
- the exact, frame-aligned B00 near and B01 residual vector caches;
- machine type, section, source/target domain, and normal/anomaly condition.

It reports B01−B00 score shifts by machine/domain/condition and the Spearman
association between score shift and residual-minus-near mean log-Mel displacement.
A negative displacement indicates that CARE removed log-Mel energy. This is not
a calibrated physical energy ratio and must not be reported as one.

Its output explains the failed B01 hypothesis; it cannot set CARE gate values,
select a model, or substantiate a later fusion architecture. A prospective
multi-view/gating experiment, if justified, requires a new predeclared protocol.
