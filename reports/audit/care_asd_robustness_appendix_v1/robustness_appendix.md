# Audit-A2 machine/domain robustness appendix

This appendix uses only the frozen B00, B01, and B02 development score tables. No audio, training, tuning, or evaluation-set access occurred.

## Headline robustness results

- **B01 vs B00:** observed mean pAUC delta -1.93 percentage points; frozen paired-bootstrap 95% CI [-3.27, -0.29]; 2/7 machine sections improved; leave-one-machine-out range [-2.49, -1.11].
- **B02 vs B00:** observed mean pAUC delta -0.60 percentage points; frozen paired-bootstrap 95% CI [-1.40, +0.32]; 4/7 machine sections improved; leave-one-machine-out range [-0.89, +0.02].

## Domain split

- **B01:** source +0.75 pp; target -2.14 pp.
- **B02:** source +0.33 pp; target -1.02 pp.

## Interpretation boundary

B01's overall pAUC harm keeps the same sign after every one-machine deletion, but its AUC effects are heterogeneous. B02 remains statistically inconclusive and its pAUC sign is not stable to every one-machine deletion. These are development-set robustness statements, not claims about hidden evaluation machines or all dual-mic methods.
