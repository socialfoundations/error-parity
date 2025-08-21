Constraints
===========

This page summarizes the implemented constraints and how to select them.

Available constraints
---------------------

- **Equalized odds** (default): equalize both TPR and FPR across groups, up to a tolerance.

  - Select with ``constraint="equalized_odds"``.
  - Relaxation via \(\ell_p\) norm between group ROC points; choose ``l_p_norm``.

- **Equal opportunity**: equalize TPR across groups.

  - Select with ``constraint="true_positive_rate_parity"``.

- **Predictive equality**: equalize FPR across groups.

  - Select with ``constraint="false_positive_rate_parity"``.

- **Demographic parity**: equalize positive prediction rate (PPR) across groups.

  - Select with ``constraint="demographic_parity"``.

Tolerance
---------

All constraints accept a nonnegative ``tolerance`` parameter specifying the maximum allowed disparity according to the constraint's metric. ``tolerance=0.0`` enforces strict parity.

Practical guidance
------------------

- Use equalized odds when both types of errors matter and the base rates differ by group.
- Use equal opportunity when minimizing false negatives for positives is paramount.
- Use predictive equality when minimizing false positives for negatives is paramount.
- Use demographic parity when the rate of positive decisions itself should be similar across groups.

