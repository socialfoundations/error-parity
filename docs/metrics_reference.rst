Metrics reference
=================

This page summarizes metrics produced by :mod:`error_parity.evaluation`.

Performance metrics
-------------------

- **accuracy**: fraction of correct predictions.
- **tpr (recall)**: true positive rate.
- **fnr**: false negative rate (1 - TPR).
- **fpr**: false positive rate.
- **tnr**: true negative rate (1 - FPR).
- **precision**: TP / predicted positives.
- **ppr**: positive prediction rate.
- **squared_loss**: mean squared error on scores vs. labels.
- **log_loss**: logistic loss on scores vs. labels.

Fairness aggregations
---------------------

For each metric ``m`` and groups ``a, b``, we compute:

- ``m_ratio = min(m_a, m_b, ...) / max(m_a, m_b, ...)``
- ``m_diff = max(m_a, m_b, ...) - min(m_a, m_b, ...)``

Equalized odds
--------------

- ``equalized_odds_ratio = min(fnr_ratio, fpr_ratio)``
- ``equalized_odds_diff = max(tpr_diff, fpr_diff)``
- ``equalized_odds_diff_l{1,2,inf}`` via \(\ell_1\), \(\ell_2\), \(\ell_\infty\) norms over (TPR, FPR) pairwise differences.

Groupwise outputs
-----------------

Set ``return_groupwise_metrics=True`` in :func:`error_parity.evaluation.evaluate_fairness` or :func:`error_parity.evaluation.evaluate_predictions` to include per-group metrics like ``tpr_group=0``, ``fpr_group=1``, etc.

