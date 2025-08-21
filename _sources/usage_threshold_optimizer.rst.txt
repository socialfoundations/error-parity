Using the RelaxedThresholdOptimizer
===================================

The :class:`error_parity.threshold_optimizer.RelaxedThresholdOptimizer` wraps a score-based predictor and realizes a fairness-constrained classifier.

Constructor
-----------

.. autoclass:: error_parity.threshold_optimizer.RelaxedThresholdOptimizer
   :members:
   :undoc-members:
   :show-inheritance:

Tips
----

- Ensure ``group`` values are integers ``0..G-1``.
- If your model returns a 2-D array of probabilities, the optimizer will use the last column (``[:, -1]``).
- Control the solution search resolution with ``max_roc_ticks`` if your ROC arrays are large.
- Use ``l_p_norm`` with ``constraint="equalized_odds"`` to pick \(\ell_1\), \(\ell_2\), or \(\ell_\infty\).
- Use ``false_pos_cost`` and ``false_neg_cost`` to reflect asymmetric error costs; the method ``cost`` reports the theoretical cost at the global solution point.

