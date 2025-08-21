Quickstart
==========

This guide demonstrates postprocessing a trained model to satisfy a fairness constraint.

Minimal example
---------------

.. code-block:: python

   import numpy as np
   from sklearn.linear_model import LogisticRegression
   from error_parity import RelaxedThresholdOptimizer

   # Assume X, y, group are numpy arrays
   # group must be encoded as integers 0..G-1
   model = LogisticRegression().fit(X, y)

   fair_clf = RelaxedThresholdOptimizer(
       predictor=lambda X: model.predict_proba(X)[:, -1],
       constraint="equalized_odds",
       tolerance=0.05,
   )

   fair_clf.fit(X=X, y=y, group=group)
   y_pred = fair_clf(X=X_test, group=group_test)

Notes
-----

- ``group`` indexing must be contiguous starting at 0. If ``np.max(group) > n_groups-1``, a ``ValueError`` is raised.
- For callable predictors, pass the callable directly as ``predictor``. For scikit-learn API, pass a ``lambda`` to ``predict_proba`` or ``decision_function`` as appropriate.
- Use ``tolerance=0.0`` for strict parity; higher tolerance allows more disparity.

Next steps
----------

- Explore :doc:`usage_threshold_optimizer` for options (costs, norms, diagnostics).
- Compute a :doc:`usage_pareto_curve` to visualize the fairness–performance frontier.
- Evaluate metrics and uncertainty with :doc:`usage_evaluation`.

