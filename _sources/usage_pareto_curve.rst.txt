Postprocessing frontier (Pareto curve)
======================================

Use :func:`error_parity.pareto_curve.compute_postprocessing_curve` to compute the fairness–performance frontier across tolerances.

API
---

.. autofunction:: error_parity.pareto_curve.compute_postprocessing_curve

Interpretation
--------------

- The returned ``pandas.DataFrame`` has one row per tolerance tick and columns for each metric and dataset split.
- Use :func:`error_parity.plotting.plot_postprocessing_frontier` to visualize the envelope of the frontier and optional bootstrap confidence intervals.

Related utilities
-----------------

.. autofunction:: error_parity.pareto_curve.fit_and_evaluate_postprocessing
.. autofunction:: error_parity.pareto_curve.get_envelope_of_postprocessing_frontier
.. autofunction:: error_parity.pareto_curve.compute_inner_and_outer_adjustment_ci

