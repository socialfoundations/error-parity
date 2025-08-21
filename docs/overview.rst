Overview
========

The ``error-parity`` package provides fast post-processing for any score-based classifier to satisfy user-chosen fairness constraints with optional relaxation. It works by:

- computing group-wise ROC curves and their convex hulls;
- solving a small convex optimization to find the optimal fairness–accuracy operating point under a constraint and tolerance;
- realizing that solution as group-specific binary classifiers, which can include carefully randomized thresholds when the target point lies in the interior of a group's ROC hull.

Key features
------------

- **Drop-in postprocessing**: Wrap any predictor that outputs scores in ``[0, 1]`` or probabilities.
- **Multiple constraints**: Equalized odds (with configurable \(\ell_p\) metric), equal opportunity, predictive equality, demographic parity.
- **Tunable tolerance**: Explore strict (``tolerance=0``) to relaxed constraints (e.g., ``tolerance=0.05``).
- **Pareto frontier**: Compute and visualize fairness–performance trade-offs across tolerances.
- **Metrics and evaluation**: Built-in evaluation and bootstrap uncertainty.
- **Plotting utilities**: Visualize ROC hulls, chosen operating points, and postprocessing frontiers.

Typical workflow
----------------

1. Choose a fairness constraint and tolerance.
2. Fit a ``RelaxedThresholdOptimizer`` on data with labels and group membership.
3. Use the fitted object to predict on new data (requires group membership).
4. Optionally compute a postprocessing curve across tolerances and plot results.

When to use error-parity
------------------------

- You have a trained model that outputs scores (e.g., ``predict_proba``) and you want to enforce fairness constraints without retraining.
- You need to compare models at the same fairness level.
- You want to quantify the fairness–performance frontier under different tolerances.

What this is not
----------------

- ``error-parity`` is a postprocessing library. It does not change the model training objective. It complements in-processing or pre-processing approaches.

See also
--------

- :doc:`quickstart`
- :doc:`concepts`
- :doc:`constraints`
- :doc:`usage_threshold_optimizer`
- :doc:`usage_pareto_curve`
- :doc:`usage_evaluation`
- :doc:`usage_plotting`

