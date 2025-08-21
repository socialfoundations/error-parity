FAQ
===

Do I need to retrain my model?
------------------------------

No. ``error-parity`` is a postprocessing method that wraps a score-based predictor.

What if my groups are not 0..G-1?
---------------------------------

Encode them to integers starting at 0 before calling ``fit`` or prediction. Noncontiguous encodings raise a ``ValueError``.

Can I use decision_function instead of predict_proba?
-----------------------------------------------------

Yes, pass ``predictor=lambda X: model.decision_function(X)``. Ensure higher values indicate higher likelihood of the positive class.

What tolerance should I use?
----------------------------

Start with ``tolerance=0.0`` (strict). Increase gradually to explore trade-offs using the postprocessing curve utilities.

How do I get uncertainty estimates?
-----------------------------------

Use :func:`error_parity.evaluation.evaluate_predictions_bootstrap` and the plotting utilities for confidence intervals on frontiers.

