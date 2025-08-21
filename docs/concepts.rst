Core concepts
=============

ROC curves and convex hulls
---------------------------

For each group, the model induces an ROC curve of achievable (FPR, TPR) pairs by thresholding scores. The convex hull of this curve characterizes all achievable points by mixing thresholds and, when necessary, randomization.

Randomized thresholds
---------------------

When the optimal target lies strictly inside a segment of the ROC hull, it is realized by a randomized classifier that mixes at most two deterministic thresholds, and, if needed, the diagonal (random) classifier. See :class:`error_parity.classifiers.RandomizedClassifier`.

Fairness constraints
--------------------

Let groups be indexed by ``a, b``. We support constraints expressed on group-specific rates. Examples:

- Equalized odds: constrain distances between (TPR, FPR) pairs across groups.
- Equal opportunity: constrain TPR parity.
- Predictive equality: constrain FPR parity.
- Demographic parity: constrain PPR (positive prediction rate) parity.

Relaxations and \(\ell_p\) norms
----------------------------------

For equalized odds, distances between group ROC points are measured with an \(\ell_p\) norm, e.g., \(\ell_\infty\) (default), \(\ell_1\) (sum of absolute differences), or \(\ell_2\).

Costs and performance
---------------------

The optimizer can compute theoretical cost at the global solution point for user-specified false positive and false negative costs. With unit costs, cost equals error rate. See :meth:`error_parity.threshold_optimizer.RelaxedThresholdOptimizer.cost`.

