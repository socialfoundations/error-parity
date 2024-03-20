# error-parity

![Tests status](https://github.com/socialfoundations/error-parity/actions/workflows/python-package.yml/badge.svg)
![PyPI status](https://github.com/socialfoundations/error-parity/actions/workflows/python-publish.yml/badge.svg)
![Documentation status](https://github.com/socialfoundations/error-parity/actions/workflows/python-docs.yml/badge.svg)
![PyPI version](https://badgen.net/pypi/v/error-parity)
![OSI license](https://badgen.net/pypi/license/error-parity)
![Python compatibility](https://badgen.net/pypi/python/error-parity)
<!-- ![PyPI version](https://img.shields.io/pypi/v/error-parity) -->
<!-- ![OSI license](https://img.shields.io/pypi/l/error-parity) -->
<!-- ![Compatible python versions](https://img.shields.io/pypi/pyversions/error-parity) -->

> Work presented as an _oral at ICLR 2024_, titled ["Unprocessing Seven Years of Algorithmic Fairness"](https://openreview.net/forum?id=jr03SfWsBS).


Fast postprocessing of any score-based predictor to meet fairness criteria.

The `error-parity` package can achieve strict or relaxed fairness constraint fulfillment, 
which can be useful to compare ML models at equal fairness levels.

Package documentation available [here](https://socialfoundations.github.io/error-parity/).


## Installing

Install package from [PyPI](https://pypi.org/project/error-parity/):
```
pip install error-parity
```

Or, for development, you can clone the repo and install from local sources:
```
git clone https://github.com/socialfoundations/error-parity.git
pip install ./error-parity
```


## Getting started

> See detailed example notebooks under the [**examples folder**](./examples/).

```py
from error_parity import RelaxedThresholdOptimizer

# Given any trained model that outputs real-valued scores
fair_clf = RelaxedThresholdOptimizer(
    predictor=lambda X: model.predict_proba(X)[:, -1],   # for sklearn API
    # predictor=model,            # use this for a callable model
    constraint="equalized_odds",  # other constraints are available
    tolerance=0.05,               # fairness constraint tolerance
)

# Fit the fairness adjustment on some data
# This will find the optimal _fair classifier_
fair_clf.fit(X=X, y=y, group=group)

# Now you can use `fair_clf` as any other classifier
# You have to provide group information to compute fair predictions
y_pred_test = fair_clf(X=X_test, group=group_test)
```


## How it works

Given a callable score-based predictor (i.e., `y_pred = predictor(X)`), and some `(X, Y, S)` data to fit, `RelaxedThresholdOptimizer` will:
1. Compute group-specific ROC curves and their convex hulls;
2. Compute the `r`-relaxed optimal solution for the chosen fairness criterion (using [cvxpy](https://www.cvxpy.org));
3. Find the set of group-specific binary classifiers that match the optimal solution found.
    - each group-specific classifier is made up of (possibly randomized) group-specific thresholds over the given predictor;
    - if a group's ROC point is in the interior of its ROC curve, partial randomization of its predictions may be necessary.


## Features and implementation road-map

We welcome community contributions for [cvxpy](https://www.cvxpy.org) implementations of other fairness constraints.

Currently implemented fairness constraints:
- [x] equality of odds (Hardt et al., 2016);
  - i.e., equal group-specific TPR and FPR;
  - use `constraint="equalized_odds"`;
- [x] equal opportunity;
  - i.e., equal group-specific TPR;
  - use `constraint="true_positive_rate_parity"`;
- [x] predictive equality;
  - i.e., equal group-specific FPR;
  - use `constraint="false_positive_rate_parity"`;
- [x] demographic parity;
  - i.e., equal group-specific predicted prevalence;
  - use `constraint="demographic_parity"`;


## Citing

```
@inproceedings{
  cruz2024unprocessing,
  title={Unprocessing Seven Years of Algorithmic Fairness},
  author={Andr{\'e} Cruz and Moritz Hardt},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=jr03SfWsBS}
}
```
