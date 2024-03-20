"""Solver for the relaxed equal odds problem.

TODO
- Add option for constraining equality of positive predictions (independence
criterion, aka demographic parity);

"""
from __future__ import annotations

import logging
from itertools import product
from typing import Callable

import numpy as np
from sklearn.metrics import roc_curve

from .cvxpy_utils import (
    compute_fair_optimum,
    ALL_CONSTRAINTS,
    NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE,
)
from .roc_utils import (
    roc_convex_hull,
    calc_cost_of_point,
)
from .classifiers import (
    Classifier,
    RandomizedClassifier,
    EnsembleGroupwiseClassifiers,
)


class RelaxedThresholdOptimizer(Classifier):
    """Class to encapsulate all the logic needed to compute the optimal equal
    odds classifier (with possibly relaxed constraints).
    """

    def __init__(
        self,
        *,
        predictor: Callable[[np.ndarray], np.ndarray],
        constraint: str = "equalized_odds",
        tolerance: float = 0.0,
        false_pos_cost: float = 1.0,
        false_neg_cost: float = 1.0,
        max_roc_ticks: int = 1000,
        seed: int = 42,
        # distance: str = 'max',    # TODO: add option to use l1 or linf distances
    ):
        """Initializes the relaxed equal odds wrapper.

        Parameters
        ----------
        predictor : callable[(np.ndarray), float]
            A trained score predictor that takes in samples, X, in shape
            (num_samples, num_features), and outputs real-valued scores, R, in
            shape (num_samples,).
        constraint : str
            The fairness constraint to use. By default "equalized_odds".
        tolerance : float
            The absolute tolerance for the equal odds fairness constraint.
            Will allow for `tolerance` difference between group-wise ROC points.
        false_pos_cost : float, optional
            The cost of a FALSE POSITIVE error, by default 1.0.
        false_neg_cost : float, optional
            The cost of a FALSE NEGATIVE error, by default 1.0.
        max_roc_ticks : int, optional
            The maximum number of ticks (points) in each group's ROC, when
            computing the optimal fair classifier, by default 1000.
        seed : int
            A random seed used for reproducibility when producing randomized
            classifiers.
        """
        # Save arguments
        self.predictor = predictor
        self.constraint = constraint
        self.tolerance = tolerance
        self.false_pos_cost = false_pos_cost
        self.false_neg_cost = false_neg_cost
        self.max_roc_ticks = max_roc_ticks
        self.seed = seed

        # Validate constraint
        if self.constraint not in ALL_CONSTRAINTS:
            raise ValueError(NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE)

        # Initialize instance variables
        self._groupwise_roc_data: dict = None
        self._groupwise_roc_hulls: dict = None
        self._groupwise_roc_points: np.ndarray = None
        self._groupwise_prevalence: np.ndarray = None
        self._global_roc_point: np.ndarray = None
        self._global_prevalence: float = None
        self._realized_classifier: EnsembleGroupwiseClassifiers = None

    @property
    def groupwise_roc_data(self) -> dict:
        """Group-specific ROC data containing (FPR, TPR, threshold) triplets."""
        return self._groupwise_roc_data

    @property
    def groupwise_roc_hulls(self) -> dict:
        """Group-specific ROC convex hulls achieved by underlying predictor."""
        return self._groupwise_roc_hulls

    @property
    def groupwise_roc_points(self) -> np.ndarray:
        """Group-specific ROC points achieved by solution."""
        return self._groupwise_roc_points

    @property
    def groupwise_prevalence(self) -> np.ndarray:
        """Group-specific prevalence, i.e., P(Y=1|A=a)"""
        return self._groupwise_prevalence

    @property
    def global_roc_point(self) -> np.ndarray:
        """Global ROC point achieved by solution."""
        return self._global_roc_point

    @property
    def global_prevalence(self) -> np.ndarray:
        """Global prevalence, i.e., P(Y=1)."""
        return self._global_prevalence

    def cost(
        self,
        false_pos_cost: float = None,
        false_neg_cost: float = None,
    ) -> float:
        """Computes the theoretical cost of the solution found.

        NOTE: use false_pos_cost==false_neg_cost==1 for the 0-1 loss (the
        standard error rate), which is equal to `1 - accuracy`.

        Parameters
        ----------
        false_pos_cost : float, optional
            The cost of a FALSE POSITIVE error, by default will take the value
            given in the object's constructor.
        false_neg_cost : float, optional
            The cost of a FALSE NEGATIVE error, by default will take the value
            given in the object's constructor.

        Returns
        -------
        float
            The cost of the solution found.
        """
        self._check_fit_status()
        global_fpr, global_tpr = self.global_roc_point

        return calc_cost_of_point(
            fpr=global_fpr,
            fnr=1 - global_tpr,
            prevalence=self._global_prevalence,
            false_pos_cost=false_pos_cost or self.false_pos_cost,
            false_neg_cost=false_neg_cost or self.false_neg_cost,
        )

    def constraint_violation(self, constraint_name: str = None) -> float:
        """Theoretical constraint violation of the LP solution found.

        Parameters
        ----------
        constraint_name : str, optional
            Optionally, may provide another constraint name that will be used
            instead of this classifier's self.constraint;

        Returns
        -------
        float
            The fairness constraint violation.
        """
        self._check_fit_status()

        if constraint_name is not None:
            logging.warning(
                f"Calculating constraint violation for {constraint_name} constraint;\n"
                f"Note: this classifier was fitted with a {self.constraint} constraint;"
            )
        else:
            constraint_name = self.constraint

        if constraint_name not in ALL_CONSTRAINTS:
            raise ValueError(NOT_SUPPORTED_CONSTRAINTS_ERROR_MESSAGE)

        if constraint_name == "equalized_odds":
            return self.equalized_odds_violation()

        elif constraint_name.endswith("rate_parity"):
            constraint_to_error_type = {
                "true_positive_rate_parity": "fn",
                "false_positive_rate_parity": "fp",
                "true_negative_rate_parity": "fp",
                "false_negative_rate_parity": "fn",
            }

            return self.error_rate_parity_constraint_violation(
                error_type=constraint_to_error_type[constraint_name],
            )

        elif constraint_name == "demographic_parity":
            return self.demographic_parity_violation()

        else:
            raise NotImplementedError(
                f"Standalone constraint violation not yet computed for "
                f"constraint='{constraint_name}'."
            )

    def error_rate_parity_constraint_violation(self, error_type: str) -> float:
        """Computes the theoretical violation of an error-rate parity constraint.

        Parameters
        ----------
        error_type : str
            One of the following values:
                "fp", for false positive errors (FPR or TNR parity);
                "fn", for false negative errors (TPR or FNR parity).

        Returns
        -------
        float
            The maximum constraint violation among all groups.
        """
        self._check_fit_status()
        valid_error_types = ("fp", "fn")
        if error_type not in valid_error_types:
            raise ValueError(
                f"Invalid error_type='{error_type}', must be one of "
                f"{valid_error_types}."
            )

        roc_idx_of_interest = 0 if error_type == "fp" else 1

        return self._max_l_inf_between_points(
            points=[
                np.reshape(     # NOTE: must pass an array object, not scalars
                    roc_point[roc_idx_of_interest],  # use only FPR or TPR (whichever was constrained)
                    newshape=(1,))
                for roc_point in self.groupwise_roc_points
            ],
        )

    def equalized_odds_violation(self) -> float:
        """Computes the theoretical violation of the equal odds constraint
        (i.e., the maximum l-inf distance between the ROC point of any pair
        of groups).

        Returns
        -------
        float
            The equal-odds constraint violation.
        """
        self._check_fit_status()

        # Compute l-inf distance between each pair of groups
        return self._max_l_inf_between_points(
            points=self.groupwise_roc_points,
        )

    def demographic_parity_violation(self) -> float:
        """Computes the theoretical violation of the demographic parity constraint.

        That is, the maximum distance between groups' PPR (positive prediction
        rate).

        Returns
        -------
        float
            The demographic parity constraint violation.
        """
        self._check_fit_status()

        # Compute groups' PPR (positive prediction rate)
        return self._max_l_inf_between_points(  # TODO: check
            points=[
                # NOTE: must pass an array object, not scalars
                np.reshape(
                    group_tpr * group_prev + group_fpr * (1 - group_prev),
                    newshape=(1,),
                )
                for (group_fpr, group_tpr), group_prev in zip(self.groupwise_roc_points, self.groupwise_prevalence)
            ],
        )

    @staticmethod
    def _max_l_inf_between_points(points: list[float | np.ndarray]) -> float:
        # Number of points (should correspond to the number of groups)
        n_points = len(points)

        # Compute l-inf distance between each pair of groups
        l_inf_constraint_violation = [
            (np.linalg.norm(points[i] - points[j], ord=np.inf), (i, j))
            for i, j in product(range(n_points), range(n_points))
            if i < j
        ]

        # Return the maximum
        max_violation, (groupA, groupB) = max(l_inf_constraint_violation)
        logging.info(
            f"Maximum fairness violation is between "
            f"group={groupA} (p={points[groupA]}) and "
            f"group={groupB} (p={points[groupB]});"
        )

        return max_violation

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        group: np.ndarray,
        y_scores: np.ndarray = None,
    ):
        """Fit this predictor to achieve the (possibly relaxed) equal odds
        constraint on the provided data.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        y : np.ndarray
            The input labels.
        group : np.ndarray
            The group membership of each sample.
            Assumes groups are numbered [0, 1, ..., num_groups-1].
        y_scores : np.ndarray, optional
            The pre-computed model predictions on this data.

        Returns
        -------
        callable
            Returns self.
        """

        # Compute group stats
        self._global_prevalence = np.sum(y) / len(y)

        unique_groups = np.unique(group)
        num_groups = len(unique_groups)
        if np.max(unique_groups) > num_groups - 1:
            raise ValueError(
                f"Groups should be numbered starting at 0, and up to "
                f"num_groups-1. Got {num_groups} groups, but max value is "
                f"{np.max(unique_groups)} != num_groups-1 == {num_groups-1}."
            )

        # Relative group sizes for LN and LP samples
        group_sizes_label_neg = np.array(
            [np.sum(1 - y[group == g]) for g in unique_groups]
        )
        group_sizes_label_pos = np.array([np.sum(y[group == g]) for g in unique_groups])

        if np.sum(group_sizes_label_neg) + np.sum(group_sizes_label_pos) != len(y):
            raise RuntimeError("Failed sanity check. Are you using non-binary labels?")

        # Convert to relative sizes
        group_sizes_label_neg = group_sizes_label_neg.astype(float) / np.sum(
            group_sizes_label_neg
        )
        group_sizes_label_pos = group_sizes_label_pos.astype(float) / np.sum(
            group_sizes_label_pos
        )

        # Compute group-wise prevalence rates
        self._groupwise_prevalence = np.array(
            [np.mean(y[group == g]) for g in unique_groups]
        )

        # Compute group-wise ROC curves
        if y_scores is None:
            y_scores = self.predictor(X)

        # Flatten y_scores array if needed
        if isinstance(y_scores, np.ndarray) and len(y_scores.shape) > 1:
            y_scores = y_scores.ravel()

        self._groupwise_roc_data = dict()
        for g in unique_groups:
            group_filter = group == g

            roc_curve_data = roc_curve(
                y[group_filter],
                y_scores[group_filter],
            )

            # Check if max_roc_ticks is exceeded
            fpr, tpr, thrs = roc_curve_data
            if self.max_roc_ticks is not None and len(fpr) > self.max_roc_ticks:
                indices_to_keep = np.arange(
                    0, len(fpr), len(fpr) / self.max_roc_ticks
                ).astype(int)

                # Bottom-left (0,0) and top-right (1,1) points must be kept
                indices_to_keep[-1] = len(fpr) - 1
                roc_curve_data = (
                    fpr[indices_to_keep],
                    tpr[indices_to_keep],
                    thrs[indices_to_keep],
                )

            self._groupwise_roc_data[g] = roc_curve_data

        # Compute convex hull of each ROC curve
        self._groupwise_roc_hulls = dict()
        for g in unique_groups:
            group_fpr, group_tpr, _group_thresholds = self._groupwise_roc_data[g]

            curr_roc_points = np.stack((group_fpr, group_tpr), axis=1)
            curr_roc_points = np.vstack(
                (curr_roc_points, [1, 0])
            )  # Add point (1, 0) to ROC curve

            self._groupwise_roc_hulls[g] = roc_convex_hull(curr_roc_points)

        # Find the group-wise optima that fulfill the fairness criteria
        self._groupwise_roc_points, self._global_roc_point = compute_fair_optimum(
            fairness_constraint=self.constraint,
            tolerance=self.tolerance,
            groupwise_roc_hulls=self._groupwise_roc_hulls,
            group_sizes_label_pos=group_sizes_label_pos,
            group_sizes_label_neg=group_sizes_label_neg,
            groupwise_prevalence=self.groupwise_prevalence,
            global_prevalence=self.global_prevalence,
            false_positive_cost=self.false_pos_cost,
            false_negative_cost=self.false_neg_cost,
        )

        # Construct each group-specific classifier
        all_rand_clfs = {
            g: RandomizedClassifier.construct_at_target_ROC(
                predictor=self.predictor,
                roc_curve_data=self._groupwise_roc_data[g],
                target_roc_point=self._groupwise_roc_points[g],
                seed=self.seed,
            )
            for g in unique_groups
        }

        # Construct the global classifier (can be used for all groups)
        self._realized_classifier = EnsembleGroupwiseClassifiers(
            group_to_clf=all_rand_clfs
        )
        return self

    def __call__(self, X: np.ndarray, *, group: np.ndarray) -> np.ndarray:
        """Generate predictions for the given input data."""
        return self._realized_classifier(X, group)

    def predict(self, X: np.ndarray, *, group: np.ndarray) -> np.ndarray:
        """Generate predictions for the given input data.

        Parameters
        ----------
        X : np.ndarray
            Input samples.
        group : np.ndarray
            Input sensitive groups.

        Returns
        -------
        np.ndarray
            A sequence of predictions, one per input sample and input group.
        """
        return self(X, group=group)

    def _check_fit_status(self, raise_error: bool = True) -> bool:
        """Checks whether this classifier has been fit on some data.

        Parameters
        ----------
        raise_error : bool, optional
            Whether to raise an error if the classifier is uninitialized
            (otherwise will just return False), by default True.

        Returns
        -------
        is_fit : bool
            Whether the classifier was already fit on some data.

        Raises
        ------
        RuntimeError
            If `raise_error==True`, raises an error if the classifier is
            uninitialized.
        """
        if self._realized_classifier is None:
            if not raise_error:
                return False

            raise RuntimeError(
                "This classifier has not yet been fitted to any data.")

        return True
