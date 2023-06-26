"""Solver for the relaxed equal odds problem.

TODO
- Add option for constraining only equality of FPR or TPR (currently it must be 
both -> equal odds);
- Add option for constraining equality of positive predictions (independence
criterion, aka demographic parity);

"""
import logging
from itertools import product
from typing import Callable

import numpy as np
from sklearn.metrics import roc_curve

from .cvxpy_utils import compute_equal_odds_optimum
from .roc_utils import (
    roc_convex_hull,
    plot_polygon_edges,
    calc_cost_of_point,
)
from .classifiers import (
    Classifier,
    RandomizedClassifier,
    EnsembleGroupwiseClassifiers,
)


class RelaxedEqualOdds(Classifier):
    """Class to encapsulate all the logic needed to compute the optimal equal
    odds classifier (with possibly relaxed constraints).
    """

    def __init__(
            self,
            predictor: Callable[[np.ndarray], np.ndarray],
            tolerance: float,
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
        self.tolerance = tolerance
        self.false_pos_cost = false_pos_cost
        self.false_neg_cost = false_neg_cost
        self.max_roc_ticks = max_roc_ticks
        self.seed = seed

        # Initialize instance variables
        self._all_roc_data: dict = None
        self._all_roc_hulls: dict = None
        self._groupwise_roc_points: np.ndarray = None
        self._global_roc_point: np.ndarray = None
        self._global_prevalence: float = None
        self._realized_classifier: EnsembleGroupwiseClassifiers = None

    @property
    def groupwise_roc_points(self) -> np.ndarray:
        return self._groupwise_roc_points

    @property
    def global_roc_point(self) -> np.ndarray:
        return self._global_roc_point

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
    
    def constraint_violation(self) -> float:
        """This method should be part of a common interface between different
        relaxed-constraint classes.

        Returns
        -------
        float
            The fairness constraint violation.
        """
        return self.equal_odds_violation()

    def equal_odds_violation(self) -> float:
        """Computes the theoretical violation of the equal odds constraint 
        (i.e., the maximum l-inf distance between the ROC point of any pair
        of groups).

        Returns
        -------
        float
            The equal-odds constraint violation.
        """
        self._check_fit_status()

        n_groups = len(self.groupwise_roc_points)

        # Compute l-inf distance between each pair of groups
        linf_constraint_violation = [
            (np.linalg.norm(
                self.groupwise_roc_points[i] - self.groupwise_roc_points[j],
                ord=np.inf), (i, j))
            for i, j in product(range(n_groups), range(n_groups))
            if i < j
        ]

        # Return the maximum
        max_violation, (groupA, groupB) = max(linf_constraint_violation)
        logging.info(
            f"Maximum fairness violation is between "
            f"group={groupA} (p={self.groupwise_roc_points[groupA]}) and "
            f"group={groupB} (p={self.groupwise_roc_points[groupB]});"
        )

        return max_violation


    def fit(self, X: np.ndarray, y: np.ndarray, group: np.ndarray, y_scores: np.ndarray = None):
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
        if np.max(unique_groups) > num_groups-1:
            raise ValueError(
                f"Groups should be numbered starting at 0, and up to "
                f"num_groups-1. Got {num_groups} groups, but max value is "
                f"{np.max(unique_groups)} != num_groups-1 == {num_groups-1}."
            )

        # Relative group sizes for LN and LP samples
        group_sizes_label_neg = np.array([
            np.sum(1 - y[group == g]) for g in unique_groups
        ])
        group_sizes_label_pos = np.array([
            np.sum(y[group == g]) for g in unique_groups
        ])

        if np.sum(group_sizes_label_neg) + np.sum(group_sizes_label_pos) != len(y):
            raise RuntimeError(f"Failed sanity check. Are you using non-binary labels?")

        # Convert to relative sizes
        group_sizes_label_neg = group_sizes_label_neg.astype(float) / np.sum(group_sizes_label_neg)
        group_sizes_label_pos = group_sizes_label_pos.astype(float) / np.sum(group_sizes_label_pos)

        # Compute group-wise ROC curves
        if y_scores is None:
            y_scores = self.predictor(X)

        self._all_roc_data = dict()
        for g in unique_groups:
            group_filter = group == g

            roc_curve_data = roc_curve(
                y[group_filter],
                y_scores[group_filter],
            )

            # Check if max_roc_ticks is exceeded
            fpr, tpr, thrs = roc_curve_data
            if self.max_roc_ticks is not None and len(fpr) > self.max_roc_ticks:
                indices_to_keep = np.arange(0, len(fpr), len(fpr) / self.max_roc_ticks).astype(int)

                # Bottom-left (0,0) and top-right (1,1) points must be kept
                indices_to_keep[-1] = len(fpr) - 1
                roc_curve_data = (fpr[indices_to_keep], tpr[indices_to_keep], thrs[indices_to_keep])

            self._all_roc_data[g] = roc_curve_data

        # Compute convex hull of each ROC curve
        self._all_roc_hulls = dict()
        for g in unique_groups:
            group_fpr, group_tpr, _group_thresholds = self._all_roc_data[g]

            curr_roc_points = np.stack((group_fpr, group_tpr), axis=1)
            curr_roc_points = np.vstack((curr_roc_points, [1, 0]))  # Add point (1, 0) to ROC curve

            self._all_roc_hulls[g] = roc_convex_hull(curr_roc_points)

        # Find the group-wise optima that fulfill the fairness criteria
        self._groupwise_roc_points, self._global_roc_point = compute_equal_odds_optimum(
            groupwise_roc_hulls=self._all_roc_hulls,
            fairness_tolerance=self.tolerance,
            group_sizes_label_pos=group_sizes_label_pos,
            group_sizes_label_neg=group_sizes_label_neg,
            global_prevalence=self._global_prevalence,
            false_positive_cost=self.false_pos_cost,
            false_negative_cost=self.false_neg_cost,
        )

        # Construct each group-specific classifier
        all_rand_clfs = {
            g: RandomizedClassifier.construct_at_target_ROC(
                predictor=self.predictor,
                roc_curve_data=self._all_roc_data[g],
                target_roc_point=self._groupwise_roc_points[g],
                seed=self.seed,
            )
            for g in unique_groups
        }

        # Construct the global classifier (can be used for all groups)
        self._realized_classifier = EnsembleGroupwiseClassifiers(group_to_clf=all_rand_clfs)
        return self
    
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
                "There's nothing to plot: this classifier has not yet been "
                "fitted to any data. Call clf.fit(...) before clf.plot().")

        return True

    def plot(
            self,
            plot_roc_curves: bool = False,
            plot_roc_hulls: bool = True,
            plot_group_optima: bool = True,
            plot_group_triangulation: bool = True,
            plot_global_optimum: bool = True,
            plot_diagonal: bool = True,
            plot_relaxation: bool = False,
            group_name_map: dict = None,
            figure = None,
            **fig_kwargs,
        ):
        self._check_fit_status()
    
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle
        import seaborn as sns

        n_groups = len(self._all_roc_hulls)

        # Set group-wise colors and global color
        palette = sns.color_palette(n_colors=n_groups + 1)
        global_color = palette[0]
        all_group_colors = palette[1:]

        fig = figure if figure is not None else plt.figure(**fig_kwargs)

        # For each group `idx`
        for idx in range(n_groups):
            group_ls = (['--', ':', '-.'] * (1 + n_groups // 3))[idx]
            group_color = all_group_colors[idx]

            # Plot group-wise (actual) ROC curves
            if plot_roc_curves:
                roc_points = np.stack(self._all_roc_data[idx], axis=1)[:, 0:2]
                plot_polygon_edges(
                    np.vstack((roc_points, [1, 0])),
                    color=group_color,
                    ls=group_ls,
                    alpha=0.5,
                )

            # Plot group-wise ROC hulls
            if plot_roc_hulls:
                plot_polygon_edges(
                    self._all_roc_hulls[idx],
                    color=group_color,
                    ls=group_ls,
                )

            # Plot group-wise fair optimum
            group_optimum = self._groupwise_roc_points[idx]
            if plot_group_optima:
                plt.plot(
                    group_optimum[0], group_optimum[1],
                    label=group_name_map[idx] if group_name_map else f"group {idx}",
                    color=group_color,
                    marker="^", markersize=5,
                    lw=0,
                )

            # Plot triangulation of target point
            if plot_group_triangulation:
                _weights, triangulated_points = RandomizedClassifier.find_points_for_target_ROC(
                    roc_curve_data=self._all_roc_data[idx],
                    target_roc_point=group_optimum,
                )
                plt.plot(
                    triangulated_points[:, 0], triangulated_points[:, 1],
                    color=group_color,
                    marker='x', lw=0,
                )

                plt.fill(
                    triangulated_points[:, 0], triangulated_points[:, 1],
                    color=group_color,
                    alpha=0.1,
                )


        # Plot global optimum
        if plot_global_optimum:
            plt.plot(
                self._global_roc_point[0], self._global_roc_point[1],
                label="global",
                marker="*", color=global_color, alpha=0.6,
                markersize=5, lw=0,
            )

        # Plot rectangle to visualize constraint relaxation
        if plot_relaxation:

            # Get rectangle points
            min_x, max_x = np.min(self._groupwise_roc_points[:, 0]), np.max(self._groupwise_roc_points[:, 0])
            min_y, max_y = np.min(self._groupwise_roc_points[:, 1]), np.max(self._groupwise_roc_points[:, 1])

            # Draw relaxation rectangle
            rect = Rectangle(
                xy=(min_x, min_y),
                width=max_x - min_x,
                height=max_y - min_y,
                facecolor="grey",
                alpha=0.3,
                label="relaxation",
            )

            # Add the patch to the Axes
            ax = plt.gca()
            ax.add_patch(rect)

        # Plot diagonal
        if plot_diagonal:
            plt.plot(
                [0, 1], [0, 1],
                ls='--', color="grey", alpha=0.5,
                label="random clf.",
            )

        # Set axis settings
        plt.title(f"Solution to {self.tolerance}-relaxed equal odds optimum")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.legend(loc="lower right", borderaxespad=2)

    def __call__(self, X: np.ndarray, group: np.ndarray) -> int:
        return self._realized_classifier(X, group)

    def predict(self, X: np.ndarray, group: np.ndarray) -> int:
        return self(X, group)
