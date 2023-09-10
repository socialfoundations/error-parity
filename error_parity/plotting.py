"""Utils for plotting postprocessing frontier and postprocessing solution."""

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from .pareto_curve import compute_inner_and_outer_adjustment_ci, get_envelope_of_postprocessing_frontier
from .threshold_optimizer import RelaxedThresholdOptimizer
from .classifiers import RandomizedClassifier


def plot_polygon_edges(polygon_points, **kwargs):
    point_to_plot = np.vstack((polygon_points, polygon_points[0]))
    plt.plot(point_to_plot[:, 0], point_to_plot[:, 1], **kwargs)


def plot_postprocessing_solution(
    *,
    postprocessed_clf: RelaxedThresholdOptimizer,
    plot_roc_curves: bool = False,
    plot_roc_hulls: bool = True,
    plot_group_optima: bool = True,
    plot_group_triangulation: bool = True,
    plot_global_optimum: bool = True,
    plot_diagonal: bool = True,
    plot_relaxation: bool = False,
    group_name_map: dict = None,
    figure=None,
    **fig_kwargs,
):
    """Plots the group-specific solutions found by this predictor."""
    postprocessed_clf._check_fit_status()

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    import seaborn as sns

    n_groups = len(postprocessed_clf.groupwise_roc_hulls)

    # Set group-wise colors and global color
    palette = sns.color_palette(n_colors=n_groups + 1)
    global_color = palette[0]
    all_group_colors = palette[1:]

    fig = figure if figure is not None else plt.figure(**fig_kwargs)

    # For each group `idx`
    for idx in range(n_groups):
        group_ls = (["--", ":", "-."] * (1 + n_groups // 3))[idx]
        group_color = all_group_colors[idx]

        # Plot group-wise (actual) ROC curves
        if plot_roc_curves:
            roc_points = np.stack(postprocessed_clf.groupwise_roc_data[idx], axis=1)[:, 0:2]
            plot_polygon_edges(
                np.vstack((roc_points, [1, 0])),
                color=group_color,
                ls=group_ls,
                alpha=0.5,
            )

        # Plot group-wise ROC hulls
        if plot_roc_hulls:
            plot_polygon_edges(
                postprocessed_clf.groupwise_roc_hulls[idx],
                color=group_color,
                ls=group_ls,
            )

        # Plot group-wise fair optimum
        group_optimum = postprocessed_clf.groupwise_roc_points[idx]
        if plot_group_optima:
            plt.plot(
                group_optimum[0],
                group_optimum[1],
                label=group_name_map[idx] if group_name_map else f"group {idx}",
                color=group_color,
                marker="^",
                markersize=5,
                lw=0,
            )

        # Plot triangulation of target point
        if plot_group_triangulation:
            (
                _weights,
                triangulated_points,
            ) = RandomizedClassifier.find_points_for_target_ROC(
                roc_curve_data=postprocessed_clf._groupwise_roc_data[idx],
                target_roc_point=group_optimum,
            )
            plt.plot(
                triangulated_points[:, 0],
                triangulated_points[:, 1],
                color=group_color,
                marker="x",
                lw=0,
            )

            plt.fill(
                triangulated_points[:, 0],
                triangulated_points[:, 1],
                color=group_color,
                alpha=0.1,
            )

    # Plot global optimum
    if plot_global_optimum:
        plt.plot(
            postprocessed_clf.global_roc_point[0],
            postprocessed_clf.global_roc_point[1],
            label="global",
            marker="*",
            color=global_color,
            alpha=0.6,
            markersize=5,
            lw=0,
        )

    # TODO: plot maximum constraint relaxation
    # (may differ from realized relaxation, for example, if either the TPR or
    # FPR diff violation is strictly smaller than the allowed tolerance)

    # Plot rectangle to visualize constraint relaxation
    if plot_relaxation:
        # Get rectangle points
        min_x, max_x = (
            np.min(postprocessed_clf.groupwise_roc_points[:, 0]),
            np.max(postprocessed_clf.groupwise_roc_points[:, 0]),
        )
        min_y, max_y = (
            np.min(postprocessed_clf.groupwise_roc_points[:, 1]),
            np.max(postprocessed_clf.groupwise_roc_points[:, 1])
        )

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
            [0, 1],
            [0, 1],
            ls="--",
            color="grey",
            alpha=0.5,
            label="random clf.",
        )

    # Set axis settings
    plt.suptitle(f"Solution to {postprocessed_clf.tolerance}-relaxed optimum", y=0.96)
    plt.title(
        f"(fairness constraint: {postprocessed_clf.constraint.replace('_', ' ')})",
        fontsize="small",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right", borderaxespad=2)


def plot_postprocessing_frontier(
        postproc_results_df: pd.DataFrame,
        perf_metric: str,
        disp_metric: str,
        show_data_type: str,
        model_name: str,
        constant_clf_perf: float,
        color: str = "black",
    ):
    """Helper to plot the given post-processing frontier results with confidence intervals."""
    # Get envelope of postprocessing adjustment frontier
    postproc_frontier = get_envelope_of_postprocessing_frontier(
        postproc_results_df,
        perf_col=f"{perf_metric}_mean_{show_data_type}",
        disp_col=f"{disp_metric}_mean_{show_data_type}",
        constant_clf_perf=constant_clf_perf,
    )

    # Get inner and outer confidence intervals
    postproc_frontier_xticks, interior_frontier_yticks, outer_frontier_yticks = \
        compute_inner_and_outer_adjustment_ci(
            postproc_results_df,
            perf_metric=perf_metric,
            disp_metric=disp_metric,
            data_type=show_data_type,
            constant_clf_perf=constant_clf_perf,
        )

    # Draw upper right portion of the line (dominated but not feasible)
    upper_right_frontier = np.array([
        postproc_frontier[-1],
        (postproc_frontier[-1, 0] - 1e-6, 1.0),
    ])

    sns.lineplot(
        x=upper_right_frontier[:, 0],
        y=upper_right_frontier[:, 1],
        linestyle=":",
        # label=r"dominated",
        color="grey",
    )

    # Plot postprocessing frontier
    sns.lineplot(
        x=postproc_frontier[:, 0],
        y=postproc_frontier[:, 1],
        label=f"post-processing of {model_name}",
        linestyle="-.",
        color=color,
    )

    # Draw confidence intervals (shaded area)
    ax = plt.gca()
    ax.fill_between(
        x=postproc_frontier_xticks,
        y1=interior_frontier_yticks,
        y2=outer_frontier_yticks,
        interpolate=True,
        color=color,
        alpha=0.1,
        label=r"$95\%$ conf. interv.",
    )
