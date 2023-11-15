"""Utils for plotting postprocessing frontier and postprocessing solution."""

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.figure
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
        figure: matplotlib.figure.Figure = None,
        **fig_kwargs,
    ):
    """Plots the group-specific solutions found for this predictor.

    Parameters
    ----------
    postprocessed_clf : RelaxedThresholdOptimizer
        A postprocessed classifier already fitted on some data.
    plot_roc_curves : bool, optional
        Whether to plot the global ROC curves, by default False.
    plot_roc_hulls : bool, optional
        Whether to plot the global ROC convex hulls, by default True.
    plot_group_optima : bool, optional
        Whether to plot the group-specific optima, by default True.
    plot_group_triangulation : bool, optional
        Whether to plot the triangulation of a group-specific solution, when
        such triangulation is needed to achieve a target ROC point.
    plot_global_optimum : bool, optional
        Whether to plot the global optimum ROC point, by default True.
    plot_diagonal : bool, optional
        Whether to plot the ROC diagonal with FPR=TPR, by default True.
    plot_relaxation : bool, optional
        Whether to plot the constraint relaxation bounding box, by default False.
    group_name_map : dict, optional
        A dictionary mapping each group's value to an appropriate name to show
        in the plot legend, by default None.
    figure : matplotlib.figure.Figure, optional
        A matplotlib figure to use when plotting, by default will generate a new
        figure for plotting.
    """
    postprocessed_clf._check_fit_status()

    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle
    import seaborn as sns

    n_groups = len(postprocessed_clf.groupwise_roc_hulls)

    # Set group-wise colors and global color
    palette = sns.color_palette(n_colors=n_groups + 1)
    global_color = palette[0]
    all_group_colors = palette[1:]

    if figure is None:
        figure = plt.figure(**fig_kwargs)

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
        *,
        perf_metric: str,
        disp_metric: str,
        show_data_type: str,
        constant_clf_perf: float,
        model_name: str = None,
        color: str = "black",
    ):
    """Helper to plot the given post-processing frontier results.

    Will use bootstrapped results if available, including plotting confidence
    intervals.

    Parameters
    ----------
    postproc_results_df : pd.DataFrame
        The DataFrame containing postprocessing results.
        This should be the output of a call to `compute_postprocessing_curve(.)`.
    perf_metric : str
        Which performance metric to plot (horizontal axis).
    disp_metric : str
        Which disparity metric to plot (vertical axis).
    show_data_type : str
        The type of data to show results for; usually this will be "test".
    constant_clf_perf : float
        Performance achieved by the constant classifier; this is the point of
        lowest performance and lowest disparity achievable by postprocessing.
    model_name : str, optional
        Shown in the plot legend. Name of the model to be postprocessed.
    color : str, optional
        Which color to use for plotting the postprocessing curve, by default "black".
    """

    # Get relevant column names
    perf_col = f"{perf_metric}_mean_{show_data_type}"
    disp_col = f"{disp_metric}_mean_{show_data_type}"

    # Check if bootstrap means are available
    has_bootstrap_results = perf_col in postproc_results_df.columns

    if not has_bootstrap_results:
        perf_col = f"{perf_metric}_{show_data_type}"
        disp_col = f"{disp_metric}_{show_data_type}"

    assert perf_col in postproc_results_df.columns, (
        f"Could not find the column '{perf_col}' for the perf. metric "
        f"'{perf_metric}' on data type '{show_data_type}'.")
    assert disp_col in postproc_results_df.columns, (
        f"Could not find the column '{disp_col}' for the disp. metric "
        f"'{disp_metric}' on data type '{show_data_type}'.")

    # Get envelope of postprocessing adjustment frontier
    postproc_frontier = get_envelope_of_postprocessing_frontier(
        postproc_results_df,
        perf_col=perf_col,
        disp_col=disp_col,
        constant_clf_perf=constant_clf_perf,
    )

    # Get inner and outer confidence intervals
    if has_bootstrap_results:
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
        label=(
            "post-processing" if model_name is None
            else f"post-processing of {model_name}"
        ),
        linestyle="-.",
        color=color,
    )

    # Draw confidence intervals (shaded area)
    if has_bootstrap_results:
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
