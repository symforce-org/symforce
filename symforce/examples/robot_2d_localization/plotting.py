# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.widgets import Slider

import sym
from symforce.opt.optimizer import Optimizer
from symforce.python_util import AttrDict
from symforce.values import Values


def plot_solution(optimizer: Optimizer, result: Optimizer.Result, animated: bool = False) -> None:
    """
    Visualize the optimization problem along its iterations. If animated is True, displays a
    matplotlib animation instead of providing an interactive slider.
    """
    # Pull out values from the result
    values_per_iter = [
        optimizer.load_iteration_values(stats.values) for stats in result.iteration_stats
    ]

    # Create the layout
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.5)
    plt.tight_layout()

    # Pull out quantities to plot
    data = get_data_to_plot(result.optimized_values)

    # Draw a circle at the origin
    plt.scatter(x=[0], y=[0], color="black", s=50, zorder=2)

    # Draw landmark locations
    plt.scatter(data.landmark_xy[:, 0], data.landmark_xy[:, 1], color="orange", s=250, zorder=3)

    # Draw poses
    poses_circles = plt.scatter(
        data.pose_xy[:, 0], data.pose_xy[:, 1], color="skyblue", zorder=3, s=500
    )

    # Draw lines connecting poses
    poses_lines = plt.plot(
        data.pose_xy[:, 0], data.pose_xy[:, 1], color="black", zorder=2, alpha=0.8
    )

    # Draw X/Y axes for pose locations
    pose_vectors_x = plt.quiver(
        data.pose_xy[:, 0],
        data.pose_xy[:, 1],
        data.pose_x_axes[:, 0],
        data.pose_x_axes[:, 1],
        zorder=4,
        width=0.003,
        color="blue",
    )
    pose_vectors_y = plt.quiver(
        data.pose_xy[:, 0],
        data.pose_xy[:, 1],
        data.pose_y_axes[:, 0],
        data.pose_y_axes[:, 1],
        zorder=4,
        width=0.003,
        color="red",
    )

    # Draw dotted lines from poses to their landmark heading measurements
    heading_arrows = [
        plt.quiver(
            data.pose_xy[:, 0],
            data.pose_xy[:, 1],
            data.heading_vectors[:, landmark_inx, 0],
            data.heading_vectors[:, landmark_inx, 1],
            scale=1.0,
            zorder=2,
            width=0.003,
            linestyle=":",
            facecolor="none",
            linewidth=0.8,
            alpha=0.5,
            headwidth=0,
            headlength=0,
            capstyle="butt",
        )
        for landmark_inx in range(data.heading_vectors.shape[1])
    ]

    # Text box to write iteration stats
    text = ax.text(3.0, -2.6, "-", fontsize=10)

    def update_plot(slider_value: np.float64) -> None:
        """
        Update the plot using the given iteration.
        """
        num = int(slider_value)

        # Set iteration text and abort if we rejected this iteration
        stats = result.iteration_stats[num]
        if num > 0 and not stats.update_accepted:
            text.set_text(f"Iteration: {num} (rejected)\nError: {stats.new_error:.6f}")
            return
        text.set_text(f"Iteration: {num}\nError: {stats.new_error:.6f}")

        # Get plottable data for this iteration
        v = values_per_iter[num]
        data = get_data_to_plot(v)

        # Update the pose locations and connecting lines
        poses_circles.set_offsets(data.pose_xy)
        poses_lines[0].set_data(data.pose_xy.T)

        # Update pose axes
        pose_vectors_x.set_offsets(data.pose_xy)
        pose_vectors_y.set_offsets(data.pose_xy)
        pose_vectors_x.set_UVC(data.pose_x_axes[:, 0], data.pose_x_axes[:, 1])
        pose_vectors_y.set_UVC(data.pose_y_axes[:, 0], data.pose_y_axes[:, 1])

        # Update heading measurement vectors to landmarks
        for landmark_inx in range(len(v["landmarks"])):
            heading_arrows[landmark_inx].set_offsets(data.pose_xy)
            heading_arrows[landmark_inx].set_UVC(
                data.heading_vectors[:, landmark_inx, 0], data.heading_vectors[:, landmark_inx, 1]
            )

    if animated:
        _ = animation.FuncAnimation(
            fig, update_plot, len(values_per_iter), fargs=tuple(), interval=250
        )
        plt.show()
    else:
        # Add a slider for iterations at the bottom of the plot
        plt.subplots_adjust(bottom=0.2)
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
        iteration_slider = Slider(
            ax=ax_slider,
            label="Iteration",
            valmin=0,
            valmax=len(values_per_iter) - 1,
            valinit=len(values_per_iter) - 1,
            valfmt="%0.0f",
        )
        iteration_slider.on_changed(update_plot)
        iteration_slider.set_val(len(values_per_iter) - 1)
        plt.show()


def get_data_to_plot(v: Values) -> AttrDict:
    """
    Compute direct quantities needed for plotting.
    """
    data = AttrDict()

    # Landmark positions
    data.landmark_xy = np.array(v["landmarks"])

    # Pose positions
    data.pose_xy = np.array([p.position().squeeze() for p in v["poses"]])

    # Pose x/y axis vectors
    data.pose_x_axes = np.array([p.rotation() * np.array([1, 0]) for p in v["poses"]])
    data.pose_y_axes = np.array([p.rotation() * np.array([0, 1]) for p in v["poses"]])

    # Measurement heading vectors from each pose to each landmark
    data.heading_vectors = np.array(
        [
            [
                v["poses"][i].rotation()
                * sym.Rot2.from_tangent(np.array([v["angles"][i][landmark_inx]]))
                * np.array([50, 0])
                for landmark_inx in range(len(v["landmarks"]))
            ]
            for i in range(len(v["poses"]))
        ]
    )

    return data
