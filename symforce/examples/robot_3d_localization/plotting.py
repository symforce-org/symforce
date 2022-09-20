# ----------------------------------------------------------------------------
# SymForce - Copyright 2022, Skydio, Inc.
# This source code is under the Apache 2.0 license found in the LICENSE file.
# ----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.widgets import Slider

from symforce.opt.optimizer import Optimizer
from symforce.python_util import AttrDict
from symforce.values import Values


def plot_solution(
    optimizer: Optimizer,
    result: Optimizer.Result,
    animated: bool = False,
    show_iteration_text: bool = False,
) -> None:
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
    ax = plt.axes(projection="3d")
    ax.set_zlim3d(0, 11)
    ax.set_ylim3d(0, 11)
    ax.set_xlim3d(0, 11)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()

    # Pull out quantities to plot
    data = get_data_to_plot(result.optimized_values)

    # Draw a circle at the origin
    ax.scatter3D([0], [0], [0], color="black", s=50, zorder=2)

    # Draw landmark locations
    ax.scatter3D(
        data.world_t_landmark[:, 0],
        data.world_t_landmark[:, 1],
        data.world_t_landmark[:, 2],
        color="orange",
        s=100,
        zorder=3,
    )

    # Draw poses
    poses_circles = [
        ax.scatter3D(
            data.world_t_body[:, 0],
            data.world_t_body[:, 1],
            data.world_t_body[:, 2],
            color="skyblue",
            zorder=3,
            s=100,
        )
    ]

    # Draw lines connecting poses
    poses_lines = ax.plot3D(
        data.world_t_body[:, 0],
        data.world_t_body[:, 1],
        data.world_t_body[:, 2],
        color="black",
        zorder=2,
        alpha=0.8,
    )

    # Draw dotted lines from poses to their landmark heading measurements
    dotted_style = dict(
        arrow_length_ratio=0.05,
        linestyle=":",
        linewidth=0.4,
        color="red",
        alpha=0.8,
        capstyle="butt",
    )
    meas_arrows = [
        ax.quiver(
            data.world_t_body[:, 0],
            data.world_t_body[:, 1],
            data.world_t_body[:, 2],
            data.meas_vectors[:, landmark_inx, 0],
            data.meas_vectors[:, landmark_inx, 1],
            data.meas_vectors[:, landmark_inx, 2],
            **dotted_style,
        )
        for landmark_inx in range(data.meas_vectors.shape[1])
    ]

    # Text box to write iteration stats
    if show_iteration_text:
        text = ax.text(8, 7, 9, "-", color="black")

    def update_plot(slider_value: np.float64) -> None:
        """
        Update the plot using the given iteration.
        """
        num = int(slider_value)

        # Set iteration text and abort if we rejected this iteration
        if show_iteration_text:
            stats = result.iteration_stats[num]
            if num > 0 and not stats.update_accepted:
                text.set_text(f"Iteration: {num} (rejected)\nError: {stats.new_error:.1f}")
                return
            text.set_text(f"Iteration: {num}\nError: {stats.new_error:.1f}")

        # Get plottable data for this iteration
        v = values_per_iter[num]
        data = get_data_to_plot(v)

        # Update the pose locations and connecting lines
        poses_circles[0].remove()
        poses_circles[0] = ax.scatter3D(
            data.world_t_body[:, 0],
            data.world_t_body[:, 1],
            data.world_t_body[:, 2],
            color="skyblue",
            zorder=3,
            s=100,
        )

        poses_lines[0].remove()
        poses_lines[0] = ax.plot3D(
            data.world_t_body[:, 0],
            data.world_t_body[:, 1],
            data.world_t_body[:, 2],
            color="black",
            zorder=2,
            alpha=0.8,
        )[0]

        # Update measurement vectors to landmarks
        for landmark_inx in range(len(v["world_t_landmark"])):
            meas_arrows[landmark_inx].remove()
            meas_arrows[landmark_inx] = ax.quiver(
                data.world_t_body[:, 0],
                data.world_t_body[:, 1],
                data.world_t_body[:, 2],
                data.meas_vectors[:, landmark_inx, 0],
                data.meas_vectors[:, landmark_inx, 1],
                data.meas_vectors[:, landmark_inx, 2],
                **dotted_style,
            )

    if animated:
        ani = animation.FuncAnimation(
            fig, update_plot, len(values_per_iter), fargs=tuple(), interval=500
        )
        filename = "robot_3d_localization.mp4"
        ani.save(filename, dpi=200)
        print(f"Wrote to {filename}")
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
    data.world_t_landmark = np.array(v["world_t_landmark"])

    # Pose positions
    data.world_t_body = np.array([p.position() for p in v["world_T_body"]])

    # Pose axis vectors
    data.pose_x_axes = np.array([p.rotation() * np.array([1, 0, 0]) for p in v["world_T_body"]])
    data.pose_y_axes = np.array([p.rotation() * np.array([0, 1, 0]) for p in v["world_T_body"]])
    data.pose_z_axes = np.array([p.rotation() * np.array([0, 0, 1]) for p in v["world_T_body"]])

    # Measurement heading vectors from each pose to each landmark
    data.meas_vectors = np.array(
        [
            [
                v["world_T_body"][i].rotation() * v["body_t_landmark_measurements"][i][j]
                for j in range(len(v["world_t_landmark"]))
            ]
            for i in range(len(v["world_T_body"]))
        ]
    )

    return data
