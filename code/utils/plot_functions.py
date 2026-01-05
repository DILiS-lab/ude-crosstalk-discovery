import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import jax
import jax.numpy as jnp
import numpy as np


def plot_data(
    time_points,
    data_1,
    experiment_folder,
    title,
    xlabel,
    ylabel,
    data_labels,
    savefig=True,
    legend=False,
    data_2=None,
    data_3=None,
    plot_random_samples=False,
    show_markers=False,
    show_title=False,
):
    """
    Plot time series data with options for multiple datasets and random sampling.
    Args:
        time_points: Array of time points for the x-axis.
        data_1: Primary dataset to plot (2D array: time_points x samples).
        experiment_folder: Folder path to save the plot.
        title: Title of the plot.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        data_labels: List of labels for the datasets.
        savefig: If True, save the figure to the experiment folder.
        legend: If True, display the legend.
        data_2: Optional secondary dataset to plot (2D array).
        data_3: Optional tertiary dataset to plot (2D array).
        plot_random_samples: If True, plot random samples from the datasets.
        show_markers: If True, show markers on the plot lines.
        show_title: If True, display the title on the plot.
    """

    fig, ax = plt.subplots(figsize=(12, 8))
    if data_1.ndim == 1:
        data_1 = data_1.reshape(-1, 1)
    if data_2 is not None and data_2.ndim == 1:
        data_2 = data_2.reshape(-1, 1)
    if data_3 is not None and data_3.ndim == 1:
        data_3 = data_3.reshape(-1, 1)

    n_gens = data_1.shape[1]

    marker_kwargs = {"marker": "o", "markersize": 4} if show_markers else {}

    if n_gens == 1:
        ax.plot(time_points, data_1[:, 0], "-", **marker_kwargs)
        if data_2 is not None:
            ax.plot(time_points, data_2[:, 0], "--", linewidth=1.5, **marker_kwargs)
        if data_3 is not None:
            ax.plot(time_points, data_3[:, 0], ":", linewidth=2, **marker_kwargs)
    else:
        for i in range(n_gens):
            (line1,) = ax.plot(time_points, data_1[:, i], "-", **marker_kwargs)
            line_color = line1.get_color()

            if data_2 is not None:
                ax.plot(
                    time_points,
                    data_2[:, i],
                    "--",
                    color=line_color,
                    linewidth=1.5,
                    **marker_kwargs,
                )
            if data_3 is not None:
                ax.plot(
                    time_points,
                    data_3[:, i],
                    ":",
                    color=line_color,
                    linewidth=2,
                    **marker_kwargs,
                )

    if show_title:
        ax.set(title=title)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.grid(True, alpha=0.3)

    if legend:
        legend_elements = [
            Line2D([0], [0], color="black", lw=2, linestyle="-", label=data_labels[0]),
        ]
        if data_2 is not None:
            legend_elements.append(
                Line2D(
                    [0], [0], color="black", lw=2, linestyle="--", label=data_labels[1]
                )
            )
        if data_3 is not None:
            legend_elements.append(
                Line2D(
                    [0], [0], color="black", lw=2, linestyle=":", label=data_labels[2]
                )
            )
        ax.legend(handles=legend_elements)

    fig.tight_layout()
    if savefig:
        plot_name = title.replace(" ", "_")
        fig.savefig(
            os.path.join(experiment_folder, f"{plot_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()

    if plot_random_samples:
        key = jax.random.PRNGKey(0)
        random_indices = jax.random.choice(key, n_gens, shape=(4,), replace=False)
        random_data_1 = data_1[:, random_indices]
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        plot_name = "Random Samples: " + title
        if show_title:
            fig.suptitle(plot_name, fontsize=16)

        colors = plt.get_cmap("tab10").colors

        for i, ax in enumerate(ax.flat):
            color = colors[i % len(colors)]
            ax.plot(time_points, random_data_1[:, i], "-", color=color, **marker_kwargs)
            if data_2 is not None:
                ax.plot(
                    time_points,
                    data_2[:, random_indices[i]],
                    "--",
                    color=color,
                    linewidth=1.5,
                    **marker_kwargs,
                )
            if data_3 is not None:
                ax.plot(
                    time_points,
                    data_3[:, random_indices[i]],
                    ":",
                    color=color,
                    linewidth=2,
                    **marker_kwargs,
                )
            ax.set(
                title=f"Sample {random_indices[i] + 1}", xlabel=xlabel, ylabel=ylabel
            )
            ax.grid(True, alpha=0.3)

        if legend:
            legend_elements = [
                Line2D(
                    [0], [0], color="black", lw=2, linestyle="-", label=data_labels[0]
                ),
            ]
            if data_2 is not None:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        lw=2,
                        linestyle="--",
                        label=data_labels[1],
                    )
                )
            if data_3 is not None:
                legend_elements.append(
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        lw=2,
                        linestyle=":",
                        label=data_labels[2],
                    )
                )

            fig.legend(
                handles=legend_elements, loc="upper right", bbox_to_anchor=(0.9, 0.9)
            )

        fig.tight_layout()
        if savefig:
            plot_name = plot_name.replace(" ", "_").replace(":", "")
            fig.savefig(
                os.path.join(experiment_folder, f"{plot_name}.png"),
                dpi=300,
                bbox_inches="tight",
            )
        plt.show()


def plot_training_loss(
    losses, experiment_folder, title, savefig=True, show_title=False
):
    """
    Plot training loss over epochs.
    Args:
        losses: List or array of loss values per epoch.
        experiment_folder: Folder path to save the plot.
        title: Title of the plot.
        savefig: If True, save the figure to the experiment folder.
        show_title: If True, display the title on the plot.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(losses) + 1), losses)
    if show_title:
        ax.set(title=title)
    ax.set(xlabel="Epochs", ylabel="Loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if savefig:
        plot_name = title.replace(" ", "_")
        fig.savefig(
            os.path.join(experiment_folder, f"{plot_name}.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def compute_and_plot_rmse_per_timepoint(
    true_values, predicted_values, time_points, experiment_folder
):
    """
    Compute and plot RMSE per time point between true and predicted values.
    Args:
        true_values: Array of true values (time_points x samples).
        predicted_values: Array of predicted values (time_points x samples).
        time_points: Array of time points corresponding to the values.
        experiment_folder: Folder path to save the RMSE plot and data.
    """

    rmse_timepoint = jnp.sqrt(jnp.mean((true_values - predicted_values) ** 2, axis=1))

    # Relative RMSE (mean-normalized)
    rrmse_timepoint = rmse_timepoint / jnp.mean(true_values, axis=1)
    np.savetxt(
        os.path.join(experiment_folder, "rmse_per_timepoint.txt"),
        rrmse_timepoint,
        fmt="%.6f",
    )

    rrmse_overall_p53 = jnp.sqrt(
        jnp.mean((true_values - predicted_values) ** 2)
    ) / jnp.mean(true_values)
    np.savetxt(
        os.path.join(experiment_folder, "rmse_overall_p53.txt"),
        np.array([rrmse_overall_p53]),
        fmt="%.6f",
    )
    print(f"Overall RMSE: {rrmse_overall_p53:.6f}")

    plot_data(
        time_points,
        rrmse_timepoint.reshape((-1, 1)),
        experiment_folder,
        "Relative RMSE per Time Point (True vs UDE Predicted)",
        "Time [h]",
        "Relative RMSE",
        ["Relative RMSE"],
        savefig=True,
        legend=False,
        show_title=False,
    )

    return rrmse_timepoint, rrmse_overall_p53
