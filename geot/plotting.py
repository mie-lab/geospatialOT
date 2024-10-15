import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 13})


def plot_cost_matrix(cost_matrix, title="Cost matrix", label="Cost"):
    """
    Plot the cost matrix as a heatmap

    Args:
        cost_matrix (np.array): 2D array of shape (M, N) with the pairwise costs between M and N locations
    """
    assert cost_matrix.ndim == 2, "Cost matrix must be 2D"
    m, n = cost_matrix.shape
    plt.imshow(cost_matrix)
    plt.xlabel(f"Locations 1-{m}")
    plt.ylabel(f"Locations 1-{n}")
    plt.colorbar(label=label)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


def plot_predictions_and_ground_truth(locations, predictions, observations):
    """
    Plot the predictions, observations and residuals as scatter plots

    Args:
        locations (np.array): 2D array of shape (N, 2) with the locations of the predictions and observations
        predictions (np.array): 1D array of shape (N,) with the predictions
        observations (np.array): 1D array of shape (N,) with the observations
    """
    # plot predictions, observations and residuals
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.scatter(locations[:, 0], locations[:, 1], c=observations)
    plt.colorbar(label="observations")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Observations")
    plt.subplot(1, 3, 2)
    plt.scatter(locations[:, 0], locations[:, 1], c=predictions)
    plt.colorbar(label="prediction")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Predictions")
    plt.subplot(1, 3, 3)
    plt.scatter(locations[:, 0], locations[:, 1], c=predictions - observations)
    plt.colorbar(label="residuals")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Residuals (predictions - observations)")
    plt.tight_layout()
    plt.show()


def plot_paired_transport_matrix(
    locations, predictions, observations, ot_matrix
):
    """
    Plot the transport matrix as arrows between predicted and true spatial
    distribution.

    Args:
        locations (np.array): 2D array of shape (N, 2), the locations
            (same locations for gt and preds)
        predictions (np.array): array of shape (N), the predictions
        observations (np.array): array of shape (N), the observations
        transport_matrix (np.array): 2D array of shape (N+1, N+1), the optimal
            transport matrix between the predicted and true spatial distribution
            (+1 for the waste vector)
    """
    head_width = 0.02 * np.mean(
        np.linalg.norm(
            locations - locations[np.random.permutation(len(locations))],
            axis=1,
        )
    )
    # get lists of indices for start and end stations
    start_station_id, end_station_id = np.where(ot_matrix[:-1, :-1] > 0)
    start_coords, end_coords = (
        locations[start_station_id],
        locations[end_station_id],
    )

    errors = np.abs(predictions - observations)
    residuals = predictions - observations

    def get_col(val):
        # one col for correct, one for over, one for under estimation
        if val < 0:
            return "purple"
        elif val == 0:
            return "green"
        else:
            return "orange"

    cols = [get_col(val) for val in residuals]

    fig, ax = plt.subplots(figsize=(10, 5))
    for (x1, y1), (x2, y2) in zip(start_coords, end_coords):
        plt.arrow(
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            head_width=head_width,
            length_includes_head=True,
            overhang=1,
            alpha=0.5,
        )
    plt.scatter(
        locations[:, 0], locations[:, 1], s=errors * 100 + 10, c=cols, alpha=0.5
    )
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="green",
            lw=0,
            markerfacecolor="g",
            markersize=7,
            label="Correct",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="orange",
            lw=0,
            label="Prediction > GT",
            markerfacecolor="orange",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="purple",
            lw=0,
            label="GT > Prediction",
            markerfacecolor="purple",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            lw=0,
            label="High error",
            markerfacecolor="black",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            lw=0,
            label="Low error",
            markerfacecolor="black",
            markersize=3,
        ),
        Line2D([0], [0], color="black", lw=2, label="Transported mass"),
    ]

    # Create the figure
    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_unpaired_transport_matrix(
    predicted_locations, true_locations, transport_matrix
):
    """Plot the transport matrix as arrows between predicted and true locations

    Args:
        predicted_locations (np.array): 2D array of shape (N, 2), the predicted locations
        true_locations (np.array): 2D array of shape (M, 2), the true locations
        transport_matrix (np.array): 2D array of shape (N, M), the optimal transport matrix between the predicted and true locations
    """
    plt.figure(figsize=(5, 3))
    # plot points
    plt.scatter(
        predicted_locations[:, 0],
        predicted_locations[:, 1],
        label="predicted locations",
    )
    plt.scatter(
        true_locations[:, 0], true_locations[:, 1], label="true locations"
    )
    # compute suitable head with for the errors based on the scale
    head_with = 0.02 * np.mean(
        np.linalg.norm(
            true_locations
            - true_locations[np.random.permutation(len(true_locations))],
            axis=1,
        )
    )

    # plot arrows
    for i, (x1, y1) in enumerate(predicted_locations):
        for j, (x2, y2) in enumerate(true_locations):
            if (
                (i != j)
                and (transport_matrix[i, j] > 0)
                and not (x1 == x2 and y1 == y2)
            ):
                plt.arrow(
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    head_width=head_with,
                    length_includes_head=True,
                    overhang=1,
                )
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc="upper left", framealpha=1)
    plt.tight_layout()
    plt.show()
