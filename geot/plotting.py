import numpy as np
import matplotlib.pyplot as plt


def plot_cost_matrix(cost_matrix):
    """
    Plot the cost matrix as a heatmap

    Args:
        cost_matrix (np.array): 2D array of shape (M, N) with the pairwise costs between M and N locations
    """
    plt.imshow(cost_matrix)
    plt.xlabel("Locations 1-100")
    plt.ylabel("Locations 1-100")
    plt.colorbar(label="Cost")
    plt.xticks([])
    plt.yticks([])
    plt.title("Cost matrix")
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
