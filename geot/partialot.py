import numpy as np
import torch
from scipy.spatial.distance import cdist
import ot
from geot.sinkhorn_loss import SinkhornLoss


class PartialOT:
    def __init__(
        self,
        cost_matrix: np.ndarray,
        penalty_waste="max",
        normalize_cost: bool = False,
        entropy_regularized: bool = False,
        spatiotemporal: bool = False,
    ):
        """
        Initialize unbalanced OT class with cost matrix
        Arguments:
            cost_matrix (np.ndarray): 2-dim numpy array with pairwise costs
                between locations
            penalty_waste (float or "max"): How much to penalize "waste vector",
                i.e. mass export and import. Either y_pred float value, or "max"
                corresponding to the maximum cost in cost_matrix
            normalize_cost (bool): Whether to normalize cost matrix by dividing
                by the maximum cost.
            entropy_regularized (bool): Set to True for using Sinkhorn loss,i.e.
                entropy-regularized OT. By default using Wasserstein distance.
            spatiotemporal (bool): Set to True to compute the error for spatio
                temporal data (across space and time)
        """
        self.entropy_regularized = entropy_regularized
        self.spatiotemporal = spatiotemporal
        if penalty_waste == "max":
            penalty_waste = np.max(cost_matrix)

        # extend cost matrix to account for mass import / export
        clen, cwid = cost_matrix.shape
        extended_cost_matrix = np.zeros((clen + 1, cwid + 1))
        extended_cost_matrix[:clen, :cwid] = cost_matrix
        extended_cost_matrix[clen, :] = penalty_waste
        extended_cost_matrix[:, cwid] = penalty_waste

        if normalize_cost:
            extended_cost_matrix = extended_cost_matrix / np.max(extended_cost_matrix)
        if entropy_regularized:
            self.sinkhorn_object = SinkhornLoss(
                extended_cost_matrix,
                blur=0.1,
                reach=0.01,
                scaling=0.1,
                mode="unbalanced",
                spatiotemporal=spatiotemporal,
            )
        else:
            self.cost_matrix = extended_cost_matrix

    def to_tensor(self, array):
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        if array.dim() == 1 or (self.spatiotemporal and (array.dim() == 2)):
            array = array.unsqueeze(0)
        return array

    def __call__(self, y_pred, y_true, return_matrix=False):
        """Compute OT error between y_pred and y_true

        Args:
            y_pred: array or tensor with predictions. Shape (batch_size, N)
            y_true: array or tensor with predictions. Shape (batch_size, N)
            return_matrix (bool, optional): Whether to output the OT matrix.
                Defaults to False.

        Returns:
            float: Optimal transport distance between the two distributions
        """
        if return_matrix:
            assert not self.entropy_regularized, "Cannot return matrix for Sinkhorn"
        y_pred, y_true = self.to_tensor(y_pred), self.to_tensor(y_true)

        if self.spatiotemporal:
            batch_size = y_pred.size()[0]
            # flatten space-time axes
            y_pred = y_pred.reshape((batch_size, -1))
            y_true = y_true.reshape((batch_size, -1))
        else:
            assert (
                y_pred.dim() == 2 and y_true.dim() == 2
            ), f"a and b must be two-dimensional (or 3-dim if spatiotemporal)\
            , here dims are {y_pred.dim()} and {y_true.dim()}"

        # convert to arrays
        # compute mass that has to be imported or exported
        diff = (torch.sum(y_pred, dim=-1) - torch.sum(y_true, dim=-1)).unsqueeze(-1)
        diff_pos = torch.relu(diff)
        diff_neg = torch.relu(diff * -1)

        # extend
        extended_pred = torch.cat((y_pred, diff_neg), dim=-1)
        extended_true = torch.cat((y_true, diff_pos), dim=-1)

        if self.entropy_regularized:
            return self.sinkhorn_object(extended_pred, extended_true)

        assert torch.all(y_pred >= 0) and torch.all(
            y_true >= 0
        ), "y_pred or y_true cannot be negative"
        # compute exact OT error with Wasserstein package
        assert (
            y_pred.size()[0] == 1 and y_true.size()[0] == 1
        ), "batching not possible in exact computation"
        extended_pred_np = extended_pred.squeeze().detach().numpy().astype(float)
        extended_true_np = extended_true.squeeze().detach().numpy().astype(float)
        # Note: extended_pred_np and extended_true_np already have the same sum
        # We still need this normalization to avoid numeric errors
        extended_pred_np = (
            extended_pred_np / np.sum(extended_pred_np) * np.sum(extended_true_np)
        )
        if return_matrix:
            transport_matrix = ot.emd(
                extended_pred_np, extended_true_np, self.cost_matrix
            )
            return transport_matrix

        cost = ot.emd2(extended_pred_np, extended_true_np, self.cost_matrix)
        return cost


def partial_ot_paired(
    cost_matrix: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    return_matrix: bool = False,
    **kwargs_partialot,
):
    """Compute OT error between y_pred and y_true

    Args:
        cost_matrix: 2-dim numpy array with pairwise costs between locations
        y_pred: array or tensor with predictions. Shape (batch_size, N)
        y_true: array or tensor with predictions. Shape (batch_size, N)
        return_matrix (bool, optional): Whether to output the OT matrix.
            Defaults to False.

    Returns:
        float: Optimal transport distance between the two distributions
    """
    pot_obj = PartialOT(cost_matrix, **kwargs_partialot)
    return pot_obj(
        torch.from_numpy(y_pred),
        torch.from_numpy(y_true),
        return_matrix=return_matrix,
    )


def partial_ot_unpaired(
    locations_pred: np.ndarray,
    locations_gt: np.ndarray,
    cost_matrix=None,
    import_location=np.array([0, 0]),
    penalty_waste=0,
    return_matrix=False,
):
    """_summary_

    Args:
        locations_pred (np.ndarray): Predicted locations (array of shape (N, d))
        locations_gt (np.ndarray): Ground true locations (array of shape (M, d))
        cost_matrix (np.ndarray, optional): Pairwise costs between locations.
            If None, Euclidean distances are used. Defaults to None.
        import_location (np.ndarray, optional): Location for importing and
            exporting mass. Can be set by appliction, e.g. bike sharing
            distribution center Defaults to np.array([0, 0]).
        penalty_waste (float or "max"): How much to penalize "waste vector",
                i.e. mass export and import. Either y_pred float value, or "max"
                corresponding to the maximum cost in cost_matrix. Defaults to 0.
        return_matrix (bool, optional): Whether to output the OT matrix.
            Defaults to False.

    Returns:
        float: Optimal transport distance between the two distributions
    """
    # extend the shorter one with import / export coords
    len_diff = len(locations_pred) - len(locations_gt)
    fill_vector = np.expand_dims(import_location, 0).repeat(len_diff, axis=0)
    if len_diff < 0:
        locations_pred_ext = np.concatenate([locations_pred, fill_vector])
        locations_gt_ext = locations_gt
    else:
        locations_gt_ext = np.concatenate([locations_gt, fill_vector])
        locations_pred_ext = locations_pred

    # at each location is y_pred mass of 1
    new_len = len(locations_gt_ext)
    weights_pred = np.ones(new_len) / new_len
    weights_gt = np.ones(new_len) / new_len

    # compute cost matrix
    if cost_matrix is None:
        cost_matrix = cdist(locations_pred_ext, locations_gt_ext)
        if len_diff < 0:
            cost_matrix[-len_diff:, :] = penalty_waste
        else:
            cost_matrix[:, -len_diff:] = penalty_waste

    transport_matrix = ot.emd(weights_pred, weights_gt, cost_matrix)
    ot_distance = np.sum(transport_matrix * cost_matrix)

    if return_matrix:
        return transport_matrix

    else:
        return ot_distance
