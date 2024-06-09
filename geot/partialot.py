import numpy as np
import torch
import wasserstein
from scipy.spatial.distance import cdist
import ot

class PartialOT:
    def __init__(
        self,
        cost_matrix,
        normalize_c=True,
        penalty_unb="max",
        compute_exact=False,
        norm_sum_1=False,
        spatiotemporal=False,
    ):
        """
        Initialize unbalanced OT class with cost matrix
        Arguments:
            norm_sum_1: Whether to compute the Wasserstein distance between
                distributions or the actual value range
        """
        self.compute_exact = compute_exact
        self.norm_sum_1 = norm_sum_1
        if penalty_unb == "max":
            penalty_unb = np.max(cost_matrix)
        clen, cwid = cost_matrix.shape
        extended_cost_matrix = np.zeros((clen + 1, cwid + 1))
        extended_cost_matrix[:clen, :cwid] = cost_matrix
        extended_cost_matrix[clen, :] = penalty_unb
        extended_cost_matrix[:, cwid] = penalty_unb

        if normalize_c:
            extended_cost_matrix = extended_cost_matrix / np.max(
                extended_cost_matrix
            )
        if compute_exact:
            self.cost_matrix = extended_cost_matrix
            self.balancedOT = wasserstein.EMD()
        else:
            self.balancedOT = SinkhornLoss(
                extended_cost_matrix,
                blur=0.1,
                reach=0.01,
                scaling=0.1,
                mode="unbalanced",
                spatiotemporal=spatiotemporal,
            )

    def __call__(self, a, b):
        # compute mass that has to be imported or exported
        diff = (torch.sum(a, dim=-1) - torch.sum(b, dim=-1)).unsqueeze(-1)

        diff_pos = torch.relu(diff)
        diff_neg = torch.relu(diff * -1)

        extended_a = torch.cat((a, diff_neg), dim=-1)
        extended_b = torch.cat((b, diff_pos), dim=-1)

        if self.compute_exact:
            assert a.size()[0] == 1 and b.size()[0] == 1 and a.dim() == 2
            a_np = extended_a.squeeze().detach().numpy().astype(float)
            b_np = extended_b.squeeze().detach().numpy().astype(float)
            if self.norm_sum_1:
                a_np = a_np / np.sum(a_np)
                b_np = b_np / np.sum(b_np)
            else:
                # still need this code to avoid numeric errors
                a_np = a_np / np.sum(a_np) * np.sum(b_np)
            cost = self.balancedOT(a_np, b_np, self.cost_matrix)
            return cost
        else:
            return self.balancedOT(extended_a, extended_b)




def partial_ot_fixed_locations():
    pass

def partial_ot_relocation(locations_pred: np.ndarray, locations_gt: np.ndarray, cost_matrix=None, import_location=np.array([0,0]), import_cost_phi=0, return_matrix=False):
    # extend the shorter one with import / export coords
    len_diff = len(locations_pred) - len(locations_gt)
    fill_vector = np.expand_dims(import_location, 0).repeat(len_diff)
    if len_diff < 0:
        locations_pred_ext = np.concatenate([locations_pred, fill_vector])
        locations_gt_ext = locations_gt
    else:
        locations_gt_ext = np.concatenate([locations_gt, fill_vector])
        locations_pred_ext = locations_pred

    # at each location is a mass of 1
    new_len = len(locations_gt_ext)
    weights_pred = np.ones(new_len) / new_len
    weights_gt = np.ones(new_len) / new_len

    # compute cost matrix
    if cost_matrix is None:
        cost_matrix = cdist(locations_pred_ext, locations_gt_ext)
        if len_diff < 0:
            cost_matrix[-len_diff:, :] = import_cost_phi
        else: 
            cost_matrix[:, -len_diff:] = import_cost_phi

    transport_matrix = ot.emd(weights_pred, weights_gt, cost_matrix)
    ot_distance = np.sum(transport_matrix * cost_matrix)

    if return_matrix:
        return transport_matrix

    else:
        return ot_distance