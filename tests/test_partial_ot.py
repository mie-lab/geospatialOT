# test with differet datatypes
# test that it is the same as balanced OT for balanced data
# test with different input types and shapes
import numpy as np
import torch
from geot.partialot import PartialOT, partial_ot_paired

test_cdist = np.array(
    [
        [0.0, 0.9166617229649182, 0.8011636143804466, 1.0],
        [0.9166617229649182, 0.0, 0.2901671214052399, 0.5131642591866252],
        [0.8011636143804466, 0.2901671214052399, 0.0, 0.28166962442054133],
        [1.0, 0.5131642591866252, 0.28166962442054133, 0.0],
    ]
)
test_pred, test_gt = (np.array([[1, 2, 3, 4]]), np.array([[1, 3, 2, 4]]))


class TestPartialOT:
    """Test main class for partial OR"""

    def test_zero_for_same_mass(self):
        """Test OT error being zero for same distributions"""
        ot_obj = PartialOT(test_cdist, entropy_regularized=False)
        ot_error = ot_obj(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[1, 2, 3, 4]]),
        )
        assert ot_error == 0

    def test_value_correct(self):
        """Test OT error being correct for hard-coded example"""
        ot_obj = PartialOT(
            test_cdist,
            entropy_regularized=False,
            penalty_waste=0,
        )
        ot_error = ot_obj(test_pred, test_gt)
        # compute with function
        function_computation = partial_ot_paired(
            test_cdist,
            test_pred,
            test_gt,
            penalty_waste=0,
        )
        # compute via matrix
        ot_matrix = partial_ot_paired(
            test_cdist,
            test_pred,
            test_gt,
            penalty_waste=0,
            return_matrix=True,
        )
        assert np.isclose(ot_error, 0.29016712)
        assert np.isclose(ot_error, np.sum(ot_matrix[:-1, :-1] * test_cdist))
        assert np.isclose(ot_error, function_computation)
