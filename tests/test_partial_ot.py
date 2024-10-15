# test with differet datatypes
# test that it is the same as balanced OT for balanced data
# test with different input types and shapes
import numpy as np
import torch
from geot.partialot import PartialOT

test_cdist = np.array(
    [
        [0.0, 0.9166617229649182, 0.8011636143804466, 1.0],
        [0.9166617229649182, 0.0, 0.2901671214052399, 0.5131642591866252],
        [0.8011636143804466, 0.2901671214052399, 0.0, 0.28166962442054133],
        [1.0, 0.5131642591866252, 0.28166962442054133, 0.0],
    ]
)


class TestPartialOT:
    """Test main class for partial OR"""

    def test_zero_for_same_mass(self):
        """Test OT error being zero for same distributions"""
        ot_obj = PartialOT(test_cdist, entropy_regularized=False)
        ot_error = ot_obj(
            # torch.tensor([[1, 3, 2, 4], [1, 3, 2, 4]]),
            # torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[1, 2, 3, 4]]),
        )
        assert ot_error == 0

    def test_value_correct(self):
        """Test OT error being correct for hard-coded example"""
        ot_obj = PartialOT(test_cdist, entropy_regularized=False)
        ot_error = ot_obj(
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[1, 3, 2, 4]]),
        )
        print(ot_error)
        assert np.isclose(ot_error, 0.29016712)
