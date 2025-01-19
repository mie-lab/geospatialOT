# test with differet datatypes
# test that it is the same as balanced OT for balanced data
# test with different input types and shapes
import numpy as np
import torch
from geot.partialot import PartialOT, partial_ot_paired
from geot.cost import space_cost_matrix, spacetime_cost_matrix

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

    def test_spatiotemporal(self):
        np.random.seed(42)
        NUM_TIME, NUM_LOCS = 5, 20  # 5 time stepsm 20 locations
        # sample x and y coordinates for NUM_LOCS locations
        locations = np.random.rand(NUM_LOCS, 2)
        # sample observations at these locations
        observations = np.random.normal(size=(NUM_TIME, NUM_LOCS), loc=20, scale=3)
        # sample predictions -> add noise to observations
        predictions = np.random.normal(
            size=(NUM_TIME, NUM_LOCS), loc=observations, scale=3
        )

        # create a normal 2D cost matrix where the costs corresond to relocation time, assuming a constant walking speed
        time_matrix = space_cost_matrix(
            locations, speed_factor=5
        )  # 5km/h walking speed
        # create cost matrix with costs across space and time
        spacetime_matrix = spacetime_cost_matrix(
            time_matrix, time_steps=5, forward_cost=1, backward_cost=1
        )
        assert spacetime_matrix.shape == (NUM_TIME * NUM_LOCS, NUM_TIME * NUM_LOCS)

        ot_computer = PartialOT(spacetime_matrix, spatiotemporal=True, penalty_waste=0)
        ot_error = ot_computer(observations, predictions)
        assert np.isclose(ot_error, 6.779446780616148)

        # test same with sinkhorn
        ot_computer = PartialOT(
            spacetime_matrix,
            spatiotemporal=True,
            penalty_waste=0,
            entropy_regularized=True,
        )
        ot_error = ot_computer(observations, predictions)
        assert np.isclose(ot_error.item(), -1.30734241547381)
