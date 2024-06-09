# test with differet datatypes
# test that it is the same as balanced OT for balanced data
# test with different input types and shapes
import numpy as np
import torch
from geot.partialot import PartialOT

if __name__ == "__main__":
    test_cdist = np.array(
        [
            [0.0, 0.9166617229649182, 0.8011636143804466, 1.0],
            [0.9166617229649182, 0.0, 0.2901671214052399, 0.5131642591866252],
            [0.8011636143804466, 0.2901671214052399, 0.0, 0.28166962442054133],
            [1.0, 0.5131642591866252, 0.28166962442054133, 0.0],
        ]
    )
    ot_obj = PartialOT(test_cdist, compute_exact=False)
    print(
        ot_obj(
            # torch.tensor([[1, 3, 2, 4], [1, 3, 2, 4]]),
            # torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]]),
            torch.tensor([[1, 2, 3, 4]]),
            torch.tensor([[1, 2, 3, 4]]),
        )
    )
