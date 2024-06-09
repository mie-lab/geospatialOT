from geot.sinkhorn_loss import SinkhornLoss, sinkhorn_loss_from_numpy
import numpy as np


class TestSinkhornLoss:
    def test_sinkhorn_loss(self):
        test_cdist = np.array(
            [
                [0.0, 0.9166617229649182, 0.8011636143804466, 1.0],
                [
                    0.9166617229649182,
                    0.0,
                    0.2901671214052399,
                    0.5131642591866252,
                ],
                [
                    0.8011636143804466,
                    0.2901671214052399,
                    0.0,
                    0.28166962442054133,
                ],
                [1.0, 0.5131642591866252, 0.28166962442054133, 0.0],
            ]
        )
        print(
            sinkhorn_loss_from_numpy(
                np.array([[1, 3, 2, 4], [1, 3, 2, 4]]),
                np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
                test_cdist,
                loss_class=SinkhornLoss,
            )
        )
