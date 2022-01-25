import math
import unittest

import numpy as np

from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.optim.optimizer import _SgdOptimizer


class TestInitialize(unittest.TestCase):
    def test_optimizer(self):
        model_weights = LinearModelWeights(np.array([0.10145129, 0.39987222, -0.96630206, -0.41208423, -0.24609715,
                                                     -0.70518652, 0.71478064, 0.57973894, 0.5703622, -0.45482125,
                                                     0.32676194, -0.00648212, 0.35542874, -0.26412695, -0.07964603,
                                                     1.2158522, -0.41255564, -0.01686044, -0.99897542, 1.56407211,
                                                     0.52040711, 0.24568055, 0.4880494, 0.52269909, -0.14431923,
                                                     0.03282471, 0.09437969, 0.21407206, -0.270922]), True)

        prev_model_weights = LinearModelWeights(np.array([0.10194331, 0.40062114, -0.96597859, -0.41202348, -0.24587005,
                                                          -0.7047801, 0.71515712, 0.58045583, 0.57079086, -0.45473676,
                                                          0.32775863, -0.00633238, 0.35567219, -0.26343469, -0.07964763,
                                                          1.2165642, -0.41244749, -0.01589344, -0.99862982, 1.56498698,
                                                          0.52058152, 0.24572171, 0.48809946, 0.52272993, -0.14330367,
                                                          0.03283002, 0.09439601, 0.21433497, -0.27011673]), True)

        prev_model_weights_null = None

        eps = 0.00001

        # 1: alpha = 0, no regularization
        learning_rate = 0.2
        alpha = 0
        penalty = "L2"
        decay = "0.2"
        decay_sqrt = "true"
        mu = 0.01

        init_params = [learning_rate, alpha, penalty, decay, decay_sqrt, mu]
        optimizer = _SgdOptimizer(*init_params)
        loss_norm = optimizer.loss_norm(model_weights, prev_model_weights_null)
        self.assertTrue(math.fabs(loss_norm) <= eps)  # == 0

        # 2
        alpha = 0.1
        init_params = [learning_rate, alpha, penalty, decay, decay_sqrt, mu]
        optimizer = _SgdOptimizer(*init_params)
        loss_norm = optimizer.loss_norm(model_weights, prev_model_weights_null)
        print("loss_norm = {}".format(loss_norm))
        self.assertTrue(math.fabs(loss_norm - 0.47661579875266186) <= eps)

        # 3
        loss_norm = optimizer.loss_norm(model_weights, prev_model_weights)
        print("loss_norm = {}".format(loss_norm))
        self.assertTrue(math.fabs(loss_norm - 0.47661583737200075) <= eps)


if __name__ == '__main__':
    unittest.main()
