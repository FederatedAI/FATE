#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import unittest
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fate.ml.evaluation.regression import *


class TestMetric(unittest.TestCase):
    def test_RMSE(self):
        rmse_metric = RMSE()
        predict = np.random.random_sample(1000) * 10
        label = np.random.random_sample(1000) * 10
        result = rmse_metric(predict, label)
        self.assertEqual(result.metric_name, "rmse")
        self.assertAlmostEqual(result.result, np.sqrt(mean_squared_error(label, predict)), places=7)

    def test_MSE(self):
        mse_metric = MSE()
        predict = np.random.random_sample(1000) * 10
        label = np.random.random_sample(1000) * 10
        result = mse_metric(predict, label)
        self.assertEqual(result.metric_name, "mse")
        self.assertAlmostEqual(result.result, mean_squared_error(label, predict), places=7)

    def test_MAE(self):
        mae_metric = MAE()
        predict = np.random.random_sample(1000) * 10
        label = np.random.random_sample(1000) * 10
        result = mae_metric(predict, label)
        self.assertEqual(result.metric_name, "mae")
        self.assertAlmostEqual(result.result, mean_absolute_error(label, predict), places=7)

    def test_R2Score(self):
        r2_metric = R2Score()
        predict = np.random.random_sample(1000) * 10
        label = np.random.random_sample(1000) * 10
        result = r2_metric(predict, label)
        self.assertEqual(result.metric_name, "r2_score")
        self.assertAlmostEqual(result.result, r2_score(label, predict), places=7)

    def test_to_dict(self):
        metrics = [RMSE(), MSE(), MAE(), R2Score()]
        predict = np.random.random_sample(1000) * 10
        label = np.random.random_sample(1000) * 10
        for m in metrics:
            print(m(predict, label).to_dict())


if __name__ == "__main__":
    unittest.main()
