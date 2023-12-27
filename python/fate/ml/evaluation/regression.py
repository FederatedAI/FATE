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

from typing import Dict
import numpy as np
from fate.ml.evaluation.metric_base import Metric
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fate.ml.evaluation.metric_base import EvalResult


class RMSE(Metric):
    metric_name = "rmse"

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        rmse = np.sqrt(mean_squared_error(label, predict))
        return EvalResult(self.metric_name, rmse)


class MSE(Metric):
    metric_name = "mse"

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        mse = mean_squared_error(label, predict)
        return EvalResult(self.metric_name, mse)


class MAE(Metric):
    metric_name = "mae"

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        mae = mean_absolute_error(label, predict)
        return EvalResult(self.metric_name, mae)


class R2Score(Metric):
    metric_name = "r2_score"

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        r2 = r2_score(label, predict)
        return EvalResult(self.metric_name, r2)
