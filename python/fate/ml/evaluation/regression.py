from typing import Dict
import numpy as np
from fate.ml.evaluation.metric_base import Metric
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fate.ml.evaluation.metric_base import EvalResult


class RMSE(Metric):

    metric_name = 'rmse'

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        rmse = np.sqrt(mean_squared_error(label, predict))
        return EvalResult(self.metric_name, rmse)


class MSE(Metric):

    metric_name = 'mse'

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        mse = mean_squared_error(label, predict)
        return EvalResult(self.metric_name, mse)


class MAE(Metric):

    metric_name = 'mae'

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        mae = mean_absolute_error(label, predict)
        return EvalResult(self.metric_name, mae)


class R2Score(Metric):

    metric_name = 'r2_score'

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_np_format(predict)
        label = self.to_np_format(label)
        r2 = r2_score(label, predict)
        return EvalResult(self.metric_name, r2)
