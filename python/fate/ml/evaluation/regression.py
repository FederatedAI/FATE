from typing import Dict
import numpy as np
import torch
from fate.ml.evaluation.metric_base import Metric
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error


class RMSE(Metric):

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        label = self.to_numpy(label)
        rmse = np.sqrt(mean_squared_error(label, predict))
        return {'rmse': rmse}


class MSE(Metric):

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        label = self.to_numpy(label)
        mse = mean_squared_error(label, predict)
        return {'mse': mse}


class MAE(Metric):

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        label = self.to_numpy(label)
        mae = mean_absolute_error(label, predict)
        return {'mae': mae}


class R2Score(Metric):

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        label = self.to_numpy(label)
        r2 = r2_score(label, predict)
        return {'r2_score': r2}
