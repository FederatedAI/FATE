from typing import Dict
import numpy as np
import torch
from fate.ml.evaluation.metric_base import Metric
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score


def to_numpy(data):
    if isinstance(data, list):
        return np.array(data)
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return data
    

class AUC(Metric):

    def __init__(self):
        super().__init__()

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = to_numpy(predict)
        label = to_numpy(label)
        auc_score = roc_auc_score(label, predict)
        ret = {'auc': auc_score}
        return ret


class BinaryMetricWithThreshold(Metric):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold


class BinaryAccuracy(BinaryMetricWithThreshold):

    def __init__(self, threshold=0.5):
        super().__init__(threshold)

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = to_numpy(predict)
        predict = (predict > self.threshold).astype(int)
        label = to_numpy(label)
        acc = accuracy_score(label, predict)
        ret = {'accuracy': acc}
        return ret


class BinaryRecall(BinaryMetricWithThreshold):

    def __init__(self, threshold=0.5):
        super().__init__(threshold)

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = to_numpy(predict)
        label = to_numpy(label)
        predict = (predict > self.threshold).astype(int)
        recall = recall_score(label, predict)
        ret = {'recall': recall}
        return ret


class BinaryPrecision(BinaryMetricWithThreshold):

    def __init__(self, threshold=0.5):
        super().__init__(threshold)

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = to_numpy(predict)
        label = to_numpy(label)
        predict = (predict > self.threshold).astype(int)
        precision = precision_score(label, predict)
        ret = {'precision': precision}
        return ret
