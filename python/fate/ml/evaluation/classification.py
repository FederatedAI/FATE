from typing import Dict
import numpy as np
import torch
from fate.ml.evaluation.metric_base import Metric
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score


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
        predict = self.to_numpy(predict)
        label = self.to_numpy(label)
        auc_score = roc_auc_score(label, predict)
        ret = {'auc': auc_score}
        return ret


class BinaryMetricWithThreshold(Metric):

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold


class MultiAccuracy(Metric):

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        label = self.to_numpy(label)
        if predict.shape != label.shape:
            predict = predict.argmax(axis=-1)
        acc = accuracy_score(label, predict)
        return {'accuracy': acc}


class MultiRecall(Metric):

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        label = self.to_numpy(label)
        if predict.shape != label.shape:
            predict = predict.argmax(axis=-1)
        recall = recall_score(label, predict, average='micro')
        return {'recall': recall}


class MultiPrecision(Metric):

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        label = self.to_numpy(label)
        if predict.shape != label.shape:
            predict = predict.argmax(axis=-1)
        precision = precision_score(label, predict, average='micro')
        return {'precision': precision}



class BinaryAccuracy(MultiAccuracy, BinaryMetricWithThreshold):

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        predict = (predict > self.threshold).astype(int)
        label = self.to_numpy(label)
        acc = accuracy_score(label, predict)
        return {'accuracy': acc}


class BinaryRecall(MultiRecall, BinaryMetricWithThreshold):

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        predict = (predict > self.threshold).astype(int)
        label = self.to_numpy(label)
        recall = recall_score(label, predict)
        return {'recall': recall}


class BinaryPrecision(MultiPrecision, BinaryMetricWithThreshold):

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        predict = (predict > self.threshold).astype(int)
        label = self.to_numpy(label)
        precision = precision_score(label, predict)
        return {'precision': precision}


class MultiF1Score(Metric):

    def __init__(self, average='micro'):
        super().__init__()
        self.average = average

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        label = self.to_numpy(label)
        if predict.shape != label.shape:
            predict = predict.argmax(axis=-1)
        f1 = f1_score(label, predict, average=self.average)  
        return {'f1_score': f1}

class BinaryF1Score(MultiF1Score, BinaryMetricWithThreshold):

    def __init__(self, threshold=0.5, average='binary'):
        super().__init__(average)
        self.threshold = threshold

    def __call__(self, predict, label, **kwargs) -> Dict:
        predict = self.to_numpy(predict)
        predict = (predict > self.threshold).astype(int)
        label = self.to_numpy(label)
        f1 = f1_score(label, predict, average=self.average)
        return {'f1_score': f1}
