from typing import Dict
from transformers import EvalPrediction
import pandas as pd


class Metric(object):

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, predict, label, **kwargs) -> Dict:
        pass


class MetricPipeline(object):

    def __init__(self) -> None:
        self._metrics = []

    def add_metric(self, metric: Metric):
        self._metrics.append(metric)
        return self
    
    def _parse_input(self, eval_rs):
        
        if isinstance(eval_rs, EvalPrediction):
            # parse hugging face format
            predict = eval_rs.predictions
            label = eval_rs.label_ids
            input_ = eval_rs.inputs

        elif isinstance(eval_rs, tuple):
            # conventional format
            predict, label = eval_rs
            input_ = None
        
        # elif isinstance(eval_rs)
        #     # pandas dataframe
        #     # FATE DataFrame

        else:
            raise ValueError('unkown eval_rs format: {}'.format(type(eval_rs)))
        
        return predict, label, input_

    def __call__(self, eval_rs, **kwargs) -> Dict:
        metric_result = {}
        predict, label, input_ = self._parse_input(eval_rs)
        for metric in self._metrics:
            metric_result.update(metric(predict=predict, label=label, input_=input_))
        return metric_result