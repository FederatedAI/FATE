from typing import Dict
from transformers import EvalPrediction
import pandas as pd
import torch
import numpy as np
from typing import Union
import json


SINGLE_VALUE = 'single_value'
TABLE_VALUE = 'table_value'


def np_torch_float_convert(val):
    if isinstance(val, (np.float16, np.float32, np.float64, torch.float16, torch.float32, torch.float64)):
        return float(val)
    else:
        raise TypeError("Value must be a numpy or PyTorch float")


class EvalResult(object):

    def __init__(self, result: Union[int, float, pd.DataFrame]):
        if isinstance(result, (int, float)):
            self.result = np_torch_float_convert(result)
            self.result_type = SINGLE_VALUE
        elif isinstance(result, pd.DataFrame):
            if len(result.shape) == 2:
                self.result = result
                self.result_type = TABLE_VALUE
            else:
                raise ValueError("DataFrame must be a 2D table.")
        else:
            raise TypeError("Invalid type for result. Expected int, float or DataFrame.")

    def __repr__(self) -> str:
        return self.result.__repr__()

    def to_dict(self):
        return {
            "result_type": self.result_type,
            "result": self.result.to_dict() if self.result_type == TABLE_VALUE else self.result
        }

    def to_json(self):
        if self.result_type == TABLE_VALUE:
            return self.result.to_json(orient='split')
        else:
            return json.dumps(self.to_dict())
        
    def get_raw_data(self):
        return self.result
    

class Metric(object):

    metric_name = None

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, predict, label, **kwargs) -> EvalResult:
        pass

    def to_numpy(self, data):
        if isinstance(data, list):
            return np.array(data)
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        else:
            return data


class MetricEnsemble(object):

    def __init__(self) -> None:
        self._metrics = []
        self._metric_suffix = set()

    def add_metric(self, metric: Metric):
        self._metrics.append(metric)
        return self        
    
    def _parse_input(self, eval_rs):
        if isinstance(eval_rs, EvalPrediction):
            # parse hugging face format
            predict = eval_rs.predictions
            label = eval_rs.label_ids
            input_ = eval_rs.inputs

        elif isinstance(eval_rs, tuple) and len(eval_rs) == 2:
            # conventional format
            predict, label = eval_rs
            input_ = None

        else:
            raise ValueError('Unknown eval_rs format: {}. Expected input formats are either '
                             'an instance of EvalPrediction or a 2-tuple (predict, label).'.format(type(eval_rs)))

        return predict, label, input_

    def __call__(self, eval_rs=None, predict=None, label=None, **kwargs) -> Dict:

        metric_result = {}
        
        if eval_rs is not None:
            predict, label, input_ = self._parse_input(eval_rs)

        for metric in self._metrics:
            metric_result[metric.metric_name] = metric(predict, label)
        return metric_result

    def fit(self, eval_rs=None, predict=None, label=None, **kwargs) -> Dict:
        return self.__call__(eval_rs, predict, label, **kwargs)


