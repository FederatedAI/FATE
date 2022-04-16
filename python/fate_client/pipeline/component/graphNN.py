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
#

from pipeline.component.component_base import Component
from pipeline.interface import Input
from pipeline.interface import Output
from pipeline.utils.tools import extract_explicit_parameter
from federatedml.util import LOGGER
from torch import Tensor, set_flush_denormal
from torch.nn import *
from federatedml.nn.backend.pytorch.layer import GCNLayer
import json


class GraphNN(Component):
    @extract_explicit_parameter
    def __init__(self, name=None, max_iter=100, batch_size=-1,
                 secure_aggregate=True, aggregate_every_n_epoch=1,
                 early_stop="diff", encode_label=False,
                 predict_param=None, cv_param=None, **kwargs):

        explicit_parameters = kwargs["explict_parameters"]
        explicit_parameters["optimizer"] = None
        explicit_parameters["loss"] = None
        explicit_parameters["metrics"] = None
        explicit_parameters["nn_define"] = None
        explicit_parameters["config_type"] = "pytorch"
        Component.__init__(self, **explicit_parameters)

        if "name" in explicit_parameters:
            del explicit_parameters["name"]
        for param_key, param_value in explicit_parameters.items():
            setattr(self, param_key, param_value)

        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.nn_define = None
        self.config_type = "pytorch"
        self.input = Input(self.name, data_type="multi")
        self.output = Output(self.name, data_type='single')
        self._module_name = "GraphNN"
        self._model = ModuleList()

    def set_model(self, model):
        self._model = model

    def add(self, layer):
        self._model.append(layer)
        return self

    def _get_nn_define(self):
        res = []
        for layer in self._model.modules():
            if isinstance(layer, Linear):
                res.append({
                    'type': 'Linear',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features,
                    'bias': True if isinstance(layer.bias, Tensor) else False,
                })
            elif isinstance(layer, ReLU):
                res.append({
                    'type': 'ReLU',
                })
            elif isinstance(layer, Sigmoid):
                res.append({
                    'type': 'Sigmoid',
                })
            elif isinstance(layer, ModuleList):
                pass
            elif isinstance(layer, GCNLayer):
                res.append({
                    'type': 'GCNLayer',
                    'in_features': layer.in_features,
                    'out_features': layer.out_features,
                    'bias': True if layer.bias is not None else False
                })
            elif isinstance(layer, LogSoftmax):
                res.append({
                    'type': 'LogSoftmax',
                })
            else:
                raise NotImplementedError

        return json.dumps(res)

    def compile(self, optimizer, loss=None, metrics=None):
        if metrics and not isinstance(metrics, list):
            raise ValueError("metrics should be a list")

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.nn_define = self._get_nn_define()
        print(self.nn_define)
        return self

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_model"]

        return state
