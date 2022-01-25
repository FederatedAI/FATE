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

from pipeline.component.component_base import FateComponent
from pipeline.component.nn.models.sequantial import Sequential
from pipeline.interface import Input
from pipeline.interface import Output
from pipeline.utils.tools import extract_explicit_parameter


class HomoNN(FateComponent):
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
        explicit_parameters["config_type"] = "keras"
        FateComponent.__init__(self, **explicit_parameters)

        if "name" in explicit_parameters:
            del explicit_parameters["name"]
        for param_key, param_value in explicit_parameters.items():
            setattr(self, param_key, param_value)

        self.optimizer = None
        self.loss = None
        self.metrics = None
        self.nn_define = None
        self.config_type = "keras"
        self.input = Input(self.name, data_type="multi")
        self.output = Output(self.name, data_type='single')
        self._module_name = "HomoNN"
        self._model = Sequential()

    def set_model(self, model):
        self._model = model

    def add(self, layer):
        self._model.add(layer)
        return self

    def compile(self, optimizer, loss=None, metrics=None):
        if metrics and not isinstance(metrics, list):
            raise ValueError("metrics should be a list")

        self.optimizer = self._model.get_optimizer_config(optimizer)
        self.loss = self._model.get_loss_config(loss)
        self.metrics = metrics
        self.config_type = self._model.get_layer_type()
        self.nn_define = self._model.get_network_config()
        return self

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_model"]

        return state
