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
from pipeline.component.nn.models.sequantial import Sequential
from pipeline.interface.output import Output
from pipeline.utils.tools import extract_explicit_parameter


class HeteroNN(Component):

    @extract_explicit_parameter
    def __init__(self, task_type="classification", epochs=None, batch_size=-1, early_stop="diff",
                 tol=1e-5, encrypt_param=None, predict_param=None, cv_param=None,
                 validation_freqs=None, early_stopping_rounds=None, use_first_metric_only=None, **kwargs):

        explicit_parameters = kwargs["explict_parameters"]
        explicit_parameters["optimizer"] = None
        explicit_parameters["loss"] = None
        explicit_parameters["metrics"] = None
        explicit_parameters["bottom_nn_define"] = None
        explicit_parameters["top_nn_define"] = None
        explicit_parameters["interactive_layer_define"] = None
        explicit_parameters["config_type"] = "keras"
        Component.__init__(self, **explicit_parameters)

        if "name" in explicit_parameters:
            del explicit_parameters["name"]
        for param_key, param_value in explicit_parameters.items():
            setattr(self, param_key, param_value)

        self.output = Output(self.name, data_type='single')
        self._module_name = "HeteroNN"
        self.optimizer = None
        self.loss = None
        self.config_type = "keras"
        self.metrics = None
        self.bottom_nn_define = None
        self.top_nn_define = None
        self.interactive_layer_define = None
        self._bottom_nn_model = Sequential()
        self._interactive_layer = Sequential()
        self._top_nn_model = Sequential()

    def add_bottom_model(self, layer):
        self._bottom_nn_model.add(layer)

    def set_interactve_layer(self, layer):
        self._interactive_layer.add(layer)

    def add_top_model(self, layer):
        self._top_nn_model.add(layer)

    def compile(self, optimizer, loss=None, metrics=None):
        if metrics and not isinstance(metrics, list):
            raise ValueError("metrics should be a list")

        self.optimizer = self._bottom_nn_model.get_optimizer_config(optimizer)
        self.loss = self._bottom_nn_model.get_loss_config(loss)
        self.metrics = metrics
        self.config_type = self._bottom_nn_model.get_layer_type()
        self.bottom_nn_define = self._bottom_nn_model.get_network_config()
        self.top_nn_define = self._top_nn_model.get_network_config()
        self.interactive_layer_define = self._interactive_layer.get_network_config()

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_bottom_nn_model"]
        del state["_interactive_layer"]
        del state["_top_nn_model"]

        return state

