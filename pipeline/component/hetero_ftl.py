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
from federatedml.util import consts


class HeteroFTL(Component):

    @extract_explicit_parameter
    def __init__(self, epochs=1, batch_size=-1,
                 encrypt_param=None, predict_param=None, cv_param=None,
                 intersect_param={'intersect_method': consts.RSA},
                 validation_freqs=None, early_stopping_rounds=None, use_first_metric_only=None,
                 mode='plain', communication_efficient=False, n_iter_no_change=False, tol=1e-5,
                 local_round=5,
                 **kwargs):

        explicit_parameters = kwargs["explict_parameters"]
        explicit_parameters["optimizer"] = None
        explicit_parameters["loss"] = None
        explicit_parameters["metrics"] = None
        explicit_parameters["nn_define"] = None
        explicit_parameters["config_type"] = "keras"
        Component.__init__(self, **explicit_parameters)

        if "name" in explicit_parameters:
            del explicit_parameters["name"]
        for param_key, param_value in explicit_parameters.items():
            setattr(self, param_key, param_value)

        self.output = Output(self.name, data_type='single')
        self._module_name = "FTL"
        self.optimizer = None
        self.loss = None
        self.config_type = "keras"
        self.metrics = None
        self.bottom_nn_define = None
        self.top_nn_define = None
        self.interactive_layer_define = None
        self._nn_model = Sequential()
        self.nn_define = None

    def add_nn_layer(self, layer):
        self._nn_model.add(layer)

    def compile(self, optimizer,):

        self.optimizer = self._nn_model.get_optimizer_config(optimizer)
        self.config_type = self._nn_model.get_layer_type()
        self.nn_define = self._nn_model.get_network_config()

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_nn_model"]

        return state
