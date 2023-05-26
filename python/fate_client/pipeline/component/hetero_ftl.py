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
from pipeline.interface import Input
from pipeline.interface import Output
from pipeline.utils.tools import extract_explicit_parameter
from pipeline.param import consts


try:
    from pipeline.component.component_base import FateComponent
    from pipeline.component.nn.models.sequantial import Sequential
    import numpy as np
except Exception as e:
    print(e)
    print('Import NN components in HeteroFTL module failed, \
this may casue by the situation that torch/keras are not installed,\
please install them to use this module')


def find_and_convert_float32_in_dict(d, path=""):
    for k, v in d.items():
        new_path = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            find_and_convert_float32_in_dict(v, new_path)
        elif isinstance(v, np.float32) or isinstance(v, np.float64):
            d[k] = float(v)


class HeteroFTL(FateComponent):

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
        # explicit_parameters["loss"] = None
        # explicit_parameters["metrics"] = None
        explicit_parameters["nn_define"] = None
        explicit_parameters["config_type"] = "keras"
        FateComponent.__init__(self, **explicit_parameters)

        if "name" in explicit_parameters:
            del explicit_parameters["name"]
        for param_key, param_value in explicit_parameters.items():
            setattr(self, param_key, param_value)

        self.input = Input(self.name, data_type="multi")
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

        find_and_convert_float32_in_dict(self.nn_define)
        find_and_convert_float32_in_dict(self.optimizer)

    def __getstate__(self):
        state = dict(self.__dict__)
        del state["_nn_model"]

        return state
