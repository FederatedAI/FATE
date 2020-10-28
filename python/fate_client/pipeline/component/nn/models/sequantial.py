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
from tensorflow.python.keras.engine import base_layer


class Sequential(object):
    def __init__(self):
        self.__config_type = None
        self._model = None

    def is_empty(self):
        return self._model is None

    def add(self, layer):
        if isinstance(layer, base_layer.Layer):
            layer_type = "keras"
        elif isinstance(layer, dict):
            layer_type = "nn"
        elif hasattr(layer, "__module__") and getattr(layer, "__module__").startswith("torch.nn.modules"):
            layer_type = "pytorch"
        else:
            raise ValueError("Layer type {} not support yet".format(type(layer)))

        self._add_layer(layer, layer_type)

    def _add_layer(self, layer, layer_type):
        if self._model is None:
            self._model = _build_model(layer_type)
            self.__config_type = layer_type

        if self.__config_type == layer_type:
            self._model.add(layer)
            self.__config_type = layer_type
        else:
            raise ValueError(
                "pre add layer type is {}, not equals to current layer {}".format(self.__config_type, layer_type))

    def get_layer_type(self):
        return self.__config_type

    def get_loss_config(self, loss):
        return self._model.get_loss_config(loss)

    def get_optimizer_config(self, optimizer):
        return self._model.get_optimizer_config(optimizer)

    def get_network_config(self):
        if not self.__config_type:
            raise ValueError("Empty layer find, can't get config")

        return self._model.get_network_config()


def _build_model(type):
    if type == "keras":
        from pipeline.component.nn.backend.keras import model_builder
        return model_builder.build_model()

    if type == "pytorch":
        from pipeline.component.nn.backend.pytorch import model_builder
        return model_builder.build_model()

    if type == "nn":
        from pipeline.component.nn.backend.tf import model_builder
        return model_builder.build_model()
