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
from pipeline.component.nn.backend.torch.base import Sequential as Seq
from pipeline.component.nn.backend.torch.cust import CustModel
from pipeline.component.nn.backend.torch.interactive import InteractiveLayer


class Sequential(object):
    def __init__(self):
        self.__config_type = None
        self._model = None

    def is_empty(self):
        return self._model is None

    def get_model(self):
        return self._model

    def add(self, layer):

        _IS_TF_KERAS = False
        try:
            import tensorflow as tf
            _IS_TF_KERAS = isinstance(layer, tf.Module)
        except ImportError:
            pass

        if _IS_TF_KERAS:
            # please notice that keras backend now is abandoned, hetero & homo nn support keras backend no more,
            # but pipeline keras interface is kept
            layer_type = "keras"
        else:
            layer_type = "torch"
            is_layer = hasattr(
                layer,
                "__module__") and "pipeline.component.nn.backend.torch.nn" == getattr(
                layer,
                "__module__")
            is_seq = isinstance(layer, Seq)
            is_cust_model = isinstance(layer, CustModel)
            is_interactive_layer = isinstance(layer, InteractiveLayer)
            if not (is_layer or is_cust_model or is_interactive_layer or is_seq):
                raise ValueError(
                    "Layer type {} not support yet, added layer must be a FateTorchLayer or a fate_torch "
                    "Sequential, remember to call fate_torch_hook() before using pipeline "
                    "".format(
                        type(layer)))

        self._add_layer(layer, layer_type)

    def _add_layer(self, layer, layer_type, replace=True):

        if layer_type == 'torch':
            if self._model is None or replace:
                self._model = Seq()
                self.__config_type = layer_type
        elif layer_type == 'keras':
            # please notice that keras backend now is abandoned, hetero & homo nn support keras backend no more,
            # but pipeline keras interface is kept
            from pipeline.component.nn.models.keras_interface import SequentialModel
            self.__config_type = layer_type
            self._model = SequentialModel()

        self._model.add(layer)

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

    def __repr__(self):
        return self._model.__repr__()
