#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HeteroNNTopModel(object):
    def __init__(self, input_shape=None, loss=None, optimizer="SGD", metrics=None, model_builder=None,
                 layer_config=None):

        self._model = model_builder(input_shape=input_shape,
                                    nn_define=layer_config,
                                    optimizer=optimizer,
                                    loss=loss,
                                    metrics=metrics)

        self.data_converter = None

    def set_data_converter(self, data_converter):
        self.data_converter = data_converter

    def train_and_get_backward_gradient(self, x, y):
        LOGGER.debug("top model start to forward propagation")
        input_gradients = self._model.get_input_gradients(x, y)

        data = self.data_converter.convert_data(x, y)
        self._model.train(data)

        return input_gradients[0]

    def predict(self, input_data):
        LOGGER.debug("top model start to backward propagation")
        output_data = self._model.predict(input_data)

        return output_data

    def evaluate(self, x, y):
        data = self.data_converter.convert_data(x, y)

        return self._model.evaluate(data)

    def export_model(self):
        return self._model.export_model()

    def restore_model(self, model_bytes):
        self._model = self._model.restore_model(model_bytes)
