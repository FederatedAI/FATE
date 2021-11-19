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
from federatedml.util import LOGGER
import numpy as np


class HeteroNNBottomModel(object):
    def __init__(self, input_shape=None, optimizer="SGD", model_builder=None, layer_config=None):
        loss = "keep_predict_loss"
        self._model = model_builder(input_shape=input_shape,
                                    nn_define=layer_config,
                                    optimizer=optimizer,
                                    loss=loss,
                                    metrics=None)

        self.data_converter = None
        self.do_backward_select_strategy = False
        self.x = []
        self.x_cached = []
        self.batch_size = None

    def set_data_converter(self, data_converter):
        self.data_converter = data_converter

    def set_backward_select_strategy(self):
        self.do_backward_select_strategy = True

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def forward(self, x):
        LOGGER.debug("bottom model start to forward propagation")
        self.x = x
        data = self.data_converter.convert_data(x)
        output_data = self._model.predict(data)

        return output_data

    def backward(self, x, y, selective_ids):
        LOGGER.debug("bottom model start to backward propagation")
        if self.do_backward_select_strategy:
            if selective_ids:
                if len(self.x_cached) == 0:
                    self.x_cached = self.x[selective_ids]
                else:
                    self.x_cached = np.vstack((self.x_cached, self.x[selective_ids]))

        if len(y) == 0:
            return

        if self.do_backward_select_strategy:
            x = self.x_cached[: self.batch_size]
            self.x_cached = self.x_cached[self.batch_size:]

        data = self.data_converter.convert_data(x, y / x.shape[0])
        self._model.train(data)

    def predict(self, x):
        data = self.data_converter.convert_data(x)

        return self._model.predict(data)

    def export_model(self):
        return self._model.export_model()

    def restore_model(self, model_bytes):
        self._model = self._model.restore_model(model_bytes)

    def recompile(self, optimizer):
        self._model.compile(loss="keep_predict_loss",
                            optimizer=optimizer,
                            metrics=None)
