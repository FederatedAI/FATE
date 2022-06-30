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

import numpy as np
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.nn.backend.tf_keras.nn_model import build_keras, KerasNNModel
from federatedml.nn.hetero_nn.backend.pytorch.pytorch_nn_model import PytorchNNModel, PytorchDataConvertor
from federatedml.nn.hetero_nn.backend.tf_keras.data_generator import KerasSequenceDataConverter


class HeteroNNTopModel(object):
    def __init__(self, input_shape, loss, optimizer, metrics, layer_config, config_type):

        self.config_type = config_type

        if self.config_type == consts.keras_backend:
            self._model: KerasNNModel = build_keras(input_shape=input_shape, loss=loss,
                                                    optimizer=optimizer, nn_define=layer_config, metrics=metrics)
            self.data_converter = KerasSequenceDataConverter()

        elif self.config_type == consts.pytorch_backend:
            self._model: PytorchNNModel = PytorchNNModel(nn_define=layer_config, optimizer_define=optimizer,
                                                         loss_fn_define=loss)
            self.data_converter = PytorchDataConvertor()

        LOGGER.debug('top model is {}'.format(self._model))

        self.batch_size = None
        self.selector = None
        self.batch_data_cached_X = []
        self.batch_data_cached_y = []

    def set_data_converter(self, data_converter):
        self.data_converter = data_converter

    def set_backward_selector_strategy(self, selector):
        self.selector = selector

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def train_and_get_backward_gradient(self, x, y):
        LOGGER.debug("top model start to forward propagation")

        selective_id = []
        input_gradient = []
        if self.selector:
            losses = self._model.get_forward_loss_from_input(x, y)
            loss = sum(losses) / len(losses)
            selective_strategy = self.selector.select_batch_sample(losses)
            for idx, select in enumerate(selective_strategy):
                if select:
                    selective_id.append(idx)
                    self.batch_data_cached_X.append(x[idx])
                    self.batch_data_cached_y.append(y[idx])

            if len(self.batch_data_cached_X) >= self.batch_size:
                data = self.data_converter.convert_data(np.array(self.batch_data_cached_X[: self.batch_size]),
                                                        np.array(self.batch_data_cached_y[: self.batch_size]))
                input_gradient = self._model.get_input_gradients(np.array(self.batch_data_cached_X[: self.batch_size]),
                                                                 np.array(self.batch_data_cached_y[: self.batch_size]))[0]

                self._model.train(data)

                self.batch_data_cached_X = self.batch_data_cached_X[self.batch_size:]
                self.batch_data_cached_y = self.batch_data_cached_y[self.batch_size:]

        else:
            input_gradient = self._model.get_input_gradients(x, y)[0]
            data = self.data_converter.convert_data(x, y)
            self._model.train(data)
            loss = self._model.get_loss()[0]
            LOGGER.debug('input gradient is {}'.format(input_gradient))

        return selective_id, input_gradient, loss

    def predict(self, input_data):
        output_data = self._model.predict(input_data)

        return output_data

    def evaluate(self, x, y):
        data = self.data_converter.convert_data(x, y)
        return self._model.evaluate(data)

    def export_model(self):
        return self._model.export_model()

    def restore_model(self, model_bytes):
        self._model = self._model.restore_model(model_bytes)

    def recompile(self, loss, optimizer, metrics):
        self._model.compile(loss=loss,
                            optimizer=optimizer,
                            metrics=metrics)
