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
from federatedml.nn.hetero_nn.backend.pytorch.pytorch_uitl import pytorch_label_reformat
from federatedml.nn.hetero_nn.backend.tf_keras.data_generator import KerasSequenceDataConverter
from federatedml.nn.hetero_nn.protection_enhance.coae import train_an_autoencoder_confuser, CoAE, coae_label_reformat, \
    CrossEntropy


class HeteroNNTopModel(object):

    def __init__(self, input_shape, loss, optimizer, metrics, layer_config, config_type, coae_config):

        self.config_type = config_type
        self.label_reformat = None
        self.coae = None
        self.coae_config = coae_config
        self.labem_num = 2

        if self.config_type == consts.keras_backend:
            self._model: KerasNNModel = build_keras(input_shape=input_shape, loss=loss,
                                                    optimizer=optimizer, nn_define=layer_config, metrics=metrics)
            self.data_converter = KerasSequenceDataConverter()

        elif self.config_type == consts.pytorch_backend:
            self._model: PytorchNNModel = PytorchNNModel(nn_define=layer_config, optimizer_define=optimizer,
                                                         loss_fn_define=loss)
            self.data_converter = PytorchDataConvertor()
            self.label_reformat = pytorch_label_reformat
            if self.coae_config:
                self._model.loss_fn = CrossEntropy()

        if self.coae_config:
            self.label_reformat = coae_label_reformat

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

        # transform label format
        if self.label_reformat:
            y = self.label_reformat(y)

        # train an auto-encoder confuser
        if self.coae_config and self.coae is None:
            self.labem_num = y.shape[1]
            LOGGER.debug('training coae encoder')
            self.coae: CoAE = train_an_autoencoder_confuser(self.labem_num, self.coae_config.epoch,
                                                            self.coae_config.lambda1, self.coae_config.lambda2,
                                                            self.coae_config.lr, self.coae_config.verbose)
        # make fake soft label
        if self.coae:
            y = self.coae.encode(y).detach().numpy()  # transform labels to fake labels
            LOGGER.debug('fake labels are {}'.format(y))

        # run selector
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
                                                                 np.array(self.batch_data_cached_y[: self.batch_size]))[
                    0]
                self._model.train(data)
                self.batch_data_cached_X = self.batch_data_cached_X[self.batch_size:]
                self.batch_data_cached_y = self.batch_data_cached_y[self.batch_size:]

        else:
            input_gradient = self._model.get_input_gradients(x, y)[0]
            data = self.data_converter.convert_data(x, y)
            self._model.train(data)
            loss = self._model.get_loss()[0]

        return selective_id, input_gradient, loss

    def predict(self, input_data):

        output_data = self._model.predict(input_data)

        if self.coae:
            real_output = self.coae.decode(output_data).detach().numpy()
            if real_output.shape[1] == 2:
                real_output = real_output[::, 1].reshape((-1, 1))
            return real_output
        else:
            return output_data

    def evaluate(self, x, y):
        data = self.data_converter.convert_data(x, y)
        return self._model.evaluate(data)

    def export_coae(self):
        if self.coae:
            model_bytes = PytorchNNModel.get_model_bytes(self.coae)
            return model_bytes
        else:
            return None

    def restore_coae(self, model_bytes):
        if model_bytes is not None and len(model_bytes) > 0:
            coae = PytorchNNModel.recover_model_bytes(model_bytes)
            self.coae = coae

    def export_model(self):
        return self._model.export_model()

    def restore_model(self, model_bytes):
        self._model = self._model.restore_model(model_bytes)

    def recompile(self, loss, optimizer, metrics):
        self._model.compile(loss=loss,
                            optimizer=optimizer,
                            metrics=metrics)
