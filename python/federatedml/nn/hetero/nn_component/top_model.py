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
import torch

from federatedml.nn.hetero.nn_component.torch_model import TorchNNModel
from federatedml.nn.hetero.protection_enhance.coae import train_an_autoencoder_confuser, CoAE, coae_label_reformat, \
    CrossEntropy
from federatedml.util import LOGGER


class TopModel(object):

    def __init__(self, loss, optimizer, layer_config, coae_config, label_num):

        self.coae = None
        self.coae_config = coae_config
        self.label_num = label_num
        LOGGER.debug('label num is {}'.format(self.label_num))
        self._model: TorchNNModel = TorchNNModel(nn_define=layer_config, optimizer_define=optimizer,
                                                 loss_fn_define=loss)
        self.label_reformat = None
        if self.coae_config:
            self._model.loss_fn = CrossEntropy()

        if self.coae_config:
            self.label_reformat = coae_label_reformat

        self.batch_size = None
        self.selector = None
        self.batch_data_cached_X = []
        self.batch_data_cached_y = []

    def set_backward_selector_strategy(self, selector):
        self.selector = selector

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def train_mode(self, mode):
        self._model.train_mode(mode)

    def train_and_get_backward_gradient(self, x, y):

        LOGGER.debug("top model start to forward propagation")
        selective_id = []
        input_gradient = []

        # transform label format
        if self.label_reformat:
            y = self.label_reformat(y, label_num=self.label_num)

        # train an auto-encoder confuser
        if self.coae_config and self.coae is None:
            LOGGER.debug('training coae encoder')
            self.coae: CoAE = train_an_autoencoder_confuser(y.shape[1], self.coae_config.epoch,
                                                            self.coae_config.lambda1, self.coae_config.lambda2,
                                                            self.coae_config.lr, self.coae_config.verbose)
        # make fake soft label
        if self.coae:
            # transform labels to fake labels
            y = self.coae.encode(y).detach().numpy()
            LOGGER.debug('fake labels are {}'.format(y))

        # run selector
        if self.selector:

            # when run selective bp, need to convert y to numpy format
            if isinstance(y, torch.Tensor):
                y = y.cpu().numpy()

            losses = self._model.get_forward_loss_from_input(x, y)
            loss = sum(losses) / len(losses)
            selective_strategy = self.selector.select_batch_sample(losses)

            for idx, select in enumerate(selective_strategy):
                if select:
                    selective_id.append(idx)
                    self.batch_data_cached_X.append(x[idx])
                    self.batch_data_cached_y.append(y[idx])

            if len(self.batch_data_cached_X) >= self.batch_size:
                data = (np.array(self.batch_data_cached_X[: self.batch_size]),
                        np.array(self.batch_data_cached_y[: self.batch_size]))
                input_gradient = self._model.get_input_gradients(data[0], data[1])[
                    0]
                self._model.train(data)
                self.batch_data_cached_X = self.batch_data_cached_X[self.batch_size:]
                self.batch_data_cached_y = self.batch_data_cached_y[self.batch_size:]
        else:
            input_gradient = self._model.get_input_gradients(x, y)[0]
            self._model.train((x, y))
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

    def export_coae(self):
        if self.coae:
            model_bytes = TorchNNModel.get_model_bytes(self.coae)
            return model_bytes
        else:
            return None

    def restore_coae(self, model_bytes):
        if model_bytes is not None and len(model_bytes) > 0:
            coae = TorchNNModel.recover_model_bytes(model_bytes)
            self.coae = coae

    def export_model(self):
        return self._model.export_model()

    def restore_model(self, model_bytes):
        self._model = self._model.restore_model(model_bytes)

    def __repr__(self):
        return 'top model contains {}'.format(self._model.__repr__())
