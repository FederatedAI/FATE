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
import torch as t
import numpy as np
from federatedml.util import LOGGER
from federatedml.nn.hetero.nn_component.torch_model import TorchNNModel


class BottomModel(object):

    def __init__(self, optimizer, layer_config):

        self._model: TorchNNModel = TorchNNModel(nn_define=layer_config, optimizer_define=optimizer,
                                                 loss_fn_define=None)
        self.do_backward_select_strategy = False
        self.x = []
        self.x_cached = []
        self.batch_size = None

    def set_backward_select_strategy(self):
        self.do_backward_select_strategy = True

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def train_mode(self, mode):
        self._model.train_mode(mode)

    def forward(self, x):
        LOGGER.debug("bottom model start to forward propagation")

        self.x = x
        if self.do_backward_select_strategy:
            if (not isinstance(x, np.ndarray) and not isinstance(x, t.Tensor)):
                raise ValueError(
                    'When using selective bp, data from dataset must be a ndarray or a torch tensor, but got {}'.format(
                        type(x)))

        if self.do_backward_select_strategy:
            output_data = self._model.predict(x)
        else:
            output_data = self._model.forward(x)

        return output_data

    def backward(self, x, error, selective_ids):
        LOGGER.debug("bottom model start to backward propagation")
        if self.do_backward_select_strategy:
            if selective_ids:
                if len(self.x_cached) == 0:
                    self.x_cached = self.x[selective_ids]
                else:
                    self.x_cached = np.vstack(
                        (self.x_cached, self.x[selective_ids]))
            if len(error) == 0:
                return
            x = self.x_cached[: self.batch_size]
            self.x_cached = self.x_cached[self.batch_size:]
            self._model.train((x, error))
        else:
            self._model.backward(error)

        LOGGER.debug('bottom model update parameters:')

    def predict(self, x):
        return self._model.predict(x)

    def export_model(self):
        return self._model.export_model()

    def restore_model(self, model_bytes):
        self._model = self._model.restore_model(model_bytes)

    def __repr__(self):
        return 'bottom model contains {}'.format(self._model.__repr__())
