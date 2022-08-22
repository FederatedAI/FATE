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

import uuid
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from federatedml.nn.backend.fate_torch.serialization import recover_sequential_from_dict
from federatedml.nn.hetero_nn.backend.pytorch.pytorch_nn_model import PytorchNNModel
from federatedml.nn.hetero_nn.backend.pytorch.pytorch_uitl import (
    keras_nn_model_to_torch_linear,
    modify_linear_input_shape,
    torch_interactive_to_keras_nn_model,
)
from federatedml.secureprotol.paillier_tensor import PaillierTensor
from federatedml.util import consts


class DenseModel(object):
    def __init__(self):
        self.input = None
        self.model_weight = None
        self.model_shape = None
        self.bias = None
        self.model = None
        self.lr = 1.0
        self.layer_config = None
        self.role = "host"
        self.activation_placeholder_name = "activation_placeholder" + str(uuid.uuid1())
        self.activation_gradient_func = None
        self.activation_func = None
        self.is_empty_model = False
        self.activation_input = None
        self.model_builder = None
        self.input_cached = np.array([])
        self.activation_cached = np.array([])

        self.do_backward_selective_strategy = False
        self.batch_size = None

        self.use_mean_gradient = True
        self.config_type = consts.keras_backend

        self.use_torch = False

    def disable_mean_gradient(self):
        # in pytorch backend, disable mean gradient to get correct result
        self.use_mean_gradient = False

    def set_backward_selective_strategy(self):
        self.do_backward_selective_strategy = True

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def forward_dense(self, x):
        pass

    def apply_update(self, delta):
        pass

    def get_weight_gradient(self, delta):
        pass

    def build(
        self,
        input_shape=None,
        layer_config=None,
        model_builder=None,
        restore_stage=False,
        use_torch=False,
    ):
        if not input_shape:
            if self.role == "host":
                raise ValueError("host input is empty!")
            else:
                self.is_empty_model = True
                return

        self.model_builder = model_builder
        self.layer_config = layer_config
        self.use_torch = use_torch

        if not use_torch:
            self.model = model_builder(
                input_shape=input_shape,
                nn_define=layer_config,
                optimizer=SimpleNamespace(optimizer="SGD", kwargs={}),
                loss="keep_predict_loss",
                metrics=None,
            )
        else:
            torch_linear = recover_sequential_from_dict(
                modify_linear_input_shape(input_shape, layer_config)
            )
            self.model = torch_interactive_to_keras_nn_model(torch_linear)

        dense_layer = self.model.get_layer_by_index(0)

        if not restore_stage:
            self._init_model_weight(
                dense_layer
            )  # if use torch, don't have to init weight again

        if self.role == "host":
            self.activation_func = dense_layer.activation

    def export_model(self):
        if self.is_empty_model:
            return "".encode()

        layer_weights = [self.model_weight]
        if self.bias is not None:
            layer_weights.append(self.bias)

        self.model.set_layer_weights_by_index(0, layer_weights)

        if self.use_torch:
            torch_linear = keras_nn_model_to_torch_linear(self.model)
            return PytorchNNModel.get_model_bytes(torch_linear)
        else:
            return self.model.export_model()

    def restore_model(self, model_bytes):

        if self.is_empty_model:
            return

        if self.use_torch:
            torch_linear = PytorchNNModel.recover_model_bytes(model_bytes)
            self.model = torch_interactive_to_keras_nn_model(torch_linear)
        else:
            self.model = self.model.restore_model(model_bytes)
        self._init_model_weight(self.model.get_layer_by_index(0))

    def _init_model_weight(self, dense_layer):

        self.model_weight = dense_layer.trainable_weights[0].numpy()
        self.model_shape = self.model_weight.shape

        if dense_layer.use_bias:
            self.bias = dense_layer.trainable_weights[1].numpy()

    def forward_activation(self, input_data):
        self.activation_input = input_data
        output = self.activation_func(input_data)
        return output

    def backward_activation(self):
        if self.do_backward_selective_strategy:
            self.activation_input = self.activation_cached[: self.batch_size]
            self.activation_cached = self.activation_cached[self.batch_size:]
        dense_layer = self.model.get_layer_by_index(0)
        dtype = dense_layer.get_weights()[0].dtype
        with tf.GradientTape() as tape:
            activation_input = tf.constant(self.activation_input, dtype=dtype)
            tape.watch(activation_input)
            activation_output = dense_layer.activation(activation_input)
        return [tape.gradient(activation_output, activation_input).numpy()]

    def get_weight(self):
        return self.model_weight

    def get_bias(self):
        return self.bias

    def set_learning_rate(self, lr):
        self.lr = lr

    @property
    def empty(self):
        return self.is_empty_model

    @property
    def output_shape(self):
        return self.model_weight.shape[1:]


class GuestDenseModel(DenseModel):
    def __init__(self):
        super(GuestDenseModel, self).__init__()
        self.role = "guest"

    def forward_dense(self, x):
        if self.empty:
            return None

        self.input = x

        output = np.matmul(x, self.model_weight)

        return output

    def select_backward_sample(self, selective_ids):
        if self.input_cached.shape[0] == 0:
            self.input_cached = self.input[selective_ids]
        else:
            self.input_cached = np.vstack(
                (self.input_cached, self.input[selective_ids])
            )

    def get_input_gradient(self, delta):
        if self.empty:
            return None

        error = np.matmul(delta, self.model_weight.T)

        return error

    def get_weight_gradient(self, delta):
        if self.empty:
            return None

        if self.do_backward_selective_strategy:
            self.input = self.input_cached[: self.batch_size]
            self.input_cached = self.input_cached[self.batch_size:]

        if self.use_mean_gradient:
            delta_w = np.matmul(delta.T, self.input) / self.input.shape[0]
        else:
            delta_w = np.matmul(delta.T, self.input)

        return delta_w

    def apply_update(self, delta):
        if self.empty:
            return None

        self.model_weight -= self.lr * delta.T


class HostDenseModel(DenseModel):
    def __init__(self):
        super(HostDenseModel, self).__init__()
        self.role = "host"

    def select_backward_sample(self, selective_ids):
        cached_shape = self.input_cached.shape[0]
        offsets = [i + cached_shape for i in range(len(selective_ids))]
        id_map = dict(zip(selective_ids, offsets))
        if cached_shape == 0:
            self.input_cached = (
                self.input.get_obj()
                .filter(lambda k, v: k in id_map)
                .map(lambda k, v: (id_map[k], v))
            )
            self.input_cached = PaillierTensor(self.input_cached)
            # selective_ids_tb = session.parallelize(zip(selective_ids, range(len(selective_ids))), include_key=True,
            #                                        partition=self.input.partitions)
            # self.input_cached = self.input.get_obj().join(selective_ids_tb, lambda v1, v2: (v1, v2))
            # self.input_cached = PaillierTensor(tb_obj=self.input_cached.map(lambda k, v: (v[1], v[0])))
            self.activation_cached = self.activation_input[selective_ids]
        else:
            # selective_ids_tb = session.parallelize(zip(selective_ids, range(len(selective_ids))), include_key=True,
            #                                        partition=self.input.partitions)
            # selective_input = self.input.get_obj().join(selective_ids_tb, lambda v1, v2: (v1, v2))
            # pre_count = self.input_cached.shape[0]
            # selective_input = selective_input.map(lambda k, v: (v[1] + pre_count, v[0]))
            selective_input = (
                self.input.get_obj()
                .filter(lambda k, v: k in id_map)
                .map(lambda k, v: (id_map[k], v))
            )
            self.input_cached = PaillierTensor(
                self.input_cached.get_obj().union(selective_input)
            )
            self.activation_cached = np.vstack(
                (self.activation_cached, self.activation_input[selective_ids])
            )

    def forward_dense(self, x, encoder=None):
        self.input = x

        if encoder is not None:
            output = x * encoder.encode(self.model_weight)
        else:
            output = x * self.model_weight

        if self.bias is not None:
            if encoder is not None:
                output += encoder.encode(self.bias)
            else:
                output += self.bias

        return output

    def get_input_gradient(self, delta, acc_noise, encoder=None):
        if not encoder:
            error = delta * self.model_weight.T + delta * acc_noise.T
        else:
            error = delta.encode(encoder) * (self.model_weight + acc_noise).T

        return error

    def get_weight_gradient(self, delta, encoder=None):
        # delta_w = self.input.fast_matmul_2d(delta) / self.input.shape[0]
        if self.do_backward_selective_strategy:
            batch_size = self.batch_size
            self.input = PaillierTensor(
                self.input_cached.get_obj().filter(lambda k, v: k < batch_size)
            )
            self.input_cached = PaillierTensor(
                self.input_cached.get_obj()
                .filter(lambda k, v: k >= batch_size)
                .map(lambda k, v: (k - batch_size, v))
            )
            # self.input_cached = self.input_cached.subtractByKey(self.input).map(lambda kv: (kv[0] - self.batch_size, kv[1]))

        if encoder:
            delta_w = self.input.fast_matmul_2d(encoder.encode(delta))
        else:
            delta_w = self.input.fast_matmul_2d(delta)

        if self.use_mean_gradient:
            delta_w /= self.input.shape[0]

        return delta_w

    def update_weight(self, delta):
        self.model_weight -= delta * self.lr

    def update_bias(self, delta):
        if self.bias is not None:
            self.bias -= np.mean(delta, axis=0) * self.lr
