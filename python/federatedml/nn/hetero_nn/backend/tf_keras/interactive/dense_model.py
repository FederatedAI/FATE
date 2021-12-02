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
from tensorflow.python.keras.backend import gradients
from tensorflow.python.keras.backend import set_session

from fate_arch.session import computing_session as session
from federatedml.secureprotol.paillier_tensor import PaillierTensor
from federatedml.util import LOGGER


try:
    from tensorflow import get_default_graph, initialize_all_variables, placeholder
except ImportError:
    from tensorflow.compat.v1 import (
        get_default_graph,
        initialize_all_variables,
        placeholder,
    )


def _init_session():
    from tensorflow.python.keras import backend

    sess = backend.get_session()
    get_default_graph()
    set_session(sess)
    return sess


class DenseModel(object):
    def __init__(self):
        self.input = None
        self.model_weight = None
        self.model_shape = None
        self.bias = None
        self.model = None
        self.lr = 1.0
        self.layer_config = None
        self.sess = _init_session()
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

    def set_backward_selective_strategy(self):
        self.do_backward_selective_strategy = True

    def forward_dense(self, x):
        pass

    def apply_update(self, delta):
        pass

    def get_weight_gradient(self, delta):
        pass

    def set_sess(self, sess):
        self.sess = sess

    def build(
        self,
        input_shape=None,
        layer_config=None,
        model_builder=None,
        restore_stage=False,
    ):
        if not input_shape:
            if self.role == "host":
                raise ValueError("host input is empty!")
            else:
                self.is_empty_model = True
                return

        self.model_builder = model_builder

        self.layer_config = layer_config

        self.model = model_builder(
            input_shape=input_shape,
            nn_define=layer_config,
            optimizer=SimpleNamespace(optimizer="SGD", kwargs={}),
            loss="keep_predict_loss",
            metrics=None,
        )

        dense_layer = self.model.get_layer_by_index(0)
        if not restore_stage:
            self._init_model_weight(dense_layer)

        if self.role == "host":
            self.activation_func = dense_layer.activation
            self.__build_activation_layer_gradients_func(dense_layer)

    def export_model(self):
        if self.is_empty_model:
            return "".encode()

        layer_weights = [self.model_weight]
        if self.bias is not None:
            layer_weights.append(self.bias)

        self.model.set_layer_weights_by_index(0, layer_weights)
        return self.model.export_model()

    def restore_model(self, model_bytes):
        if self.is_empty_model:
            return

        # LOGGER.debug("model_bytes is {}".format(model_bytes))
        self.model = self.model.restore_model(model_bytes)
        self._init_model_weight(self.model.get_layer_by_index(0), restore_stage=True)

    def _init_model_weight(self, dense_layer, restore_stage=False):
        if not restore_stage:
            self.sess.run(initialize_all_variables())

        trainable_weights = self.sess.run(dense_layer.trainable_weights)
        self.model_weight = trainable_weights[0]
        self.model_shape = dense_layer.get_weights()[0].shape

        if dense_layer.use_bias:
            self.bias = trainable_weights[1]

    def __build_activation_layer_gradients_func(self, dense_layer):
        shape = dense_layer.output_shape
        dtype = dense_layer.get_weights()[0].dtype

        input_data = placeholder(
            shape=shape, dtype=dtype, name=self.activation_placeholder_name
        )

        self.activation_gradient_func = gradients(
            dense_layer.activation(input_data), input_data
        )

    def forward_activation(self, input_data):
        self.activation_input = input_data
        output = self.activation_func(input_data)
        if not isinstance(output, np.ndarray):
            output = self.sess.run(output)

        return output

    def backward_activation(self):
        placeholder = get_default_graph().get_tensor_by_name(
            ":".join([self.activation_placeholder_name, "0"])
        )
        if self.do_backward_selective_strategy:
            self.activation_input = self.activation_cached[: self.batch_size]
            self.activation_cached = self.activation_cached[self.batch_size:]

        return self.sess.run(
            self.activation_gradient_func,
            feed_dict={placeholder: self.activation_input},
        )

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

        delta_w = np.matmul(delta.T, self.input) / self.input.shape[0]

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
            self.input = self.input_cached.filter(lambda k, v: k < self.batch_size)
            self.input_cached = self.input_cached.filter(
                lambda k, v: k >= self.batch_size
            ).map(lambda kv: (kv[0] - self.batch_size, kv[1]))
            # self.input_cached = self.input_cached.subtractByKey(self.input).map(lambda kv: (kv[0] - self.batch_size, kv[1]))

        if encoder:
            delta_w = self.input.fast_matmul_2d(encoder.encode(delta))
        else:
            delta_w = self.input.fast_matmul_2d(delta)

        delta_w /= self.input.shape[0]

        return delta_w

    def update_weight(self, delta):
        self.model_weight -= delta * self.lr

    def update_bias(self, delta):
        self.bias -= np.mean(delta, axis=0) * self.lr
