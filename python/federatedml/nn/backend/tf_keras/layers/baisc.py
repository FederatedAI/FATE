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
from tensorflow.python.keras import layers
from .util import _get_initializer


def _build_dense(units, activation, use_bias=True, kernel_initializer="glorot_uniform",
                 bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, seed=None, **kwargs):
    return layers.Dense(units=units,
                        activation=activation,
                        use_bias=use_bias,
                        kernel_initializer=_get_initializer(kernel_initializer, seed),
                        bias_initializer=_get_initializer(bias_initializer, seed),
                        kernel_regularizer=kernel_regularizer,
                        bias_regularizer=bias_regularizer,
                        activity_regularizer=activity_regularizer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=bias_constraint,
                        **kwargs)


def _build_dropout(rate, noise_shape=None, seed=None, **kwargs):
    return layers.Dropout(rate, noise_shape=noise_shape, seed=seed, **kwargs)


def _build_flatten(data_format=None, **kwargs):
    return layers.Flatten(data_format=data_format, **kwargs)

