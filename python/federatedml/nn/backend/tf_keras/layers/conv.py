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


def _build_conv1d(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1,
                  activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                  kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                  bias_constraint=None, **kwargs):
    return layers.convolutional.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)


def _build_conv2d(filters, kernel_size, strides=(1, 1), padding='valid', data_format='channels_last',
                  dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                  bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                  kernel_constraint=None, bias_constraint=None, **kwargs):
    return layers.convolutional.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)


def _build_conv3d(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format='channels_last',
                  dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                  bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                  kernel_constraint=None, bias_constraint=None, **kwargs):
    return layers.convolutional.Conv3D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        **kwargs)
