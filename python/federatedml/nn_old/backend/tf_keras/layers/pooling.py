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


def _build_maxpooling1d(pool_size=2,
                        strides=None,
                        padding='valid',
                        data_format=None,
                        **kwargs):
    return layers.pooling.MaxPooling1D(pool_size=pool_size,
                                       strides=strides,
                                       padding=padding,
                                       data_format=data_format,
                                       **kwargs)


def _build_maxpooling2d(pool_size=(2, 2),
                        strides=None,
                        padding='valid',
                        data_format=None,
                        **kwargs):
    return layers.pooling.MaxPooling2D(pool_size=pool_size,
                                       strides=strides,
                                       padding=padding,
                                       data_format=data_format,
                                       **kwargs)


def _build_maxpooling3d(pool_size=(2, 2, 2),
                        strides=None,
                        padding='valid',
                        data_format=None,
                        **kwargs):
    return layers.pooling.MaxPooling3D(pool_size=pool_size,
                                       strides=strides,
                                       padding=padding,
                                       data_format=data_format,
                                       **kwargs)


def _build_averagepooling1d(pool_size=2,
                            strides=None,
                            padding='valid',
                            data_format=None,
                            **kwargs):
    return layers.pooling.AveragePooling1D(pool_size=pool_size,
                                           strides=strides,
                                           padding=padding,
                                           data_format=data_format,
                                           **kwargs)


def _build_averagepooling2d(pool_size=(2, 2),
                            strides=None,
                            padding='valid',
                            data_format=None,
                            **kwargs):
    return layers.pooling.AveragePooling2D(pool_size=pool_size,
                                           strides=strides,
                                           padding=padding,
                                           data_format=data_format,
                                           **kwargs)


def _build_averagepooling3d(pool_size=(2, 2, 2),
                            strides=None,
                            padding='valid',
                            data_format=None,
                            **kwargs):
    return layers.pooling.AveragePooling3D(pool_size=pool_size,
                                           strides=strides,
                                           padding=padding,
                                           data_format=data_format,
                                           **kwargs)


_build_global_averagepooling1d = layers.pooling.GlobalAveragePooling1D.__init__

_build_global_averagepooling2d = layers.pooling.GlobalAveragePooling2D.__init__

_build_global_averagepooling3d = layers.pooling.GlobalAveragePooling3D.__init__

_build_global_maxpooling1d = layers.pooling.GlobalMaxPooling1D.__init__

_build_global_maxpooling2d = layers.pooling.GlobalMaxPooling2D.__init__

_build_global_maxpooling3d = layers.pooling.GlobalMaxPooling3D.__init__
