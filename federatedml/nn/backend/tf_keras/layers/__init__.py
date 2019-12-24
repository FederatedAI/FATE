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

from .baisc import _build_dense, _build_dropout, _build_flatten
from .conv import _build_conv1d, _build_conv2d, _build_conv3d
from .pooling import _build_maxpooling1d, _build_maxpooling2d, _build_maxpooling3d, _build_averagepooling3d

DENSE = "Dense".lower()
DROPOUT = "Dropout".lower()
FLATTEN = "Flatten".lower()

CONV_1D = "Conv1D".lower()
CONV_2D = "Conv2D".lower()
CONV_3D = "Conv3D".lower()

MAX_POOLING_1D = "MaxPooling1D".lower()
MAX_POOLING_2D = "MaxPooling2D".lower()
MAX_POOLING_3D = "MaxPooling3D".lower()

layer2builder = {
    DENSE: _build_dense,
    DROPOUT: _build_dropout,
    FLATTEN: _build_flatten,
    CONV_1D: _build_conv1d,
    CONV_2D: _build_conv2d,
    CONV_3D: _build_conv3d,
    MAX_POOLING_1D: _build_maxpooling1d,
    MAX_POOLING_2D: _build_maxpooling2d,
    MAX_POOLING_3D: _build_maxpooling3d
}


def get_builder(layer):
    return layer2builder.get(layer.lower())


def has_builder(layer):
    return layer.lower() in layer2builder
