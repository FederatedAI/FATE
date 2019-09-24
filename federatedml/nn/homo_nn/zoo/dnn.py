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

import typing

from federatedml.nn.homo_nn.backend.tf_keras.layers import has_builder, DENSE, DROPOUT
from federatedml.nn.homo_nn.backend.tf_keras.nn_model import KerasNNModel, KerasSequenceDataConverter
from federatedml.nn.homo_nn.zoo.nn import build_nn


def is_supported_layer(layer):
    return has_builder(layer) and layer in {DENSE, DROPOUT}


def build_dnn(nn_define, loss, optimizer, metrics) -> typing.Tuple[KerasNNModel, KerasSequenceDataConverter]:
    return build_nn(nn_define=nn_define,
                    loss=loss,
                    optimizer=optimizer,
                    metrics=metrics,
                    is_supported_layer=is_supported_layer,
                    default_layer=DENSE)
