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

from arch.api.utils import log_utils
from federatedml.framework.weights import Weights

LOGGER = log_utils.getLogger()


class NNModel(object):

    def get_model_weights(self) -> Weights:
        pass

    def set_model_weights(self, weights: Weights):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass

    def train(self, data, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, **kwargs):
        pass

    def modify(self, func: typing.Callable[[Weights], Weights]) -> Weights:
        weights = self.get_model_weights()
        self.set_model_weights(func(weights))
        return weights


class DataConverter(object):
    def convert(self, data, *args, **kwargs):
        pass


def parse(config_type, nn_define, **kwargs) -> typing.Tuple[NNModel, DataConverter]:
    if config_type == "keras":
        from federatedml.nn.homo_nn.backend.tf_keras.nn_model import build_keras
        return build_keras(nn_define, **kwargs)

    elif config_type == "nn":
        from federatedml.nn.homo_nn.zoo.nn import build_nn
        return build_nn(nn_define, **kwargs)

    elif config_type == "dnn":
        from federatedml.nn.homo_nn.zoo.dnn import build_dnn
        return build_dnn(nn_define, **kwargs)

    # if config_type == "tf":
    #     from federatedml.nn.homo_nn.backend.tf.nn_model import TFFitDictDataConverter
    #     return None, TFFitDictDataConverter()
