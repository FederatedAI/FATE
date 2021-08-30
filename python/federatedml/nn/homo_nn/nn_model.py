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

from federatedml.framework.weights import Weights


class NNModel(object):
    def get_model_weights(self) -> Weights:
        pass

    def set_model_weights(self, weights: Weights):
        pass

    def export_model(self):
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


def get_data_converter(config_type) -> DataConverter:
    if config_type == "pytorch":
        from federatedml.nn.backend.pytorch.nn_model import PytorchDataConverter

        return PytorchDataConverter()
    else:
        from federatedml.nn.backend.tf_keras.nn_model import KerasSequenceDataConverter

        return KerasSequenceDataConverter()


def get_nn_builder(config_type):
    if config_type == "nn":

        from tensorflow.keras import Sequential

        from federatedml.nn.backend.tf_keras.layers import get_builder, has_builder
        from federatedml.nn.backend.tf_keras.nn_model import KerasNNModel

        def build_nn_model(
            input_shape,
            nn_define,
            loss,
            optimizer,
            metrics,
            is_supported_layer=has_builder,
            default_layer=None,
        ) -> KerasNNModel:
            model = Sequential()
            is_first_layer = True
            for layer_config in nn_define:
                layer = layer_config.get("layer", default_layer)
                if layer and is_supported_layer(layer):
                    del layer_config["layer"]
                    if is_first_layer:
                        layer_config["input_shape"] = input_shape
                        is_first_layer = False
                    builder = get_builder(layer)
                    model.add(builder(**layer_config))

                else:
                    raise ValueError(f"dnn not support layer {layer}")

            keras_model = KerasNNModel(model)
            keras_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

            return keras_model

        return build_nn_model

    elif config_type == "keras":
        from federatedml.nn.backend.tf_keras.nn_model import build_keras

        return build_keras
    elif config_type == "pytorch":
        from federatedml.nn.backend.pytorch.nn_model import build_pytorch

        return build_pytorch
    else:
        raise ValueError(f"{config_type} is not supported")


def restore_nn_model(config_type, model_bytes):
    if config_type == "nn" or config_type == "keras":
        from federatedml.nn.backend.tf_keras.nn_model import KerasNNModel

        return KerasNNModel.restore_model(model_bytes)

    elif config_type == "pytorch":
        from federatedml.nn.backend.pytorch.nn_model import PytorchNNModel

        return PytorchNNModel.restore_model(model_bytes)

    else:
        raise ValueError(f"{config_type} is not supported")
