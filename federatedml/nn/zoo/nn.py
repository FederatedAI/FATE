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

from tensorflow.keras import Sequential
from federatedml.nn.backend.tf_keras.layers import get_builder, has_builder
from federatedml.nn.backend.tf_keras.nn_model import from_keras_sequential_model, KerasNNModel, \
    restore_keras_nn_model



def build_nn_model(input_shape, nn_define, loss, optimizer, metrics,
                   is_supported_layer=has_builder,
                   default_layer=None) -> KerasNNModel:
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

    return from_keras_sequential_model(model=model,
                                       loss=loss,
                                       optimizer=optimizer,
                                       metrics=metrics)


def restore_nn_model(config_type, model_bytes):
    if config_type =="pytorch":
        from federatedml.nn.backend.pytorch.nn_model import restore_pytorch_nn_model
        return restore_pytorch_nn_model(model_bytes)
    elif config_type =="cv":
        from federatedml.nn.backend.pytorch.nn_model import restore_pytorch_nn_model
        return restore_pytorch_nn_model(model_bytes)
    else:
        return restore_keras_nn_model(model_bytes)



