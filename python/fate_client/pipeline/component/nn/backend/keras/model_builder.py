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
import json

_TF_KERAS_VALID = False
try:
    from tensorflow.keras.models import Sequential
    _TF_KERAS_VALID = True
except ImportError:
    pass


def build_model(model_type="sequential"):
    if model_type != "sequential":
        raise ValueError("Only support sequential model now")

    return SequentialModel()


class SequentialModel(object):
    def __init__(self):
        if _TF_KERAS_VALID:
            self._model = Sequential()
        else:
            self._model = None

    def add(self, layer):
        if not _TF_KERAS_VALID:
            raise ImportError("Please install tensorflow first, "
                              "can not import sequential model from tensorflow.keras.model !!!")

        self._model.add(layer)

    @staticmethod
    def get_loss_config(loss):
        if isinstance(loss, str):
            return loss

        if loss.__module__ == "tensorflow.python.keras.losses":
            return loss.__name__

        raise ValueError("keras sequential model' loss should be string of losses function of tf_keras")

    @staticmethod
    def get_optimizer_config(optimizer):
        if isinstance(optimizer, str):
            return optimizer

        opt_config = optimizer.get_config()
        if "name" in opt_config:
            opt_config["optimizer"] = opt_config["name"]
            del opt_config["name"]

        return opt_config

    def get_network_config(self):
        if not _TF_KERAS_VALID:
            raise ImportError("Please install tensorflow first, "
                              "can not import sequential model from tensorflow.keras.model !!!")

        return json.loads(self._model.to_json())
