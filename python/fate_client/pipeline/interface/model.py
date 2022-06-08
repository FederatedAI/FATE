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


class Model(object):
    def __init__(self, model=None, isometric_model=None):
        self._model = model
        self._isometric_model = isometric_model

    def __getattr__(self, model_key):
        if model_key == "model":
            return self.model
        elif model_key == "isometric_model":
            return self._isometric_model
        else:
            raise ValueError("model key {} not support".format(model_key))
