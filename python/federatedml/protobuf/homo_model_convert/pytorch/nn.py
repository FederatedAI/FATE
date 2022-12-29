#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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

import io
import torch as t
import tempfile
from ..component_converter import ComponentConverterBase


class NNComponentConverter(ComponentConverterBase):

    @staticmethod
    def get_target_modules():
        return ['HomoNN']

    def convert(self, model_dict):

        param_obj = model_dict["HomoNNParam"]
        meta_obj = model_dict["HomoNNMeta"]

        if not hasattr(param_obj, 'model_bytes'):
            raise ValueError("Did not find model_bytes in model param protobuf")

        with tempfile.TemporaryFile() as f:
            f.write(param_obj.model_bytes)
            f.seek(0)
            model_dict = t.load(f)

        return model_dict
