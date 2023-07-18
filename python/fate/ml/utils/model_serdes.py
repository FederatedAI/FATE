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
from typing import Dict, Tuple


def serialize_models(models):
    from google.protobuf import json_format

    serialized_models: Dict[str, Tuple[str, bytes, dict]] = {}

    for model_name, buffer_object in models.items():
        serialized_string = buffer_object.SerializeToString()
        pb_name = type(buffer_object).__name__
        json_format_dict = json_format.MessageToDict(
            buffer_object, including_default_value_fields=True)

        serialized_models[model_name] = (
            pb_name,
            serialized_string,
            json_format_dict,
        )

    return serialized_models
