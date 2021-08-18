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

import importlib
from pathlib import Path
import inspect


def get_proto_buffer_class(buffer_name):
    package_base_path = Path(__file__).absolute().parent.parent.parent
    package_path = Path(__file__).absolute().parent.joinpath("generated")
    for f in package_path.glob("*.py"):
        module_rel_path = package_path.joinpath(f.stem).relative_to(package_base_path)
        module_path = f"{module_rel_path}".replace("/", ".")
        proto_module = importlib.import_module(module_path)
        for name, obj in inspect.getmembers(proto_module):
            if inspect.isclass(obj) and name == buffer_name:
                return obj
    raise ModuleNotFoundError(buffer_name)


def deserialize_models(model_input):
    for model_type, models in model_input.items():
        for cpn_name, cpn_models in models.items():
            for model_name, (pb_name, pb_buffer) in cpn_models.items():
                pb_object = get_proto_buffer_class(pb_name)()
                pb_object.ParseFromString(pb_buffer)
                model_input[model_type][cpn_name][model_name] = pb_object
