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
import json
import os
import re
import typing

home_dir = os.path.split(os.path.realpath(__file__))[0]
setting_conf_dir = os.path.join(home_dir, "..", "..", "conf", "setting_conf")


class _Program:
    def __init__(self, module, module_path, class_name) -> None:
        self.module = module
        self.module_path = module_path
        self.class_name = class_name

    @classmethod
    def load(cls, module, program_str: str):
        split = program_str.split("/", -1)
        split[-2] = split[-2].replace(".py", "")
        return _Program(module, ".".join(split[:-1]), split[-1])

    def get_obj(self):
        try:
            module = importlib.import_module(self.module_path)
        except Exception as e:
            raise ValueError(
                f"module={self.module}, path={self.module_path} does not exist"
            ) from e
        try:
            obj = getattr(module, self.class_name)()
        except Exception as e:
            raise ValueError(
                f"module={self.module}, path={self.module_path}, class={self.class_name} does not exist"
            ) from e
        return obj


class SettingConf:
    def __init__(
        self, param_program: _Program, roles: typing.Dict[str, _Program]
    ) -> None:
        self.param_program = param_program
        self.roles = roles

    @classmethod
    def load(cls, module):
        path = os.path.join(setting_conf_dir, module + ".json")
        with open(path, "r") as fin:
            setting = json.loads(fin.read())
            module_path = setting["module_path"]
            param_class_path = setting["param_class"]
            roles = {}
            for key, value in setting["role"].items():
                program = _Program.load(module, f"{module_path}/{value['program']}")
                for role in re.findall(r"guest|host|arbiter|local", key):
                    roles[role] = program
            param_program = _Program.load(module, param_class_path)
            return SettingConf(param_program, roles)

    def option_map_role(self, role: str, func: typing.Callable[[_Program], typing.Any]):
        if role not in self.roles:
            return None
        else:
            return func(self.roles[role])


def get_all_module():
    return [name for name in os.listdir(setting_conf_dir) if name.endswith(".json")]
