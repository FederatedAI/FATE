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
import argparse
import importlib
import inspect
import json
import os
import re

from arch.api.utils import file_utils
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables, Variable


class _Camel2Snake(object):
    __pattern = re.compile('((?<=[a-z0-9])[A-Z]|(?!^)[A-Z](?=[a-z]))')

    @classmethod
    def to_snake(cls, camel):
        return cls.__pattern.sub(r'_\1', camel).lower()


# noinspection PyBroadException
def search_subclasses(module_name, class_type):
    ret = []
    try:
        module = importlib.import_module(module_name)
        _class = inspect.getmembers(module, inspect.isclass)
        _subclass = [m for m in _class if issubclass(m[1], class_type) and m[1] != class_type]
        ret = [m[1] for m in _subclass if m[1].__module__ == module.__name__]
    except ImportError as e:
        print(f"import {module_name} fail, {e.args}, skip")
    return ret


def search_transfer_variable(transfer_variable):
    fields = transfer_variable.__dict__.items()
    d = {}
    for k, v in fields:
        if isinstance(v, Variable):
            name = v.name.split(".", 1)[-1]
            d[name] = v
        elif isinstance(v, BaseTransferVariables):
            d.update(search_transfer_variable(v))
    return d


DEFAULT_SRC = os.path.join(file_utils.get_project_base_directory(), "federatedml")
DEFAULT_DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth_conf", "federatedml")


# noinspection PyProtectedMember
def main():
    parser = argparse.ArgumentParser(description="generate conf from transfer variable class")
    parser.add_argument("-s", "--src", type=str, default=DEFAULT_SRC,
                        help=f"dir to search transfer classes, defaults to {DEFAULT_SRC}")
    parser.add_argument("-d", "--dst", type=str, default=DEFAULT_DST,
                        help=f"dir to save generated json files, defaults to {DEFAULT_DST}")

    arg = parser.parse_args()
    dst = arg.dst
    if not os.path.exists(dst):
        os.mkdir(dst)
    if not os.path.isdir(dst):
        raise ValueError(f"{dst} should be a directory")
    src = arg.src
    if not os.path.exists(src):
        raise ValueError(f"{src} not exists")

    BaseTransferVariables._disable__singleton()
    base = file_utils.get_project_base_directory()

    for path, _, files in os.walk(src):
        path = os.path.relpath(path, base)
        for file_name in files:
            if file_name.endswith(".py"):
                name = f"{path.replace('/', '.')}.{file_name[:-3]}"
                for cls in search_subclasses(name, BaseTransferVariables):
                    class_name = cls.__name__
                    file_name = _Camel2Snake.to_snake(class_name)
                    variable_dict = {}
                    for k, v in search_transfer_variable(cls()).items():
                        variable_dict[k] = {"src": v._src, "dst": v._dst}

                    if len(variable_dict) < 1:
                        continue
                    saving_path = os.path.join(dst, f"{file_name}.json")

                    print(f"found transfer_class: {class_name} in {name}, saving to {saving_path}")
                    with open(saving_path, "w") as g:
                        json.dump({class_name: variable_dict}, g, indent=2)


if __name__ == '__main__':
    main()
