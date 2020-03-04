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
import inspect
import json
import os
import pathlib
import re

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
    except ImportError:
        pass

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


# noinspection PyProtectedMember
def main():
    BaseTransferVariables._disable__singleton()
    base = pathlib.Path(__file__).parent.parent
    json_save_path = "./definition"
    if not os.path.exists(json_save_path):
        os.mkdir(json_save_path)
    for path, _, files in os.walk(pathlib.Path(__file__).parent.parent):
        path = os.path.relpath(path, base.parent)
        for file_name in files:
            if file_name.endswith(".py"):
                name = f"{path.replace('/', '.')}.{file_name[:-3]}"
                for cls in search_subclasses(name, BaseTransferVariables):
                    class_name = cls.__name__
                    file_name = _Camel2Snake.to_snake(class_name)
                    variable_dict = {}
                    for k, v in search_transfer_variable(cls()).items():
                        variable_dict[k] = {"src": v._src, "dst": v._dst}
                    with open(f"{json_save_path}/{file_name}.json", "w") as g:
                        json.dump({class_name: variable_dict}, g, indent=2)


if __name__ == '__main__':
    main()
