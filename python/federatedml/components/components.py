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
import typing
from pathlib import Path

from federatedml.model_base import ModelBase
from federatedml.param.base_param import BaseParam
from federatedml.util import LOGGER

_ml_base = Path(__file__).resolve().parent.parent.parent


class _RunnerDocorator:
    def __init__(self, meta) -> None:
        self._roles = set()
        self._meta = meta

    @property
    def on_guest(self):
        self._roles.add("guest")
        return self

    @property
    def on_host(self):
        self._roles.add("host")
        return self

    @property
    def on_arbiter(self):
        self._roles.add("arbiter")
        return self

    @property
    def on_local(self):
        self._roles.add("local")
        return self

    def __call__(self, cls):
        if inspect.isclass(cls) and issubclass(cls, ModelBase):
            for role in self._roles:
                self._meta._role_to_runner_cls[role] = cls
        elif inspect.isfunction(cls):
            for role in self._roles:
                self._meta._role_to_runner_cls_getter[role] = cls
        else:
            raise NotImplementedError(f"type of {cls} not supported")

        return cls


class ComponentMeta:
    __name_to_obj: typing.Dict[str, "ComponentMeta"] = {}

    def __init__(self, name) -> None:
        self.name = name
        self._role_to_runner_cls = {}
        self._role_to_runner_cls_getter = {}  # lazy
        self._param_cls = None
        self._param_cls_getter = None  # lazy

        self.__name_to_obj[name] = self

    @classmethod
    def get_meta(cls, name):
        return cls.__name_to_obj[name]

    @property
    def bind_runner(self):
        return _RunnerDocorator(self)

    @property
    def bind_param(self):
        def _wrap(cls):
            if inspect.isclass(cls) and issubclass(cls, BaseParam):
                self._param_cls = cls
            elif inspect.isfunction(cls):
                self._param_cls_getter = cls
            else:
                raise NotImplementedError(f"type of {cls} not supported")
            return cls

        return _wrap

    def register_info(self):
        return {
            self.name: dict(
                module=self.__module__,
            )
        }

    def _get_runner(self, role: str):
        if role in self._role_to_runner_cls:
            return self._role_to_runner_cls[role]

        elif role in self._role_to_runner_cls_getter:
            return self._role_to_runner_cls_getter[role]()

        else:
            raise ModuleNotFoundError(
                f"Runner for component `{self.name}` at role `{role}` not found"
            )

    def get_run_obj(self, role: str):
        return self._get_runner(role)()

    def get_run_obj_name(self, role: str) -> str:
        return self._get_runner(role).__name__

    def get_param_obj(self, cpn_name: str):
        if self._param_cls is not None:
            param_obj = self._param_cls()
        elif self._param_cls_getter is not None:
            param_obj = self._param_cls_getter()()
        else:
            raise ModuleNotFoundError(f"Param for component `{self.name}` not found")
        return param_obj.set_name(f"{self.name}#{cpn_name}")

    def get_supported_roles(self):
        return set(self._role_to_runner_cls) | set(self._role_to_runner_cls_getter)


def _search_components(path):
    try:
        module_name = (
            path.absolute()
            .relative_to(_ml_base)
            .with_suffix("")
            .__str__()
            .replace("/", ".")
        )
        module = importlib.import_module(module_name)
    except ImportError as e:
        # or skip ?
        raise e
    _obj_pairs = inspect.getmembers(module, lambda obj: isinstance(obj, ComponentMeta))
    return _obj_pairs, module_name


class Components:
    provider_version = None
    provider_name = None

    @classmethod
    def get_names(cls) -> typing.Dict[str, dict]:
        names = {}
        _components_base = Path(__file__).resolve().parent
        for p in _components_base.glob("**/*.py"):
            obj_pairs, module_name = _search_components(p)
            for name, obj in obj_pairs:
                names[obj.name] = {"module": module_name}
                LOGGER.info(f"component register {obj.name} with cache info {module_name}")
        return names

    @classmethod
    def get(cls, name: str, cache) -> ComponentMeta:
        if cache:
            importlib.import_module(cache[name]["module"])

        else:
            _components_base = Path(__file__).resolve().parent
            for p in _components_base.glob("**/*.py"):
                module_name = (
                    p.absolute()
                    .relative_to(_ml_base)
                    .with_suffix("")
                    .__str__()
                    .replace("/", ".")
                )
                importlib.import_module(module_name)

        return ComponentMeta.get_meta(name)
