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

import importlib
import io

from ruamel import yaml
from ._serdes_base import p_dumps, Unpickler


def get_restricted_serdes():
    return WhitelistRestrictedSerdes


class WhitelistRestrictedSerdes:
    @classmethod
    def serialize(cls, obj) -> bytes:
        return p_dumps(obj)

    @classmethod
    def deserialize(cls, bytes) -> object:
        return RestrictedUnpickler(io.BytesIO(bytes)).load()


class RestrictedUnpickler(Unpickler):
    def _load(self, module, name):
        try:
            return super().find_class(module, name)
        except:
            return getattr(importlib.import_module(module), name)

    def find_class(self, module, name):
        if name in Whitelist.get_whitelist().get(module, set()):
            return self._load(module, name)
        else:
            for m in Whitelist.get_whitelist_glob():
                if module.startswith(m):
                    return self._load(module, name)
        raise ValueError(f"forbidden unpickle class {module} {name}")


class Whitelist:
    loaded = False
    deserialize_whitelist = {}
    deserialize_glob_whitelist = set()

    @classmethod
    def get_whitelist_glob(cls):
        if not cls.loaded:
            cls.load_deserialize_whitelist()
        return cls.deserialize_glob_whitelist

    @classmethod
    def get_whitelist(cls):
        if not cls.loaded:
            cls.load_deserialize_whitelist()
        return cls.deserialize_whitelist

    @classmethod
    def get_whitelist_path(cls):
        import os.path

        return os.path.abspath(
            os.path.join(
                __file__,
                os.path.pardir,
                os.path.pardir,
                os.path.pardir,
                os.path.pardir,
                os.path.pardir,
                os.path.pardir,
                "configs",
                "whitelist.yaml",
            )
        )

    @classmethod
    def load_deserialize_whitelist(cls):
        with open(cls.get_whitelist_path()) as f:
            for k, v in yaml.load(f, Loader=yaml.SafeLoader).items():
                if k.endswith("*"):
                    cls.deserialize_glob_whitelist.add(k[:-1])
                else:
                    cls.deserialize_whitelist[k] = set(v)
        cls.loaded = True
