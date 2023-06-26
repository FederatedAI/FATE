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

from typing import Any, Dict, Union

import pydantic


class Role(str):
    def __init__(self, name) -> None:
        self.name = name

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type="string", format="role")

    @classmethod
    def __get_validators__(cls) -> "CallableGenerator":
        yield cls.validate

    @classmethod
    def validate(cls, value: Union[str]) -> str:
        return value

    @property
    def is_guest(self) -> bool:
        return self.name == "guest"

    @property
    def is_host(self) -> bool:
        return self.name == "host"

    @property
    def is_arbiter(self) -> bool:
        return self.name == "arbiter"

    @property
    def local(self) -> bool:
        return self.name == "local"

    @classmethod
    def from_str(cls, role: str):
        if role == "local":
            return LOCAL
        if role == "guest":
            return GUEST
        elif role == "host":
            return HOST
        elif role == "arbiter":
            return ARBITER
        else:
            raise ValueError(f"role {role} is not supported")

    def __str__(self):
        return f"Role<{self.name}>"

    def __repr__(self):
        return f"Role<{self.name}>"


GUEST = Role("guest")
HOST = Role("host")
ARBITER = Role("arbiter")
LOCAL = Role("local")
