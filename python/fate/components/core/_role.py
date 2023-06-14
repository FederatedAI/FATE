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
from typing import Literal

T_ROLE = Literal["guest", "host", "arbiter"]


class Role:
    def __init__(self, name: T_ROLE) -> None:
        self.name: T_ROLE = name

    @property
    def is_guest(self) -> bool:
        return self.name == "guest"

    @property
    def is_host(self) -> bool:
        return self.name == "host"

    @property
    def is_arbiter(self) -> bool:
        return self.name == "arbiter"


GUEST = Role("guest")
HOST = Role("host")
ARBITER = Role("arbiter")


def load_role(role: str):
    if role == "guest":
        return GUEST
    elif role == "host":
        return HOST
    elif role == "arbiter":
        return ARBITER
    else:
        raise ValueError(f"role {role} is not supported")
