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
from typing import Any, List, Literal, Optional, Protocol, Tuple, TypeVar, overload

T = TypeVar("T")


class _KeyedParty(Protocol):
    def put(self, value):
        ...

    def get(self) -> Any:
        ...


class Party(Protocol):
    def get(self, name: str) -> Any:
        ...

    @overload
    def put(self, name: str, value):
        ...

    @overload
    def put(self, **kwargs):
        ...

    def __call__(self, key: str) -> _KeyedParty:
        ...


class Parties(Protocol):
    def get(self, name: str) -> List:
        ...

    @overload
    def put(self, name: str, value):
        ...

    @overload
    def put(self, **kwargs):
        ...

    def __getitem__(self, key: int) -> Party:
        ...

    def get_neighbor(self, shift: int, module: bool = False) -> Party:
        ...

    def get_neighbors(self) -> "Parties":
        ...

    def get_local_index(self) -> Optional[int]:
        ...

    def __call__(self, key: str) -> _KeyedParty:
        ...


PartyMeta = Tuple[Literal["guest", "host", "arbiter", "local"], str]
