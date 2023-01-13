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
from contextlib import contextmanager
from typing import Iterable, Iterator, Protocol, Tuple, TypeVar

from ._cipher import CipherKit
from ._computing import ComputingEngine
from ._federation import FederationEngine
from ._metric import MetricsWrap
from ._party import Parties, Party

T = TypeVar("T")


class Context(Protocol):
    metrics: MetricsWrap
    guest: Party
    hosts: Parties
    arbiter: Party
    parties: Parties
    cipher: CipherKit
    computing: ComputingEngine
    federation: FederationEngine

    @contextmanager
    def sub_ctx(self, namespace) -> Iterator["Context"]:
        ...

    def range(self, end) -> Iterator[Tuple[int, "Context"]]:
        ...

    def iter(self, iterable: Iterable[T]) -> Iterator[Tuple["Context", T]]:
        ...

    def destroy(self):
        ...
