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
from typing import List, Optional, Protocol

from ._party import PartyMeta


class GarbageCollector(Protocol):
    def register_clean_action(self, name: str, tag: str, obj, method: str, kwargs):
        ...

    def clean(self, name: str, tag: str):
        ...


class FederationEngine(Protocol):
    session_id: str
    get_gc: Optional[GarbageCollector]
    remote_gc: Optional[GarbageCollector]
    local_party: PartyMeta
    parties: List[PartyMeta]

    def pull(self, name: str, tag: str, parties: List[PartyMeta]) -> List:
        ...

    def push(
        self,
        v,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ):
        ...

    def destroy(self):
        ...
