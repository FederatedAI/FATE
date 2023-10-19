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

import logging
import typing
from typing import List

from fate.arch.abc import PartyMeta
from ._gc import GarbageCollector

if typing.TYPE_CHECKING:
    from fate.arch.computing.table import KVTable

LOGGER = logging.getLogger(__name__)


class Federation:
    def __init__(self):
        self._push_history = set()
        self._pull_history = set()

        self.get_gc: GarbageCollector = GarbageCollector()
        self.remote_gc: GarbageCollector = GarbageCollector()

    def _pull_table(
        self,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ) -> List["KVTable"]:
        raise NotImplementedError(f"pull table is not supported in {self.__class__.__name__}")

    def _pull_bytes(
        self,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ) -> List[bytes]:
        raise NotImplementedError(f"pull bytes is not supported in {self.__class__.__name__}")

    def _push_table(
        self,
        table: "KVTable",
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ):
        raise NotImplementedError(f"push table is not supported in {self.__class__.__name__}")

    def _push_bytes(
        self,
        v: bytes,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ):
        raise NotImplementedError(f"push bytes is not supported in {self.__class__.__name__}")

    def push_table(
        self,
        table: "KVTable",
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ):
        for party in parties:
            if (name, tag, party) in self._push_history:
                raise ValueError(f"push table to {parties} with duplicate name and tag: name={name}, tag={tag}")
            self._push_history.add((name, tag, party))

        self.remote_gc.register_clean_action(name, tag, table, "destroy", {})
        self._push_table(
            table=table,
            name=name,
            tag=tag,
            parties=parties,
        )

    def push_bytes(
        self,
        v: bytes,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ):
        for party in parties:
            if (name, tag, party) in self._push_history:
                raise ValueError(f"push bytes to {parties} with duplicate name and tag: name={name}, tag={tag}")
            self._push_history.add((name, tag, party))

        self._push_bytes(
            v=v,
            name=name,
            tag=tag,
            parties=parties,
        )

    def pull_table(
        self,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ) -> List["KVTable"]:
        for party in parties:
            if (name, tag, party) in self._pull_history:
                raise ValueError(f"pull table from {party} with duplicate name and tag: name={name}, tag={tag}")
            self._pull_history.add((name, tag, party))

        tables = self._pull_table(
            name=name,
            tag=tag,
            parties=parties,
        )
        for table in tables:
            self.get_gc.register_clean_action(name, tag, table, "destroy", {})
        return tables

    def pull_bytes(
        self,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ) -> List[bytes]:
        for party in parties:
            if (name, tag, party) in self._pull_history:
                raise ValueError(f"pull bytes from {party} with duplicate name and tag: name={name}, tag={tag}")
            self._pull_history.add((name, tag, party))
        return self._pull_bytes(
            name=name,
            tag=tag,
            parties=parties,
        )
