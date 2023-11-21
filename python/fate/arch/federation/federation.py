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
import traceback
import typing
from typing import List

from fate.arch.abc import PartyMeta
from ._gc import GarbageCollector

if typing.TYPE_CHECKING:
    from fate.arch.computing.table import KVTable

logger = logging.getLogger(__name__)


def _federation_info(func):
    def wrapper(*args, **kwargs):
        name = kwargs.get("name")
        tag = kwargs.get("tag")
        logger.debug(f"federation enter {func.__name__}: name={name}, tag={tag}")
        try:
            stacks = "".join(traceback.format_stack(limit=7)[:-1])
            logger.debug(f"stack:\n{stacks}")
            return func(*args, **kwargs)
        finally:
            logger.debug(f"federation exit {func.__name__}: name={name}, tag={tag}")

    return wrapper


class Federation:
    def __init__(self, session_id: str, party: PartyMeta, parties: List[PartyMeta]):
        logger.debug(f"[federation]initializing({self.__class__.__name__}): {session_id=}, {party=}, {parties=}")
        self._session_id = session_id
        self._local_party = party
        self._parties = parties
        self._push_history = set()
        self._pull_history = set()
        self._get_gc: GarbageCollector = GarbageCollector()
        self._remote_gc: GarbageCollector = GarbageCollector()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def local_party(self) -> PartyMeta:
        return self._local_party

    @property
    def parties(self) -> List[PartyMeta]:
        return self._parties

    @property
    def world_size(self) -> int:
        return len(self._parties)

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

    @_federation_info
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

        self._remote_gc.register_clean_action(name, tag, table, "destroy", {})
        self._push_table(
            table=table,
            name=name,
            tag=tag,
            parties=parties,
        )

    @_federation_info
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

    @_federation_info
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
            self._get_gc.register_clean_action(name, tag, table, "destroy", {})
        return tables

    @_federation_info
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
