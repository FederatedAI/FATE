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

from fate.arch.trace import (
    federation_push_table_trace,
    federation_pull_table_trace,
    federation_push_bytes_trace,
    federation_pull_bytes_trace,
)
from ._table_meta import TableMeta
from ._type import PartyMeta

if typing.TYPE_CHECKING:
    from fate.arch.computing.api import KVTable

logger = logging.getLogger(__name__)


class Federation:
    def __init__(self, session_id: str, party: PartyMeta, parties: List[PartyMeta]):
        logger.debug(f"[federation]initializing({self.__class__.__name__}): {session_id=}, {party=}, {parties=}")
        if session_id is None:
            raise ValueError("session_id is None")
        self._session_id = session_id
        self._local_party = party
        self._parties = parties
        self._push_history = set()
        self._pull_history = set()

    def get_default_max_message_size(self):
        from fate.arch.config import cfg

        return cfg.federation.split_large_object.max_message_size

    def get_default_partition_num(self):
        from fate.arch.config import cfg

        return cfg.federation.split_large_object.partition_num

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
        table_metas: List[TableMeta],
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

    def _destroy(self):
        raise NotImplementedError(f"destroy is not supported in {self.__class__.__name__}")

    def destroy(self):
        self._destroy()

    @federation_push_table_trace
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

        self._push_table(
            table=table,
            name=name,
            tag=tag,
            parties=parties,
        )

    @federation_push_bytes_trace
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

    @federation_pull_table_trace
    def pull_table(
        self,
        name: str,
        tag: str,
        parties: List[PartyMeta],
        table_metas: List[TableMeta] = None,
    ) -> List["KVTable"]:
        for party in parties:
            if (name, tag, party) in self._pull_history:
                raise ValueError(f"pull table from {party} with duplicate name and tag: name={name}, tag={tag}")
            self._pull_history.add((name, tag, party))

        tables = self._pull_table(
            name=name,
            tag=tag,
            parties=parties,
            table_metas=table_metas,
        )
        for table in tables:
            table.mask_federated_received()
        return tables

    @federation_pull_bytes_trace
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
