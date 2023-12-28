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
from typing import List, Tuple, TypeVar, Union

from fate.arch.federation.api import PartyMeta, TableRemotePersistentPickler, TableRemotePersistentUnpickler
from fate.arch.trace import federation_get_timer, federation_remote_timer
from ._namespace import NS

logger = logging.getLogger(__name__)
T = TypeVar("T")

if typing.TYPE_CHECKING:
    from fate.arch.context import Context
    from fate.arch.federation.api import Federation
    from fate.arch.computing.api import KVTableContext


class _KeyedParty:
    def __init__(self, party: Union["Party", "Parties"], key) -> None:
        self._party = party
        self._key = key

    def put(self, value):
        return self._party.put(self._key, value)

    def get(self):
        return self._party.get(self._key)


class Party:
    def __init__(
        self,
        ctx: "Context",
        federation: "Federation",
        computing: "KVTableContext",
        party: PartyMeta,
        rank: int,
        namespace: NS,
        key=None,
    ) -> None:
        self._ctx = ctx
        self._party = party
        self.federation = federation
        self.computing = computing
        self.rank = rank
        self.namespace = namespace

    def __call__(self, key: str) -> "_KeyedParty":
        return _KeyedParty(self, key)

    def __str__(self):
        return f"{self.__class__.__name__}(party={self.party}, rank={self.rank}, namespace={self.namespace})"

    @property
    def party(self) -> PartyMeta:
        return self._party

    @property
    def role(self) -> str:
        return self.party[0]

    @property
    def party_id(self) -> str:
        return self.party[1]

    @property
    def name(self) -> str:
        return f"{self.party[0]}_{self.party[1]}"

    def put(self, *args, **kwargs):
        if args:
            assert len(args) == 2 and isinstance(args[0], str), "invalid position parameter"
            assert not kwargs, "keywords parameters not allowed when position parameter provided"
            kvs = [args]
        else:
            kvs = kwargs.items()

        for k, v in kvs:
            return _push(
                federation=self.federation,
                computing=self.computing,
                name=k,
                namespace=self.namespace,
                parties=[self.party],
                value=v,
                max_message_size=self.federation.get_default_max_message_size(),
                num_partitions_of_slice_table=self.federation.get_default_partition_num(),
            )

    def get(self, name: str):
        return _pull(self._ctx, self.federation, name, self.namespace, [self.party])[0]

    def get_int(self, name: str):
        ...


class Parties:
    def __init__(
        self,
        ctx: "Context",
        federation: "Federation",
        computing: "KVTableContext",
        parties: List[Tuple[int, PartyMeta]],
        namespace: NS,
    ) -> None:
        self._ctx = ctx
        self.federation = federation
        self.computing = computing
        self.parties = parties
        self.namespace = namespace

    def __str__(self):
        return f"{self.__class__.__name__}(parties={self.parties}, namespace={self.namespace})"

    @property
    def ranks(self):
        return [p[0] for p in self.parties]

    def __getitem__(self, key: int) -> Party:
        rank, party = self.parties[key]
        return Party(self._ctx, self.federation, self.computing, party, rank, self.namespace)

    def __iter__(self):
        return iter(
            [
                Party(self._ctx, self.federation, self.computing, party, rank, self.namespace)
                for rank, party in self.parties
            ]
        )

    def __len__(self) -> int:
        return len(self.parties)

    def __call__(self, key: str) -> "_KeyedParty":
        return _KeyedParty(self, key)

    def put(self, *args, **kwargs):
        if args:
            assert len(args) == 2 and isinstance(args[0], str), "invalid position parameter"
            assert not kwargs, "keywords parameters not allowed when position parameter provided"
            kvs = [args]
        else:
            kvs = kwargs.items()
        for k, v in kvs:
            return _push(
                federation=self.federation,
                computing=self.computing,
                name=k,
                namespace=self.namespace,
                parties=[p[1] for p in self.parties],
                value=v,
                max_message_size=self.federation.get_default_max_message_size(),
                num_partitions_of_slice_table=self.federation.get_default_partition_num(),
            )

    def get(self, name: str):
        return _pull(self._ctx, self.federation, name, self.namespace, [p[1] for p in self.parties])


def _push(
    federation: "Federation",
    computing: "KVTableContext",
    name: str,
    namespace: NS,
    parties: List[PartyMeta],
    value,
    max_message_size,
    num_partitions_of_slice_table,
):
    tag = namespace.federation_tag
    timer = federation_remote_timer(name=name, full_name=name, tag=tag, local=federation.local_party, parties=parties)
    TableRemotePersistentPickler.push(
        value,
        federation,
        computing,
        name,
        tag,
        parties,
        max_message_size=max_message_size,
        num_partitions_of_slice_table=num_partitions_of_slice_table,
    )
    timer.done()


def _pull(
    ctx: "Context",
    federation: "Federation",
    name: str,
    namespace: NS,
    parties: List[PartyMeta],
):
    tag = namespace.federation_tag
    timer = federation_get_timer(name=name, full_name=name, tag=tag, local=federation.local_party, parties=parties)
    buffer_list = federation.pull_bytes(
        name=name,
        tag=tag,
        parties=parties,
    )
    values = []
    for party, buffers in zip(parties, buffer_list):
        values.append(TableRemotePersistentUnpickler.pull(buffers, ctx, federation, name, tag, party))
    timer.done()
    return values
