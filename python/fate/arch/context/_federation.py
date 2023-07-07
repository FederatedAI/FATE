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
import io
import pickle
from typing import Any, List, Tuple, TypeVar, Union

from fate.arch.abc import FederationEngine, PartyMeta

from ..computing import is_table
from ..federation._gc import IterationGC
from ._namespace import NS

T = TypeVar("T")


class GC:
    def __init__(self) -> None:
        self._push_gc_dict = {}
        self._pull_gc_dict = {}

    def get_or_set_push_gc(self, key):
        if key not in self._push_gc_dict:
            self._push_gc_dict[key] = IterationGC()
        return self._push_gc_dict[key]

    def get_or_set_pull_gc(self, key):
        if key not in self._pull_gc_dict:
            self._pull_gc_dict[key] = IterationGC()
        return self._pull_gc_dict[key]


class _KeyedParty:
    def __init__(self, party: Union["Party", "Parties"], key) -> None:
        self.party = party
        self.key = key

    def put(self, value):
        return self.party.put(self.key, value)

    def get(self):
        return self.party.get(self.key)


class Party:
    def __init__(self, federation, party: PartyMeta, rank: int, namespace: NS, key=None) -> None:
        self.federation = federation
        self.party = party
        self.rank = rank
        self.namespace = namespace
        self.key = key

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
            return _push(self.federation, k, self.namespace, [self.party], v)

    def get(self, name: str):
        return _pull(self.federation, name, self.namespace, [self.party])[0]


class Parties:
    def __init__(
        self,
        federation: FederationEngine,
        parties: List[Tuple[int, PartyMeta]],
        namespace: NS,
    ) -> None:
        self.federation = federation
        self.parties = parties
        self.namespace = namespace

    @property
    def ranks(self):
        return [p[0] for p in self.parties]

    def __getitem__(self, key: int) -> Party:
        rank, party = self.parties[key]
        return Party(self.federation, party, rank, self.namespace)

    def __iter__(self):
        return iter([Party(self.federation, party, rank, self.namespace) for rank, party in self.parties])

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
            return _push(self.federation, k, self.namespace, [p[1] for p in self.parties], v)

    def get(self, name: str):
        return _pull(self.federation, name, self.namespace, [p[1] for p in self.parties])


def _push(
    federation: FederationEngine,
    name: str,
    namespace: NS,
    parties: List[PartyMeta],
    value,
):
    tag = namespace.get_federation_tag()
    _TableRemotePersistentPickler.push(value, federation, name, tag, parties)


def _pull(
    federation: FederationEngine,
    name: str,
    namespace: NS,
    parties: List[PartyMeta],
):
    tag = namespace.get_federation_tag()
    raw_values = federation.pull(
        name=name,
        tag=tag,
        parties=parties,
    )
    values = []
    for party, buffers in zip(parties, raw_values):
        values.append(_TableRmotePersistentUnpickler.pull(buffers, federation, name, tag, party))
    return values


class _TablePersistentId:
    def __init__(self, key) -> None:
        self.key = key


class _TableRemotePersistentPickler(pickle.Pickler):
    def __init__(
        self,
        federation: FederationEngine,
        name: str,
        tag: str,
        parties: List[PartyMeta],
        f,
    ) -> None:
        self._federation = federation
        self._name = name
        self._tag = tag
        self._parties = parties

        self._tables = {}
        self._table_index = 0
        super().__init__(f)

    def _get_next_table_key(self):
        # or uuid?
        return f"{self._name}__table_persistent_{self._table_index}__"

    def persistent_id(self, obj: Any) -> Any:
        if is_table(obj):
            key = self._get_next_table_key()
            self._federation.push(v=obj, name=key, tag=self._tag, parties=self._parties)
            self._table_index += 1
            return _TablePersistentId(key)

    @classmethod
    def push(
        cls,
        value,
        federation: FederationEngine,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ):
        with io.BytesIO() as f:
            pickler = _TableRemotePersistentPickler(federation, name, tag, parties, f)
            pickler.dump(value)
            federation.push(v=f.getvalue(), name=name, tag=tag, parties=parties)


class _TableRmotePersistentUnpickler(pickle.Unpickler):
    def __init__(
        self,
        federation: FederationEngine,
        name: str,
        tag: str,
        party: PartyMeta,
        f,
    ):
        self._federation = federation
        self._name = name
        self._tag = tag
        self._party = party
        super().__init__(f)

    def persistent_load(self, pid: Any) -> Any:
        if isinstance(pid, _TablePersistentId):
            table = self._federation.pull(pid.key, self._tag, [self._party])[0]
            return table

    @classmethod
    def pull(
        cls,
        buffers,
        federation: FederationEngine,
        name: str,
        tag: str,
        party: PartyMeta,
    ):
        with io.BytesIO(buffers) as f:
            unpickler = _TableRmotePersistentUnpickler(federation, name, tag, party, f)
            return unpickler.load()
