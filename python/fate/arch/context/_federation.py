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
import struct
import typing
from typing import Any, List, Tuple, TypeVar, Union

from fate.arch.abc import FederationEngine, PartyMeta

from ..computing import is_table
from ..federation._gc import IterationGC
from ._namespace import NS

T = TypeVar("T")

if typing.TYPE_CHECKING:
    from fate.arch.context import Context


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
        self._party = party
        self._key = key

    def put(self, value):
        return self._party.put(self._key, value)

    def get(self):
        return self._party.get(self._key)


class Party:
    def __init__(self, ctx: "Context", federation, party: PartyMeta, rank: int, namespace: NS, key=None) -> None:
        self._ctx = ctx
        self._party = party
        self.federation = federation
        self.rank = rank
        self.namespace = namespace

    def __call__(self, key: str) -> "_KeyedParty":
        return _KeyedParty(self, key)

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
            return _push(self.federation, k, self.namespace, [self.party], v)

    def get(self, name: str):
        return _pull(self._ctx, self.federation, name, self.namespace, [self.party])[0]

    def get_int(self, name: str):
        ...


class Parties:
    def __init__(
        self,
        ctx: "Context",
        federation: FederationEngine,
        parties: List[Tuple[int, PartyMeta]],
        namespace: NS,
    ) -> None:
        self._ctx = ctx
        self.federation = federation
        self.parties = parties
        self.namespace = namespace

    @property
    def ranks(self):
        return [p[0] for p in self.parties]

    def __getitem__(self, key: int) -> Party:
        rank, party = self.parties[key]
        return Party(self._ctx, self.federation, party, rank, self.namespace)

    def __iter__(self):
        return iter([Party(self._ctx, self.federation, party, rank, self.namespace) for rank, party in self.parties])

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
        return _pull(self._ctx, self.federation, name, self.namespace, [p[1] for p in self.parties])


def _push(
    federation: FederationEngine,
    name: str,
    namespace: NS,
    parties: List[PartyMeta],
    value,
):
    tag = namespace.federation_tag
    _TableRemotePersistentPickler.push(value, federation, name, tag, parties)


class Serde:
    @classmethod
    def encode_int(cls, value: int) -> bytes:
        return struct.pack("!q", value)  # '!q' is for long long (8 bytes)

    @classmethod
    def decode_int(cls, value: bytes) -> int:
        return struct.unpack("!q", value)[0]

    @classmethod
    def encode_str(cls, value: str) -> bytes:
        utf8_str = value.encode("utf-8")
        return struct.pack("!I", len(utf8_str)) + utf8_str  # prepend length of string

    @classmethod
    def decode_str(cls, value: bytes) -> str:
        length = struct.unpack("!I", value[:4])[0]  # get length of string
        return value[4 : 4 + length].decode("utf-8")  # decode string

    @classmethod
    def encode_bytes(cls, value: bytes) -> bytes:
        return struct.pack("!I", len(value)) + value  # prepend length of bytes

    @classmethod
    def decode_bytes(cls, value: bytes) -> bytes:
        length = struct.unpack("!I", value[:4])[0]  # get length of bytes
        return value[4 : 4 + length]  # extract bytes

    @classmethod
    def encode_float(cls, value: float) -> bytes:
        return struct.pack("!d", value)

    @classmethod
    def decode_float(cls, value: bytes) -> float:
        return struct.unpack("!d", value)[0]


def _push_int(federation: FederationEngine, name: str, namespace: NS, parties: List[PartyMeta], value: int):
    tag = namespace.federation_tag
    federation.push(v=Serde.encode_int(value), name=name, tag=tag, parties=parties)


def _pull(
    ctx: "Context",
    federation: FederationEngine,
    name: str,
    namespace: NS,
    parties: List[PartyMeta],
):
    tag = namespace.federation_tag
    raw_values = federation.pull(
        name=name,
        tag=tag,
        parties=parties,
    )
    values = []
    for party, buffers in zip(parties, raw_values):
        values.append(_TableRemotePersistentUnpickler.pull(buffers, ctx, federation, name, tag, party))
    return values


class _TablePersistentId:
    def __init__(self, key) -> None:
        self.key = key


class _ContextPersistentId:
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
        from fate.arch.context import Context

        if is_table(obj):
            key = self._get_next_table_key()
            self._federation.push(v=obj, name=key, tag=self._tag, parties=self._parties)
            self._table_index += 1
            return _TablePersistentId(key)
        if isinstance(obj, Context):
            key = f"{self._name}__context__"
            return _ContextPersistentId(key)

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


class _TableRemotePersistentUnpickler(pickle.Unpickler):
    def __init__(
        self,
        ctx: "Context",
        federation: FederationEngine,
        name: str,
        tag: str,
        party: PartyMeta,
        f,
    ):
        self._ctx = ctx
        self._federation = federation
        self._name = name
        self._tag = tag
        self._party = party
        super().__init__(f)

    def persistent_load(self, pid: Any) -> Any:
        if isinstance(pid, _TablePersistentId):
            table = self._federation.pull(pid.key, self._tag, [self._party])[0]
            return table
        if isinstance(pid, _ContextPersistentId):
            return self._ctx

    @classmethod
    def pull(
        cls,
        buffers,
        ctx: "Context",
        federation: FederationEngine,
        name: str,
        tag: str,
        party: PartyMeta,
    ):
        with io.BytesIO(buffers) as f:
            unpickler = _TableRemotePersistentUnpickler(ctx, federation, name, tag, party, f)
            return unpickler.load()
