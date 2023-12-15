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
import logging
import pickle
import struct
import typing
from typing import Any, List, Tuple, TypeVar, Union

from fate.arch.federation.api import PartyMeta, TableMeta
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
    _TableRemotePersistentPickler.push(
        value,
        federation,
        computing,
        name,
        tag,
        parties,
        max_message_size=max_message_size,
        num_partitions_of_slice_table=num_partitions_of_slice_table,
    )


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


def _pull(
    ctx: "Context",
    federation: "Federation",
    name: str,
    namespace: NS,
    parties: List[PartyMeta],
):
    tag = namespace.federation_tag
    buffer_list = federation.pull_bytes(
        name=name,
        tag=tag,
        parties=parties,
    )
    values = []
    for party, buffers in zip(parties, buffer_list):
        values.append(_TableRemotePersistentUnpickler.pull(buffers, ctx, federation, name, tag, party))
    return values


class _TablePersistentId:
    def __init__(self, key, table_meta: "TableMeta") -> None:
        self.key = key
        self.table_meta = table_meta


class _ContextPersistentId:
    def __init__(self, key) -> None:
        self.key = key


class _FederationBytesCoder:
    @staticmethod
    def encode_base(v: bytes) -> bytes:
        return struct.pack("!B", 0) + v

    @staticmethod
    def encode_split(slice_table_meta: "TableMeta", total_size: int, num_slice: int, slice_size: int) -> bytes:
        return struct.pack("!B", 1) + struct.pack(
            "!IIIIIII",
            total_size,
            num_slice,
            slice_size,
            slice_table_meta.num_partitions,
            slice_table_meta.key_serdes_type,
            slice_table_meta.value_serdes_type,
            slice_table_meta.partitioner_type,
        )

    @classmethod
    def decode_mode(cls, v: bytes) -> int:
        return struct.unpack("!B", v[:1])[0]

    @classmethod
    def decode_base(cls, v: bytes) -> bytes:
        return v[1:]

    @classmethod
    def decode_split(cls, v: bytes) -> Tuple["TableMeta", int, int, int]:
        (
            total_size,
            num_slice,
            slice_size,
            num_partitions,
            key_serdes_type,
            value_serdes_type,
            partitioner_type,
        ) = struct.unpack("!IIIIIII", v[1:29])
        table_meta = TableMeta(
            num_partitions=num_partitions,
            key_serdes_type=key_serdes_type,
            value_serdes_type=value_serdes_type,
            partitioner_type=partitioner_type,
        )
        return table_meta, total_size, num_slice, slice_size


class _SplitTableUtil:
    @staticmethod
    def get_split_table_key(name):
        return f"{name}__table_persistent_split__"


class _TableRemotePersistentPickler(pickle.Pickler):
    def __init__(
        self,
        federation: "Federation",
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
        from fate.arch.computing.api import KVTable

        if isinstance(obj, KVTable):
            key = self._get_next_table_key()
            self._push_table(obj, key)
            return _TablePersistentId(
                key=key,
                table_meta=TableMeta(
                    num_partitions=obj.num_partitions,
                    key_serdes_type=obj.key_serdes_type,
                    value_serdes_type=obj.value_serdes_type,
                    partitioner_type=obj.partitioner_type,
                ),
            )
        if isinstance(obj, Context):
            key = f"{self._name}__context__"
            return _ContextPersistentId(key)

    def _push_table(self, table, key):
        self._federation.push_table(table=table, name=key, tag=self._tag, parties=self._parties)
        self._table_index += 1
        return key

    @classmethod
    def push(
        cls,
        value,
        federation: "Federation",
        computing: "KVTableContext",
        name: str,
        tag: str,
        parties: List[PartyMeta],
        max_message_size: int,
        num_partitions_of_slice_table: int,
    ):
        with io.BytesIO() as f:
            pickler = _TableRemotePersistentPickler(federation, name, tag, parties, f)
            pickler.dump(value)
            if f.tell() > max_message_size:
                total_size = f.tell()
                num_slice = (total_size - 1) // max_message_size + 1
                # create a table to store the slice
                f.seek(0)
                data = [(i, f.read(max_message_size)) for i in range(num_slice)]
                slice_table = computing.parallelize(
                    data,
                    partition=num_partitions_of_slice_table,
                    key_serdes_type=0,
                    value_serdes_type=0,
                    partitioner_type=0,
                )
                split_table_meta = TableMeta(
                    num_partitions=num_partitions_of_slice_table,
                    key_serdes_type=0,
                    value_serdes_type=0,
                    partitioner_type=0,
                )
                # push the slice table with a special key
                federation.push_table(slice_table, _SplitTableUtil.get_split_table_key(name), tag=tag, parties=parties)
                # push the slice table info
                federation.push_bytes(
                    v=_FederationBytesCoder.encode_split(split_table_meta, total_size, num_slice, max_message_size),
                    name=name,
                    tag=tag,
                    parties=parties,
                )

            else:
                federation.push_bytes(
                    v=_FederationBytesCoder.encode_base(f.getvalue()), name=name, tag=tag, parties=parties
                )


class _TableRemotePersistentUnpickler(pickle.Unpickler):
    def __init__(
        self,
        ctx: "Context",
        federation: "Federation",
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
            table = self._federation.pull_table(pid.key, self._tag, [self._party], table_metas=[pid.table_meta])[0]
            return table
        if isinstance(pid, _ContextPersistentId):
            return self._ctx

    @classmethod
    def pull(
        cls,
        buffers,
        ctx: "Context",
        federation: "Federation",
        name: str,
        tag: str,
        party: PartyMeta,
    ):
        mode = _FederationBytesCoder.decode_mode(buffers)
        if mode == 0:
            with io.BytesIO(_FederationBytesCoder.decode_base(buffers)) as f:
                unpickler = _TableRemotePersistentUnpickler(ctx, federation, name, tag, party, f)
                return unpickler.load()
        elif mode == 1:
            # get num_slice and slice_size
            table_meta, total_size, num_slice, slice_size = _FederationBytesCoder.decode_split(buffers)

            # pull the slice table with a special key
            slice_table = federation.pull_table(
                name=_SplitTableUtil.get_split_table_key(name), tag=tag, parties=[party], table_metas=[table_meta]
            )[0]
            # merge the bytes
            with io.BytesIO() as f:
                for i, b in slice_table.collect():
                    f.seek(i * slice_size)
                    f.write(b)
                f.seek(0)
                unpickler = _TableRemotePersistentUnpickler(ctx, federation, name, tag, party, f)
                return unpickler.load()
        else:
            raise ValueError(f"invalid mode: {mode}")
