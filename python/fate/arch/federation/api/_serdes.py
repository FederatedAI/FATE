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
from typing import Any, List, Tuple, TypeVar

from fate.arch.config import cfg
from ._table_meta import TableMeta
from ._type import PartyMeta

logger = logging.getLogger(__name__)
T = TypeVar("T")

if typing.TYPE_CHECKING:
    from fate.arch.context import Context
    from fate.arch.federation.api import Federation
    from fate.arch.computing.api import KVTableContext, KVTable
    import torch
    import numpy as np


class _TablePersistentId:
    def __init__(self, key, table_meta: "TableMeta") -> None:
        self.key = key
        self.table_meta = table_meta

    @staticmethod
    def dump(pickler: "TableRemotePersistentPickler", obj: "KVTable") -> Any:
        from fate.arch.computing.api import KVTable

        assert isinstance(obj, KVTable)
        key = pickler._get_next_table_key()
        pickler._push_table(obj, key)
        return _TablePersistentId(
            key=key,
            table_meta=TableMeta(
                num_partitions=obj.num_partitions,
                key_serdes_type=obj.key_serdes_type,
                value_serdes_type=obj.value_serdes_type,
                partitioner_type=obj.partitioner_type,
            ),
        )

    def load(self, unpickler: "TableRemotePersistentUnpickler"):
        table = unpickler._federation.pull_table(
            self.key, unpickler._tag, [unpickler._party], table_metas=[self.table_meta]
        )[0]
        return table


class _ContextPersistentId:
    def __init__(self, key) -> None:
        self.key = key

    @staticmethod
    def dump(pickler: "TableRemotePersistentPickler", obj: "Context") -> Any:
        from fate.arch.context import Context

        assert isinstance(obj, Context)
        key = f"{pickler._name}__context__"
        return _ContextPersistentId(key)

    def load(self, unpickler: "TableRemotePersistentUnpickler"):
        return unpickler._ctx


class _TorchSafeTensorPersistentId:
    def __init__(self, bytes) -> None:
        self.bytes = bytes

    @staticmethod
    def dump(_pickler: "TableRemotePersistentPickler", obj: "torch.Tensor") -> Any:
        import torch
        import safetensors.torch

        assert isinstance(obj, torch.Tensor)
        tensor_bytes = safetensors.torch.save({"t": obj})
        return _TorchSafeTensorPersistentId(tensor_bytes)

    def load(self, _unpickler: "TableRemotePersistentUnpickler"):
        import safetensors.torch

        return safetensors.torch.load(self.bytes)["t"]


class _NumpySafeTensorPersistentId:
    def __init__(self, bytes) -> None:
        self.bytes = bytes

    @staticmethod
    def dump(_pickler: "TableRemotePersistentPickler", obj: "np.ndarray") -> Any:
        import numpy as np
        import safetensors.numpy

        assert isinstance(obj, np.ndarray)
        if obj.dtype != np.dtype("object"):
            tensor_bytes = safetensors.numpy.save({"n": obj})
            return _NumpySafeTensorPersistentId(tensor_bytes)

    def load(self, _unpickler: "TableRemotePersistentUnpickler"):
        import safetensors.numpy

        return safetensors.numpy.load(self.bytes)["n"]


class _FederationBytesCoder:
    @staticmethod
    def encode_base(v: bytes) -> bytes:
        return struct.pack("!B", 0) + v

    @staticmethod
    def encode_split(slice_table_meta: "TableMeta", total_size: int, num_slice: int, slice_size: int) -> bytes:
        return struct.pack("!B", 1) + struct.pack(
            "!QIIIIII",
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
        ) = struct.unpack("!QIIIIII", v[1:33])
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


class TableRemotePersistentPickler(pickle.Pickler):
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
        # TODO: use serdes method check instead of isinstance
        from fate.arch.context import Context
        from fate.arch.computing.api import KVTable
        import torch
        import numpy as np

        if isinstance(obj, KVTable):
            return _TablePersistentId.dump(self, obj)
        if isinstance(obj, Context):
            return _ContextPersistentId.dump(self, obj)

        if isinstance(obj, torch.Tensor):
            return _TorchSafeTensorPersistentId.dump(self, obj)

        if isinstance(obj, np.ndarray) and obj.dtype != np.dtype("object"):
            return _NumpySafeTensorPersistentId.dump(self, obj)

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
            pickler = TableRemotePersistentPickler(federation, name, tag, parties, f)
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


class TableRemotePersistentUnpickler(pickle.Unpickler):
    __ALLOW_CLASSES = {
        "torch": {"device", "Size", "int64", "int32", "float64", "float32", "dtype"},
        "collections": {"OrderedDict"},
        # we can remove following after we customize the serdes for `DataFrame`
        "builtins": {"slice"},
        "pandas.core.series": {"Series"},
        "pandas.core.internals.managers": {"SingleBlockManager"},
        "pandas.core.indexes.base": {"_new_Index", "Index"},
        "numpy.core.multiarray": {"_reconstruct"},
        "numpy": {"ndarray", "dtype"},
    }
    __BUILDIN_MODULES = {"fate.", "fate_utils."}
    __ALLOW_MODULES = {}

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
        if isinstance(
            pid,
            (
                _TablePersistentId,
                _ContextPersistentId,
                _TorchSafeTensorPersistentId,
                _NumpySafeTensorPersistentId,
            ),
        ):
            return pid.load(self)

    def find_class(self, module, name):
        if cfg.safety.serdes.federation.restricted_type == "unrestricted":
            return super().find_class(module, name)
        for m in self.__BUILDIN_MODULES:
            if module.startswith(m):
                return super().find_class(module, name)
        if module in self.__ALLOW_MODULES:
            return super().find_class(module, name)
        elif module in self.__ALLOW_CLASSES and name in self.__ALLOW_CLASSES[module]:
            return super().find_class(module, name)
        else:
            if cfg.safety.serdes.federation.restricted_type == "restricted_catch_miss":
                self.__ALLOW_CLASSES.setdefault(module, set()).add(name)
                path_to_write = f"{cfg.safety.serdes.federation.restricted_catch_miss_path}_{self._federation.local_party[0]}_{self._federation.local_party[1]}"
                with open(path_to_write, "a") as f:
                    f.write(f"{module}.{name}\n")
                return super().find_class(module, name)
            elif cfg.safety.serdes.federation.restricted_type == "restricted":
                raise ValueError(
                    f"Deserialization is restricted for class `{module}`.`{name}`, allowlist: {self.__ALLOW_CLASSES}"
                )
            else:
                raise ValueError(f"invalid restricted_type: {cfg.safety.serdes.federation.restricted_type}")

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
                unpickler = TableRemotePersistentUnpickler(ctx, federation, name, tag, party, f)
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
                unpickler = TableRemotePersistentUnpickler(ctx, federation, name, tag, party, f)
                return unpickler.load()
        else:
            raise ValueError(f"invalid mode: {mode}")
