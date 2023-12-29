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
#


import logging
from typing import Callable, Iterable, Any

from eggroll.computing import RollPair
from fate.arch.computing.api import ComputingEngine, KVTable
from fate.arch.unify import URI

LOGGER = logging.getLogger(__name__)


class Table(KVTable):
    def __init__(self, rp: RollPair):
        self._rp = rp
        self._engine = ComputingEngine.EGGROLL

        super().__init__(
            key_serdes_type=self._rp.get_store().key_serdes_type,
            value_serdes_type=self._rp.get_store().value_serdes_type,
            partitioner_type=self._rp.get_store().partitioner_type,
            num_partitions=rp.get_partitions(),
        )

    @property
    def engine(self):
        return self._engine

    def _impl_map_reduce_partitions_with_index(
        self,
        map_partition_op: Callable[[int, Iterable], Iterable],
        reduce_partition_op: Callable[[Any, Any], Any],
        shuffle: bool,
        input_key_serdes,
        input_key_serdes_type: int,
        input_value_serdes,
        input_value_serdes_type: int,
        input_partitioner,
        input_partitioner_type: int,
        output_key_serdes,
        output_key_serdes_type: int,
        output_value_serdes,
        output_value_serdes_type: int,
        output_partitioner,
        output_partitioner_type: int,
        output_num_partitions: int,
    ):
        rp = self._rp.map_reduce_partitions_with_index(
            map_partition_op=map_partition_op,
            reduce_partition_op=reduce_partition_op,
            shuffle=shuffle,
            input_key_serdes=input_key_serdes,
            input_key_serdes_type=input_key_serdes_type,
            input_value_serdes=input_value_serdes,
            input_value_serdes_type=input_value_serdes_type,
            input_partitioner=input_partitioner,
            input_partitioner_type=input_partitioner_type,
            output_key_serdes=output_key_serdes,
            output_key_serdes_type=output_key_serdes_type,
            output_value_serdes=output_value_serdes,
            output_value_serdes_type=output_value_serdes_type,
            output_partitioner=output_partitioner,
            output_partitioner_type=output_partitioner_type,
            output_num_partitions=output_num_partitions,
        )
        return Table(rp)

    def _binary_sorted_map_partitions_with_index(
        self,
        other: "Table",
        binary_map_partitions_with_index_op: Callable[[int, Iterable, Iterable], Iterable],
        key_serdes,
        key_serdes_type,
        partitioner,
        partitioner_type,
        first_input_value_serdes,
        first_input_value_serdes_type,
        second_input_value_serdes,
        second_input_value_serdes_type,
        output_value_serdes,
        output_value_serdes_type,
    ):
        rp = self._rp.binary_sorted_map_partitions_with_index(
            other=other._rp,
            binary_map_partitions_with_index_op=binary_map_partitions_with_index_op,
            key_serdes=key_serdes,
            key_serdes_type=key_serdes_type,
            partitioner=partitioner,
            partitioner_type=partitioner_type,
            first_input_value_serdes=first_input_value_serdes,
            first_input_value_serdes_type=first_input_value_serdes_type,
            second_input_value_serdes=second_input_value_serdes,
            second_input_value_serdes_type=second_input_value_serdes_type,
            output_value_serdes=output_value_serdes,
            output_value_serdes_type=output_value_serdes_type,
        )
        return Table(rp)

    def _take(self, n=1, **kwargs):
        return self._rp.take(num=n, **kwargs)

    def _count(self, **kwargs):
        return self._rp.count(**kwargs)

    def _collect(self):
        return self._rp.get_all()

    def _reduce(self, func: Callable[[bytes, bytes], bytes]):
        return self._rp.reduce(func=func)

    def _save(self, uri: URI, schema: dict, options: dict):
        from ._type import EggRollStoreType

        if uri.scheme != "eggroll":
            raise ValueError(f"uri scheme {uri.scheme} not supported with eggroll backend")
        try:
            _, namespace, name = uri.path_splits()
        except Exception as e:
            raise ValueError(f"uri {uri} not supported with eggroll backend") from e

        store_type = options.get("store_type", EggRollStoreType.ROLLPAIR_LMDB)
        self._rp.copy_as(
            name=name,
            namespace=namespace,
            store_type=store_type,
        )

    def _drop_num(self, num: int, partitioner):
        for k, v in self._rp.take(num=num):
            self._rp.delete(k, partitioner=partitioner)
        return self

    def _destroy(self):
        self._rp.destroy()
