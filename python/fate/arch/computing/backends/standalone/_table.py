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
from typing import Callable, Iterable, Any, Tuple

from fate.arch.computing.api import ComputingEngine, KVTable, K, V
from fate.arch.unify import URI
from ._standalone import Table as StandaloneTable

LOGGER = logging.getLogger(__name__)


class Table(KVTable):
    def __init__(self, table: StandaloneTable):
        self._table = table
        self._engine = ComputingEngine.STANDALONE
        super().__init__(
            key_serdes_type=table.key_serdes_type,
            value_serdes_type=table.value_serdes_type,
            partitioner_type=table.partitioner_type,
            num_partitions=table.partitions,
        )

    @property
    def table(self):
        return self._table

    @property
    def engine(self):
        return self._engine

    def _destroy(self):
        pass

    def _drop_num(self, num: int, partitioner):
        for k, v in self._table.take(num=num):
            self._table.delete(k, partitioner=partitioner)
        return Table(table=self._table)

    def _impl_map_reduce_partitions_with_index(
        self,
        map_partition_op: Callable[[int, Iterable[Tuple[K, V]]], Iterable],
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
        return Table(
            table=self._table.map_reduce_partitions_with_index(
                map_partition_op=map_partition_op,
                reduce_partition_op=reduce_partition_op,
                shuffle=shuffle,
                output_key_serdes_type=output_key_serdes_type,
                output_value_serdes_type=output_value_serdes_type,
                output_partitioner=output_partitioner,
                output_partitioner_type=output_partitioner_type,
                output_num_partitions=output_num_partitions,
            ),
        )

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
        return Table(
            table=self._table.binary_sorted_map_partitions_with_index(
                other=other._table,
                binary_map_partitions_with_index_op=binary_map_partitions_with_index_op,
                key_serdes_type=key_serdes_type,
                partitioner_type=partitioner_type,
                output_value_serdes_type=output_value_serdes_type,
            ),
        )

    def _collect(self, **kwargs):
        return self._table.collect(**kwargs)

    def _take(self, num=1, **kwargs):
        return self._table.take(num=num, **kwargs)

    def _count(self):
        return self._table.count()

    def _reduce(self, func, **kwargs):
        return self._table.reduce(func)

    def _save(self, uri: URI, schema, options: dict):
        if uri.scheme != "standalone":
            raise ValueError(f"uri scheme `{uri.scheme}` not supported with standalone backend")
        try:
            *database, namespace, name = uri.path_splits()
        except Exception as e:
            raise ValueError(f"uri `{uri}` not supported with standalone backend") from e
        self._table.copy_as(
            name=name,
            namespace=namespace,
            need_cleanup=False,
        )
