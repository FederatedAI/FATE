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
from typing import Callable, Iterable, Any, Tuple

from ...unify import URI
from .._profile import computing_profile
from .._type import ComputingEngine
from ..table import KVTable, V, K
from ..._standalone import Table as StandaloneTable

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
    def engine(self):
        return self._engine

    def __getstate__(self):
        pass

    def __reduce__(self):
        raise NotImplementedError("Table is not picklable, please don't do this or it may cause unexpected error")

    def _map_reduce_partitions_with_index(
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
                output_partitioner=output_partitioner,
                shuffle=shuffle,
                output_key_serdes_type=output_key_serdes_type,
                output_value_serdes_type=output_value_serdes_type,
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

    def _take(self, n=1, **kwargs):
        return self._table.take(n=n, **kwargs)

    def _count(self):
        return self._table.count()

    def _reduce(self, func, **kwargs):
        return self._table.reduce(func)

    @property
    def partitions(self):
        return self._table.partitions

    @computing_profile
    def save(self, uri: URI, schema, options: dict = None):
        if options is None:
            options = {}

        if uri.scheme != "standalone":
            raise ValueError(f"uri scheme `{uri.scheme}` not supported with standalone backend")
        try:
            *database, namespace, name = uri.path_splits()
        except Exception as e:
            raise ValueError(f"uri `{uri}` not supported with standalone backend") from e
        self._table.save_as(
            name=name,
            namespace=namespace,
            partitions=options.get("partitions", self.partitions),
            need_cleanup=False,
        )
        # TODO: self.schema is a bit confusing here, it set by property assignment directly, not by constructor
        schema.update(self.schema)

    @computing_profile
    def sample(
        self,
        *,
        fraction: typing.Optional[float] = None,
        num: typing.Optional[int] = None,
        seed=None,
    ):
        if fraction is not None:
            return Table(self._sample(fraction=fraction, seed=seed))

        if num is not None:
            total = self._table.count()
            if num > total:
                raise ValueError(f"not enough data to sample, own {total} but required {num}")

            frac = num / float(total)
            while True:
                sampled_table = self._sample(fraction=frac, seed=seed)
                sampled_count = sampled_table.count()
                if sampled_count < num:
                    frac += 0.1
                else:
                    break

            if sampled_count > num:
                drops = sampled_table.take(sampled_count - num)
                for k, v in drops:
                    sampled_table.delete(k)

            return Table(sampled_table)

        raise ValueError(f"exactly one of `fraction` or `num` required, fraction={fraction}, num={num}")
