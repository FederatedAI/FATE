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
import typing
from typing import Callable, Iterable, Any

from ...unify import URI
from .._profile import computing_profile
from .._type import ComputingEngine
from ..table import KVTable
from eggroll.roll_pair.roll_pair import RollPair

LOGGER = logging.getLogger(__name__)


class Table(KVTable):
    def destroy(self):
        self._rp.destroy()

    def _map_reduce_partitions_with_index(
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
        return Table(
            rp,
            key_serdes_type=output_key_serdes_type,
            value_serdes_type=output_value_serdes_type,
            partitioner_type=output_partitioner_type,
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
        return Table(
            rp,
            key_serdes_type=key_serdes_type,
            value_serdes_type=output_value_serdes_type,
            partitioner_type=partitioner_type,
        )

    def _take(self, n=1, **kwargs):
        return self._rp.take(n=n, **kwargs)

    def _count(self, **kwargs):
        return self._rp.count(**kwargs)

    def _collect(self):
        return self._rp.get_all()

    def _reduce(self, func: Callable[[bytes, bytes], bytes]):
        return self._rp.reduce(func=func)

    def __init__(self, rp: RollPair, key_serdes_type, value_serdes_type, partitioner_type):
        self._rp = rp
        self._engine = ComputingEngine.EGGROLL

        super().__init__(
            key_serdes_type=key_serdes_type,
            value_serdes_type=value_serdes_type,
            partitioner_type=partitioner_type,
            num_partitions=rp.get_partitions(),
        )

    @property
    def engine(self):
        return self._engine

    @property
    def partitions(self):
        return self._rp.get_partitions()

    @computing_profile
    def save(self, uri: URI, schema: dict, options: dict = None):
        if options is None:
            options = {}

        from ._type import EggRollStoreType

        if uri.scheme != "eggroll":
            raise ValueError(f"uri scheme {uri.scheme} not supported with eggroll backend")
        try:
            _, namespace, name = uri.path_splits()
        except Exception as e:
            raise ValueError(f"uri {uri} not supported with eggroll backend") from e

        if "store_type" not in options:
            options["store_type"] = EggRollStoreType.ROLLPAIR_LMDB

        partitions = options.get("partitions", self.partitions)
        self._rp.save_as(
            name=name,
            namespace=namespace,
            partition=partitions,
            options=options,
        )
        schema.update(self.schema)
        return

    @computing_profile
    def sample(
        self,
        *,
        fraction: typing.Optional[float] = None,
        num: typing.Optional[int] = None,
        seed=None,
    ):
        if fraction is not None:
            return Table(self._rp.sample(fraction=fraction, seed=seed))

        if num is not None:
            total = self._rp.count()
            if num > total:
                raise ValueError(f"not enough data to sample, own {total} but required {num}")

            frac = num / float(total)
            while True:
                sampled_table = self._rp.sample(fraction=frac, seed=seed)
                sampled_count = sampled_table.count()
                if sampled_count < num:
                    frac *= 1.1
                else:
                    break

            if sampled_count > num:
                drops = sampled_table.take(sampled_count - num)
                for k, v in drops:
                    sampled_table.delete(k)

            return Table(sampled_table)

        raise ValueError(f"exactly one of `fraction` or `num` required, fraction={fraction}, num={num}")
