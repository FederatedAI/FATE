#
#  Copyright 2019 The Eggroll Authors. All Rights Reserved.
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

import itertools
import typing

from fate_arch.abc import CTableABC
from fate_arch.common import log
from fate_arch.common.profile import computing_profile
from fate_arch.computing._type import ComputingEngine

LOGGER = log.getLogger()


class Table(CTableABC):
    def __init__(self, table):
        self._table = table
        self._engine = ComputingEngine.STANDALONE

        self._count = None

    @property
    def engine(self):
        return self._engine

    def __getstate__(self):
        pass

    @property
    def partitions(self):
        return self._table.partitions

    def copy(self):
        return Table(self._table.mapValues(lambda x: x))

    @computing_profile
    def save(self, address, partitions, schema, **kwargs):
        from fate_arch.common.address import StandaloneAddress

        if isinstance(address, StandaloneAddress):
            self._table.save_as(
                name=address.name,
                namespace=address.namespace,
                partition=partitions,
                need_cleanup=False,
            )
            schema.update(self.schema)
            return

        from fate_arch.common.address import PathAddress

        if isinstance(address, PathAddress):
            from fate_arch.computing.non_distributed import LocalData

            return LocalData(address.path)
        raise NotImplementedError(
            f"address type {type(address)} not supported with standalone backend"
        )

    @computing_profile
    def count(self) -> int:
        if self._count is None:
            self._count = self._table.count()
        return self._count

    @computing_profile
    def collect(self, **kwargs):
        return self._table.collect(**kwargs)

    @computing_profile
    def take(self, n=1, **kwargs):
        return self._table.take(n=n, **kwargs)

    @computing_profile
    def first(self, **kwargs):
        resp = list(itertools.islice(self._table.collect(**kwargs), 1))
        if len(resp) < 1:
            raise RuntimeError("table is empty")
        return resp[0]

    @computing_profile
    def reduce(self, func, **kwargs):
        return self._table.reduce(func)

    @computing_profile
    def map(self, func):
        return Table(self._table.map(func))

    @computing_profile
    def mapValues(self, func):
        return Table(self._table.mapValues(func))

    @computing_profile
    def flatMap(self, func):
        return Table(self._table.flatMap(func))

    @computing_profile
    def applyPartitions(self, func):
        return Table(self._table.applyPartitions(func))

    @computing_profile
    def mapPartitions(
        self, func, use_previous_behavior=True, preserves_partitioning=False
    ):
        if use_previous_behavior is True:
            LOGGER.warning(
                "please use `applyPartitions` instead of `mapPartitions` "
                "if the previous behavior was expected. "
                "The previous behavior will not work in future"
            )
            return Table(self._table.applyPartitions(func))
        return Table(
            self._table.mapPartitions(
                func, preserves_partitioning=preserves_partitioning
            )
        )

    @computing_profile
    def mapReducePartitions(self, mapper, reducer, **kwargs):
        return Table(self._table.mapReducePartitions(mapper, reducer))

    @computing_profile
    def glom(self):
        return Table(self._table.glom())

    @computing_profile
    def sample(
        self,
        *,
        fraction: typing.Optional[float] = None,
        num: typing.Optional[int] = None,
        seed=None,
    ):
        if fraction is not None:
            return Table(self._table.sample(fraction=fraction, seed=seed))

        if num is not None:
            total = self._table.count()
            if num > total:
                raise ValueError(
                    f"not enough data to sample, own {total} but required {num}"
                )

            frac = num / float(total)
            while True:
                sampled_table = self._table.sample(fraction=frac, seed=seed)
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

        raise ValueError(
            f"exactly one of `fraction` or `num` required, fraction={fraction}, num={num}"
        )

    @computing_profile
    def filter(self, func):
        return Table(self._table.filter(func))

    @computing_profile
    def join(self, other: "Table", func):
        return Table(self._table.join(other._table, func))

    @computing_profile
    def subtractByKey(self, other: "Table"):
        return Table(self._table.subtractByKey(other._table))

    @computing_profile
    def union(self, other: "Table", func=lambda v1, v2: v1):
        return Table(self._table.union(other._table, func))
