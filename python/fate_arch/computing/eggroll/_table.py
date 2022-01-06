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


import typing

from fate_arch.abc import CTableABC
from fate_arch.common import log
from fate_arch.common.profile import computing_profile
from fate_arch.computing._type import ComputingEngine

LOGGER = log.getLogger()


class Table(CTableABC):

    def __init__(self, rp):
        self._rp = rp
        self._engine = ComputingEngine.EGGROLL

        self._count = None

    @property
    def engine(self):
        return self._engine

    @property
    def partitions(self):
        return self._rp.get_partitions()

    def copy(self):
        return Table(self._rp.map_values(lambda x: x))

    @computing_profile
    def save(self, address, partitions, schema: dict, **kwargs):
        options = kwargs.get("options", {})
        from fate_arch.common.address import EggRollAddress
        from fate_arch.storage import EggRollStoreType
        if isinstance(address, EggRollAddress):
            options["store_type"] = kwargs.get("store_type", EggRollStoreType.ROLLPAIR_LMDB)
            self._rp.save_as(name=address.name, namespace=address.namespace, partition=partitions, options=options)
            schema.update(self.schema)
            return

        from fate_arch.common.address import PathAddress
        if isinstance(address, PathAddress):
            from fate_arch.computing.non_distributed import LocalData
            return LocalData(address.path)

        raise NotImplementedError(f"address type {type(address)} not supported with eggroll backend")

    @computing_profile
    def collect(self, **kwargs) -> list:
        return self._rp.get_all()

    @computing_profile
    def count(self, **kwargs) -> int:
        if self._count is None:
            self._count = self._rp.count()
        return self._count

    @computing_profile
    def take(self, n=1, **kwargs):
        options = dict(keys_only=False)
        return self._rp.take(n=n, options=options)

    @computing_profile
    def first(self):
        options = dict(keys_only=False)
        return self._rp.first(options=options)

    @computing_profile
    def map(self, func, **kwargs):
        return Table(self._rp.map(func))

    @computing_profile
    def mapValues(self, func: typing.Callable[[typing.Any], typing.Any], **kwargs):
        return Table(self._rp.map_values(func))

    @computing_profile
    def applyPartitions(self, func):
        return Table(self._rp.collapse_partitions(func))

    @computing_profile
    def mapPartitions(self, func, use_previous_behavior=True, preserves_partitioning=False, **kwargs):
        if use_previous_behavior is True:
            LOGGER.warning(f"please use `applyPartitions` instead of `mapPartitions` "
                           f"if the previous behavior was expected. "
                           f"The previous behavior will not work in future")
            return self.applyPartitions(func)

        return Table(self._rp.map_partitions(func, options={"shuffle": not preserves_partitioning}))

    @computing_profile
    def mapReducePartitions(self, mapper, reducer, **kwargs):
        return Table(self._rp.map_partitions(func=mapper, reduce_op=reducer))

    @computing_profile
    def reduce(self, func, **kwargs):
        return self._rp.reduce(func)

    @computing_profile
    def join(self, other: 'Table', func, **kwargs):
        return Table(self._rp.join(other._rp, func=func))

    @computing_profile
    def glom(self, **kwargs):
        return Table(self._rp.glom())

    @computing_profile
    def sample(self, *, fraction: typing.Optional[float] = None, num: typing.Optional[int] = None, seed=None):
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

    @computing_profile
    def subtractByKey(self, other: 'Table', **kwargs):
        return Table(self._rp.subtract_by_key(other._rp))

    @computing_profile
    def filter(self, func, **kwargs):
        return Table(self._rp.filter(func))

    @computing_profile
    def union(self, other: 'Table', func=lambda v1, v2: v1, **kwargs):
        return Table(self._rp.union(other._rp, func=func))

    @computing_profile
    def flatMap(self, func, **kwargs):
        flat_map = self._rp.flat_map(func)
        shuffled = flat_map.map(lambda k, v: (k, v))  # trigger shuffle
        return Table(shuffled)
