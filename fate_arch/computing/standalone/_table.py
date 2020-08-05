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

from fate_arch.abc import AddressABC, CTableABC
from fate_arch.standalone import Table as RawTable


class Table(CTableABC):
    def __init__(self, table: RawTable):
        self._table = table

    def as_federation_format(self):
        return self._table

    @property
    def partitions(self):
        return self._table.partitions

    def save(self, address: AddressABC, partitions: int, schema: dict, **kwargs):
        from fate_arch.storage.address import EggRollAddress
        if isinstance(address, EggRollAddress):
            self._table.save_as(name=address.name, namespace=address.namespace, partition=partitions,
                                need_cleanup=False)
            schema.update(self.schema)
        raise NotImplementedError(f"address type {type(address)} not supported with standalone backend")

    def count(self) -> int:
        return self._table.count()

    def collect(self, **kwargs):
        return self._table.collect(**kwargs)

    def take(self, n=1, **kwargs):
        if n <= 0:
            raise ValueError(f"{n} <= 0")
        return list(itertools.islice(self.collect(), n))

    def first(self, **kwargs):
        resp = self.take(1, **kwargs)
        if len(resp) < 1:
            raise RuntimeError(f"table is empty")
        return resp[0]

    def reduce(self, func):
        return self._table.reduce(func)

    def map(self, func):
        return self._table.map(func)

    def mapValues(self, func):
        return self._table.mapValues(func)

    def flatMap(self, func):
        return self._table.flatMap(func)

    def mapPartitions(self, func):
        return self._table.mapPartitions(func)

    def glom(self):
        return self._table.glom()

    def sample(self, fraction, seed=None):
        return self._table.sample(fraction, seed)

    def filter(self, func):
        return self._table.filter(func)

    def join(self, other: 'Table', func):
        return self._table.join(other, func)

    def subtractByKey(self, other: 'Table'):
        return self._table.subtractByKey(other)

    def union(self, other: 'Table', func=lambda v1, v2: v1):
        return self._table.union(other, func)
