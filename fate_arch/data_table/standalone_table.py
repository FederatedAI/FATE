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
from typing import Iterable

from fate_arch.abc import TableABC
from fate_arch.backend.standalone import Table as RawTable
from fate_arch.data_table.address import EggRollAddress


class StandaloneTable(TableABC):

    def __init__(self, table: RawTable,
                 address, persistent_engine):
        self._table = table
        if not address:
            address = EggRollAddress(name=table.name, namespace=table.namespace, storage_type=persistent_engine)
        self._address = address
        self._storage_engine = persistent_engine

    def count(self):
        return self._table.count()

    def collect(self, **kwargs):
        return self._table.collect(**kwargs)

    def close(self):
        pass

    def save_as(self, name, namespace, partition=None, schema_data=None, **kwargs):
        return self._table.save_as(name=name, namespace=namespace, partition=partition, need_cleanup=False)

    def put_all(self, kv_list: Iterable, **kwargs):
        return self._table.put_all(kv_list)

    def get_address(self):
        return self._address

    def get_storage_engine(self):
        return self._storage_engine

    def get_partitions(self):
        return self._table.partitions

    def get_name(self):
        return self._table.name

    def get_namespace(self):
        return self._table.namespace
