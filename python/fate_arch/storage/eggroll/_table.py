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
from fate_arch.storage import StorageTableBase, StorageEngine, EggRollStoreType


class StorageTable(StorageTableBase):
    def __init__(self,
                 context,
                 name,
                 namespace,
                 address,
                 partitions: int = None,
                 store_type: EggRollStoreType = None,
                 options=None):
        super(StorageTable, self).__init__(name=name, namespace=namespace)
        self._context = context
        self._address = address
        self._partitions = partitions if partitions else 1
        self._store_type = store_type if store_type else EggRollStoreType.ROLLPAIR_LMDB
        self._options = options if options else {}
        self._engine = StorageEngine.EGGROLL

        if self._store_type:
            self._options["store_type"] = self._store_type
        self._options["total_partitions"] = partitions
        self._options["create_if_missing"] = True
        self._table = self._context.load(namespace=self._namespace, name=self._name, options=self._options)

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def get_address(self):
        return self._address

    def get_engine(self):
        return self._engine

    def get_store_type(self):
        return self._store_type

    def get_partitions(self):
        return self._table.get_partitions()

    def get_options(self):
        return self._options

    def put_all(self, kv_list: Iterable, **kwargs):
        super(StorageTable, self).update_write_access_time()
        return self._table.put_all(kv_list)

    def table(self):
        return self._table

    def union(self, other):
        return self._table.union(other.table(), func=lambda v1, v2 : v1)

    def save_as(self, dest_name, dest_namespace, partitions=None, schema=None):
        return self._table.save_as(name=dest_name, namespace=dest_namespace)

    def collect(self, **kwargs) -> list:
        super(StorageTable, self).update_read_access_time()
        return self._table.get_all(**kwargs)

    def destroy(self):
        super().destroy()
        return self._table.destroy()

    def count(self, **kwargs):
        count = self._table.count()
        self.meta.update_metas(count=count)
        return count
