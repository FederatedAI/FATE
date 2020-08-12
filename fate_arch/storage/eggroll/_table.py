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
from fate_arch.common.profile import log_elapsed
from fate_arch.common import StorageEngine, EggRollStorageType
from fate_arch.storage import StorageTableBase


class StorageTable(StorageTableBase):
    def __init__(self,
                 context,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = 1,
                 storage_type: EggRollStorageType = None,
                 options=None):
        self._context = context
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions
        self._storage_type = storage_type
        self._options = options if options else {}
        self._storage_engine = StorageEngine.EGGROLL

        if self._storage_type:
            self._options["store_type"] = self._storage_type
        self._options["total_partitions"] = partitions
        self._table = self._context.load(namespace=self._namespace, name=self._name, options=self._options)

    def get_partitions(self):
        return self._table.get_partitions()

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def get_storage_engine(self):
        return self._storage_engine

    def get_address(self):
        return self._address

    def put_all(self, kv_list: Iterable, **kwargs):
        return self._table.put_all(kv_list)

    @log_elapsed
    def collect(self, **kwargs) -> list:
        return self._table.get_all(**kwargs)

    def destroy(self):
        super().destroy()
        return self._table.destroy()

    @log_elapsed
    def save_as(self, name=None, namespace=None, partition=None, schema=None, **kwargs):
        super().save_as(name, namespace, schema=schema, partition=partition)

        options = kwargs.get("options", {})
        store_type = options.get("store_type", StorageEngine.LMDB)
        options["store_type"] = store_type

        if partition is None:
            partition = self._partitions
        self._table.save_as(name=name, namespace=namespace, partition=partition, options=options).disable_gc()

    @log_elapsed
    def count(self, **kwargs):
        return self._table.count()

    def close(self):
        self.session.stop()
