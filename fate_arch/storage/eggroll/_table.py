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
from fate_arch.storage import StorageTableBase, StorageEngine, EggRollStorageType


class StorageTable(StorageTableBase):
    def __init__(self,
                 context,
                 address,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = 1,
                 storage_type: EggRollStorageType = EggRollStorageType.ROLLPAIR_LMDB,
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
        self._table = self._context.load(namespace=self._namespace, name=self._name, options=self._options) if self._context else None

    def get_address(self):
        return self._address

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def get_storage_engine(self):
        return self._storage_engine

    def get_storage_type(self):
        return self._storage_type

    def get_partitions(self):
        return self._table.get_partitions()

    def get_options(self):
        return self._options

    def put_all(self, kv_list: Iterable, **kwargs):
        return self._table.put_all(kv_list)

    @log_elapsed
    def collect(self, **kwargs) -> list:
        return self._table.get_all(**kwargs)

    def destroy(self):
        super().destroy()
        return self._table.destroy()

    @log_elapsed
    def save_as(self, name=None, namespace=None, partitions=None, schema=None, **kwargs):
        super().save_as(name, namespace, schema=schema, partitions=partitions)

        options = kwargs.get("options", {})
        store_type = options.get("store_type", EggRollStorageType.ROLLPAIR_LMDB)
        options["store_type"] = store_type

        if partitions is None:
            partitions = self._partitions
        self._table.save_as(name=name, namespace=namespace, partition=partitions, options=options).disable_gc()

    @log_elapsed
    def count(self, **kwargs):
        return self._table.count()
