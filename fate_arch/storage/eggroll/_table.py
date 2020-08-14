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
                 name,
                 namespace,
                 address,
                 partitions: int = 1,
                 storage_type: EggRollStorageType = EggRollStorageType.ROLLPAIR_LMDB,
                 options=None):
        super(StorageTable, self).__init__(name=name, namespace=namespace)
        self._context = context
        self._address = address
        self._partitions = partitions
        self._type = storage_type
        self._options = options if options else {}
        self._engine = StorageEngine.EGGROLL

        if self._type:
            self._options["store_type"] = self._type
        self._options["total_partitions"] = partitions
        self._table = self._context.load(namespace=self._namespace, name=self._name, options=self._options)

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def get_address(self):
        return self._address

    def get_engine(self):
        return self._engine

    def get_type(self):
        return self._type

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
    def count(self, **kwargs):
        return self._table.count()
