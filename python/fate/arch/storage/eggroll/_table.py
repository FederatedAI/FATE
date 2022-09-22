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
    def __init__(
        self,
        context,
        name,
        namespace,
        address,
        partitions: int = 1,
        store_type: EggRollStoreType = EggRollStoreType.ROLLPAIR_LMDB,
        options=None,
    ):
        super(StorageTable, self).__init__(
            name=name,
            namespace=namespace,
            address=address,
            partitions=partitions,
            options=options,
            engine=StorageEngine.EGGROLL,
            store_type=store_type,
        )
        self._context = context
        self._options["store_type"] = self._store_type
        self._options["total_partitions"] = partitions
        self._options["create_if_missing"] = True
        self._table = self._context.load(
            namespace=self._namespace, name=self._name, options=self._options
        )

    def _save_as(self, address, name, namespace, partitions=None, **kwargs):
        self._table.save_as(name=name, namespace=namespace)

        table = StorageTable(
            context=self._context,
            address=address,
            partitions=partitions,
            name=name,
            namespace=namespace
        )
        return table

    def _put_all(self, kv_list: Iterable, **kwargs):
        return self._table.put_all(kv_list)

    def _collect(self, **kwargs) -> list:
        return self._table.get_all(**kwargs)

    def _destroy(self):
        return self._table.destroy()

    def _count(self, **kwargs):
        return self._table.count()
