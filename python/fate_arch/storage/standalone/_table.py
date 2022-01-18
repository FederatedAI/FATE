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

from fate_arch._standalone import Session
from fate_arch.storage import StorageEngine, StandaloneStoreType
from fate_arch.storage import StorageTableBase


class StorageTable(StorageTableBase):
    def __init__(
        self,
        session: Session,
        address=None,
        name: str = None,
        namespace: str = None,
        partitions: int = 1,
        store_type: StandaloneStoreType = StandaloneStoreType.ROLLPAIR_LMDB,
        options=None,
    ):
        super(StorageTable, self).__init__(
            name=name,
            namespace=namespace,
            address=address,
            partitions=partitions,
            options=options,
            engine=StorageEngine.STANDALONE,
            store_type=store_type,
        )
        self._session = session
        self._table = self._session.create_table(
            namespace=self._namespace,
            name=self._name,
            partitions=partitions,
            need_cleanup=self._store_type == StandaloneStoreType.ROLLPAIR_IN_MEMORY,
            error_if_exist=False,
        )

    def _put_all(self, kv_list: Iterable, **kwargs):
        return self._table.put_all(kv_list)

    def _collect(self, **kwargs):
        return self._table.collect(**kwargs)

    def _count(self):
        return self._table.count()

    def _destroy(self):
        return self._table.destroy()

    def _save_as(self, address, name, namespace, partitions=None, **kwargs):
        self._table.save_as(name=name, namespace=namespace)

        table = StorageTable(
            session=self._session,
            address=address,
            partitions=partitions,
            name=name,
            namespace=namespace,
            **kwargs,
        )
        return table
