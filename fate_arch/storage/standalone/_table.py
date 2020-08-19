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

from fate_arch.standalone import Session
from fate_arch.storage import StorageEngine, StandaloneStorageType
from fate_arch.storage import StorageTableBase


class StorageTable(StorageTableBase):
    def __init__(self,
                 session: Session,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = 1,
                 storage_type: StandaloneStorageType = None,
                 options=None):
        super(StorageTable, self).__init__(name=name, namespace=namespace)
        self._session = session
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions
        self._type = storage_type if storage_type else StandaloneStorageType.ROLLPAIR_LMDB
        self._options = options if options else {}
        self._storage_engine = StorageEngine.STANDALONE
        need_cleanup = self._type == StandaloneStorageType.ROLLPAIR_IN_MEMORY
        self._table = self._session.create_table(namespace=self._namespace, name=self._name, partitions=partitions,
                                                 need_cleanup=need_cleanup, error_if_exist=False)

    def get_name(self):
        return self._table.name

    def get_namespace(self):
        return self._table.namespace

    def get_address(self):
        return self._address

    def get_engine(self):
        return self._storage_engine

    def get_type(self):
        return self._type

    def get_partitions(self):
        return self._table.partitions

    def get_options(self):
        return self._options

    def put_all(self, kv_list: Iterable, **kwargs):
        return self._table.put_all(kv_list)

    def collect(self, **kwargs):
        return self._table.collect(**kwargs)

    def destroy(self):
        super().destroy()
        return self._table.destroy()

    def count(self):
        count = self._table.count()
        self.get_meta().update_metas(count=count)
        return count
