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
from fate_arch.storage import StorageEngine, StandaloneStorageType
from fate_arch.storage import StorageTableBase


class StorageTable(StorageTableBase):
    def __init__(self,
                 context,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = 1,
                 storage_type: StandaloneStorageType = None,
                 options=None):
        self._context = context
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions
        self._storage_type = storage_type
        self._options = options if options else {}
        self._storage_engine = StorageEngine.STANDALONE

    def count(self):
        return self._table.count()

    def collect(self, **kwargs):
        return self._table.collect(**kwargs)

    def close(self):
        return self._session.stop()

    def save_as(self, name, namespace, partitions=None, schema=None, **kwargs):
        return self._table.save_as(name=name, namespace=namespace, partitions=partitions, need_cleanup=False)

    def put_all(self, kv_list: Iterable, **kwargs):
        return self._table.put_all(kv_list)

    def get_address(self):
        return self._address

    def get_engine(self):
        return self._storage_engine

    def get_partitions(self):
        return self._table.partitions

    def get_name(self):
        return self._table.name

    def get_namespace(self):
        return self._table.namespace
