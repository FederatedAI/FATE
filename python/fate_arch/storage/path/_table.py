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


from fate_arch.common.log import getLogger
from fate_arch.storage import StorageEngine, PathStorageType
from fate_arch.storage import StorageTableBase

LOGGER = getLogger()


class StorageTable(StorageTableBase):
    def __init__(self,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = None,
                 storage_type: PathStorageType = None,
                 options=None):
        super(StorageTable, self).__init__(name=name, namespace=namespace)
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions if partitions else 1
        self._type = storage_type if storage_type else PathStorageType.PICTURE
        self._options = options if options else {}
        self._engine = StorageEngine.PATH

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
        return self._partitions

    def get_options(self):
        return self._options

    def put_all(self, kv_list: Iterable, append=True, assume_file_exist=False, **kwargs):
        pass

    def collect(self, **kwargs) -> list:
        return []

    def read(self) -> list:
        return []

    def destroy(self):
        super().destroy()

    def count(self):
        return None

    def save_as(self, address, partitions=None, name=None, namespace=None, schema=None, **kwargs):
        return None