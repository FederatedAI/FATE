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

from fate_arch.common import path_utils
from fate_arch.common.log import getLogger
from fate_arch.storage import StorageEngine, PathStoreType
from fate_arch.storage import StorageTableBase

LOGGER = getLogger()


class StorageTable(StorageTableBase):
    def __init__(
        self,
        address=None,
        name: str = None,
        namespace: str = None,
        partitions: int = None,
        store_type: PathStoreType = PathStoreType.PICTURE,
        options=None,
    ):
        super(StorageTable, self).__init__(
            name=name,
            namespace=namespace,
            address=address,
            partitions=partitions,
            options=options,
            engine=StorageEngine.PATH,
            store_type=store_type,
        )

    def _collect(self, **kwargs) -> list:
        return []

    def _read(self) -> list:
        return []

    def _destroy(self):
        pass

    def _save_as(self, **kwargs):
        pass

    def _count(self):
        return path_utils.get_data_table_count(self._address.path)
