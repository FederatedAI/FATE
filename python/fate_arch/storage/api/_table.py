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
from contextlib import closing

import requests
import os

from fate_arch.common.log import getLogger
from fate_arch.storage import StorageEngine, ApiStoreType
from fate_arch.storage import StorageTableBase

LOGGER = getLogger()


class StorageTable(StorageTableBase):
    def __init__(
        self,
        path,
        address=None,
        name: str = None,
        namespace: str = None,
        partitions: int = None,
        store_type: ApiStoreType = ApiStoreType.EXTERNAL,
        options=None,
    ):
        self.path = path
        self.data_count = 0
        super(StorageTable, self).__init__(
            name=name,
            namespace=namespace,
            address=address,
            partitions=partitions,
            options=options,
            engine=StorageEngine.API,
            store_type=store_type,
        )

    def _collect(self, **kwargs) -> list:
        self.request = getattr(requests, self.address.method.lower(), None)
        id_delimiter = self._meta.get_id_delimiter()
        with closing(self.request(url=self.address.url, json=self.address.body, headers=self.address.header,
                                  stream=True)) as response:
            if response.status_code == 200:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, 'wb') as fw:
                    for chunk in response.iter_content(1024):
                        if chunk:
                            fw.write(chunk)
                with open(self.path, "r") as f:
                    while True:
                        lines = f.readlines(1024 * 1024 * 1024)
                        if lines:
                            for line in lines:
                                self.data_count += 1
                                id = line.split(id_delimiter)[0]
                                feature = id_delimiter.join(line.split(id_delimiter)[1:])
                                yield id, feature
                            else:
                                _, self._meta = self._meta.update_metas(count=self.data_count)
                                break
            else:
                raise Exception(response.status_code, response.text)

    def _read(self) -> list:
        return []

    def _destroy(self):
        pass

    def _save_as(self, **kwargs):
        pass

    def _count(self):
        return self.data_count
