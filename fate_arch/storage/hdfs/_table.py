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
import io
from typing import Iterable

from pyarrow import fs

from fate_arch.common import hdfs_utils
from fate_arch.common.log import getLogger
from fate_arch.storage import StorageEngine, HDFSStorageType
from fate_arch.storage import StorageTableBase

LOGGER = getLogger()


class StorageTable(StorageTableBase):
    def __init__(self,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = None,
                 storage_type: HDFSStorageType = None,
                 options=None):
        super(StorageTable, self).__init__(name=name, namespace=namespace)
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions if partitions else 1
        self._type = storage_type if storage_type else HDFSStorageType.DISK
        self._options = options if options else {}
        self._engine = StorageEngine.HDFS

        # tricky way to load libhdfs
        try:
            from pyarrow import HadoopFileSystem
            HadoopFileSystem(self._path)
        except:
            pass
        self._hdfs_client = fs.HadoopFileSystem.from_uri(uri=self._path)

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

    def put_all(self, kv_list: Iterable, append=False, **kwargs):
        stream = self._hdfs_client.open_append_stream(path=self._path, compression=None) \
            if append else self._hdfs_client.open_output_stream(source=self._path, compression=None)

        counter = 0
        with io.TextIOWrapper(stream) as writer:
            for k, v in kv_list:
                writer.write(hdfs_utils.serialize(k, v))
                writer.write(hdfs_utils.NEWLINE)
                counter = counter + 1
        self._meta.update_metas(count=counter)

    def collect(self, **kwargs) -> list:
        with io.TextIOWrapper(self._hdfs_client.open_input_stream(self._path)) as reader:
            for line in reader:
                yield hdfs_utils.deserialize(line.rstrip())

    def read(self) -> list:
        with io.TextIOWrapper(self._hdfs_client.open_input_stream(self._path), encoding="utf-8") as reader:
            for line in reader:
                yield line

    def destroy(self):
        super().destroy()
        self._hdfs_client.delete_file(self._path)

    def count(self):
        count = 0
        with io.TextIOWrapper(self._hdfs_client.open_input_stream(self._path), encoding="utf-8") as reader:
            for _ in reader:
                count += 1

    def save_as(self, address, partitions=None, name=None, namespace=None, schema=None, **kwargs):
        super().save_as(name, namespace, partitions=partitions, schema=schema)
        self._hdfs_client.copy_file(src=self._path, dst=address.path)
        return StorageTable(address=address, partitions=partitions, name=name, namespace=namespace, **kwargs)

    def close(self):
        pass

    @property
    def _path(self) -> str:
        return self._address.path
