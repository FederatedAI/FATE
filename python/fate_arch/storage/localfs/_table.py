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
import os
from typing import Iterable

from pyarrow import fs

from fate_arch.common import hdfs_utils
from fate_arch.common.log import getLogger
from fate_arch.storage import StorageEngine, LocalFSStoreType
from fate_arch.storage import StorageTableBase

LOGGER = getLogger()


class StorageTable(StorageTableBase):
    def __init__(
        self,
        address=None,
        name: str = None,
        namespace: str = None,
        partitions: int = 1,
        storage_type: LocalFSStoreType = LocalFSStoreType.DISK,
        options=None,
    ):
        super(StorageTable, self).__init__(
            name=name,
            namespace=namespace,
            address=address,
            partitions=partitions,
            options=options,
            engine=StorageEngine.LOCALFS,
            store_type=storage_type,
        )
        self._local_fs_client = fs.LocalFileSystem()

    @property
    def path(self):
        return self._address.path

    def _put_all(
        self, kv_list: Iterable, append=True, assume_file_exist=False, **kwargs
    ):
        LOGGER.info(f"put in file: {self.path}")

        # always create the directory first, otherwise the following creation of file will fail.
        self._local_fs_client.create_dir("/".join(self.path.split("/")[:-1]))

        if append and (assume_file_exist or self._exist()):
            stream = self._local_fs_client.open_append_stream(
                path=self.path, compression=None
            )
        else:
            stream = self._local_fs_client.open_output_stream(
                path=self.path, compression=None
            )

        counter = self._meta.get_count() if self._meta.get_count() else 0
        with io.TextIOWrapper(stream) as writer:
            for k, v in kv_list:
                writer.write(hdfs_utils.serialize(k, v))
                writer.write(hdfs_utils.NEWLINE)
                counter = counter + 1
        self._meta.update_metas(count=counter)

    def _collect(self, **kwargs) -> list:
        for line in self._as_generator():
            yield hdfs_utils.deserialize(line.rstrip())

    def _read(self) -> list:
        for line in self._as_generator():
            yield line

    def _destroy(self):
        # use try/catch to avoid stop while deleting an non-exist file
        try:
            self._local_fs_client.delete_file(self.path)
        except Exception as e:
            LOGGER.debug(e)

    def _count(self):
        count = 0
        for _ in self._as_generator():
            count += 1
        return count

    def _save_as(
        self, address, partitions=None, name=None, namespace=None, **kwargs
    ):
        self._local_fs_client.copy_file(src=self.path, dst=address.path)
        return StorageTable(
            address=address,
            partitions=partitions,
            name=name,
            namespace=namespace,
            **kwargs,
        )

    def close(self):
        pass

    def _exist(self):
        info = self._local_fs_client.get_file_info([self.path])[0]
        return info.type != fs.FileType.NotFound

    def _as_generator(self):
        info = self._local_fs_client.get_file_info([self.path])[0]
        if info.type == fs.FileType.NotFound:
            raise FileNotFoundError(f"file {self.path} not found")

        elif info.type == fs.FileType.File:
            for line in self._read_buffer_lines():
                yield line
        else:
            selector = fs.FileSelector(self.path)
            file_infos = self._local_fs_client.get_file_info(selector)
            for file_info in file_infos:
                if file_info.base_name.startswith(".") or file_info.base_name.startswith("_"):
                    continue
                assert (
                    file_info.is_file
                ), f"{self.path} is directory contains a subdirectory: {file_info.path}"
                with io.TextIOWrapper(
                    buffer=self._local_fs_client.open_input_stream(
                        f"{self._address.file_path:}/{file_info.path}"
                    ),
                    encoding="utf-8",
                ) as reader:
                    for line in reader:
                        yield line

    def _read_buffer_lines(self, path=None):
        if not path:
            path = self.path
        buffer = self._local_fs_client.open_input_file(self.path)
        offset = 0
        block_size = 1024 * 1024 * 10
        size = buffer.size()

        while offset < size:
            block_index = 1
            buffer_block = buffer.read_at(block_size, offset)
            if offset + block_size >= size:
                for line in self._read_lines(buffer_block):
                    yield line
                break
            if buffer_block.endswith(b"\n"):
                for line in self._read_lines(buffer_block):
                    yield line
                offset += block_size
                continue
            end_index = -1
            buffer_len = len(buffer_block)
            while not buffer_block[:end_index].endswith(b"\n"):
                if offset + block_index * block_size >= size:
                    break
                end_index -= 1
                if abs(end_index) == buffer_len:
                    block_index += 1
                    buffer_block = buffer.read_at(block_index * block_size, offset)
                    end_index = block_index * block_size
            for line in self._read_lines(buffer_block[:end_index]):
                yield line
            offset += len(buffer_block[:end_index])

    def _read_lines(self, buffer_block):
        with io.TextIOWrapper(buffer=io.BytesIO(buffer_block), encoding="utf-8") as reader:
            for line in reader:
                yield line
