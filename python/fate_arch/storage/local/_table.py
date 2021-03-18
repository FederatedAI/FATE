########################################################
# Copyright 2020-2021 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

import io
import os
from typing import Iterable

from pyarrow import fs

from fate_arch.common import hdfs_utils
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
        self._engine = StorageEngine.LOCAL

        self._local_fs_client = fs.LocalFileSystem()

    def _get_data_dir():
        return _data_dir

    def _get_storage_dir(*args):
        return _data_dir.joinpath(*args)

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

    @property
    def _path(self):
        return self._address.path

    def put_all(self, kv_list: Iterable, append=True, assume_file_exist=False, **kwargs):
        LOGGER.info(f"put in file: {self._path}")

        # always create the directory first, otherwise the following creation of file will fail.
        self._local_fs_client. create_dir('/'.join(self._path.split('/')[:-1]))

        if append and (assume_file_exist or self._exist()):
            stream = self._local_fs_client.open_append_stream(
                path=self._path, compression=None)
        else:
            stream = self._local_fs_client.open_output_stream(
                path=self._path, compression=None)

        counter = 0
        with io.TextIOWrapper(stream) as writer:
            for k, v in kv_list:
                writer.write(hdfs_utils.serialize(k, v))
                writer.write(hdfs_utils.NEWLINE)
                counter = counter + 1
        self._meta.update_metas(count=counter)

    def collect(self, **kwargs) -> list:
        for line in self._as_generator():
            yield hdfs_utils.deserialize(line.rstrip())

    def read(self) -> list:
        for line in self._as_generator():
            yield line

    def destroy(self):
        super().destroy()

        # use try/catch to avoid stop while deleting an non-exist file
        try:
            self._local_fs_client.delete_file(self._path)
        except Exception as e:
            LOGGER.debug(e)

    def count(self):
        count = 0
        for _ in self._as_generator():
            count += 1
        self.get_meta().update_metas(count=count)
        return count

    def save_as(self, address, partitions=None, name=None, namespace=None, schema=None, **kwargs):
        super().save_as(name, namespace, partitions=partitions, schema=schema)
        self._local_fs_client.copy_file(src=self._path, dst=address.path)
        return StorageTable(address=address, partitions=partitions, name=name, namespace=namespace, **kwargs)

    def close(self):
        pass

    def _exist(self):
        info = self._local_fs_client.get_file_info([self._path])[0]
        return info.type != fs.FileType.NotFound

    def _as_generator(self):
        info = self._local_fs_client.get_file_info([self._path])[0]
        if info.type == fs.FileType.NotFound:
            raise FileNotFoundError(f"file {self._path} not found")

        elif info.type == fs.FileType.File:
            with io.TextIOWrapper(buffer=self._local_fs_client.open_input_stream(self._path),
                                  encoding="utf-8") as reader:
                for line in reader:
                    yield line

        else:
            selector = fs.FileSelector(self._path)
            file_infos = self._local_fs_client.get_file_info(selector)
            for file_info in file_infos:
                if file_info.base_name == "_SUCCESS":
                    continue
                assert file_info.is_file, f"{self._path} is directory contains a subdirectory: {file_info.path}"
                with io.TextIOWrapper(
                        buffer=self._local_fs_client.open_input_stream(
                            f"{self._address.file_path:}/{file_info.path}"),
                        encoding="utf-8") as reader:
                    for line in reader:
                        yield line
