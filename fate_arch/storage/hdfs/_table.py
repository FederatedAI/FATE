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

import pickle
from typing import Iterable

from pyspark import SparkContext

from fate_arch.common.log import getLogger
from fate_arch.storage import StorageEngine, HDFSStorageType
from fate_arch.storage import StorageTableBase

LOGGER = getLogger()


class StorageTable(StorageTableBase):
    def __init__(self,
                 context,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = 1,
                 storage_type: HDFSStorageType = None,
                 options=None):
        self._context = context
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions
        self._storage_type = storage_type
        self._options = options if options else {}
        self._storage_engine = StorageEngine.HDFS

    def get_partitions(self):
        return self._partitions

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def get_storage_engine(self):
        return self._storage_engine

    def get_address(self):
        return self.address

    def put_all(self, kv_list: Iterable, **kwargs):
        path, fs = StorageTable.get_hadoop_fs(address=self.address)
        if fs.exists(path):
            out = fs.append(path)
        else:
            out = fs.create_table(path)

        counter = 0
        for k, v in kv_list:
            content = u"{}{}{}\n".format(k, StorageTable.delimiter, pickle.dumps(v).hex())
            out.write(bytearray(content, "utf-8"))
            counter = counter + 1
        out.flush()
        out.close()
        self.update_metas(count=counter)

    def collect(self, **kwargs) -> list:
        sc = SparkContext.getOrCreate()
        hdfs_path = StorageTable.generate_hdfs_path(self.address)
        path = StorageTable.get_path(sc, hdfs_path)
        fs = StorageTable.get_file_system(sc)
        istream = fs.open(path)
        reader = sc._gateway.jvm.java.io.BufferedReader(sc._jvm.java.io.InputStreamReader(istream))
        while True:
            line = reader.readLine()
            if line is not None:
                fields = line.strip().partition(StorageTable.delimiter)
                yield fields[0], pickle.loads(bytes.fromhex(fields[2]))
            else:
                break
        istream.close()

    def destroy(self):
        super().destroy()
        path, fs = StorageTable.get_hadoop_fs(self.address)
        if fs.exists(path):
            fs.delete(path)

    def count(self):
        meta = self.get_meta(meta_type='count')
        if meta:
            return meta.f_count
        else:
            return -1

    def save_as(self, address, partitions=None, name=None, namespace=None, schema=None, **kwargs):
        super().save_as(name, namespace, partitions=partitions, schema=schema)
        sc = SparkContext.getOrCreate()
        src_path = StorageTable.get_path(sc, address.path)
        dst_path = StorageTable.get_path(sc, address.path)
        fs = StorageTable.get_file_system(sc)
        fs.rename(src_path, dst_path)
        return StorageTable(address=address, partitions=partitions, name=name, namespace=namespace, **kwargs)

    delimiter = '\t'

    @classmethod
    def generate_hdfs_path(cls, address):
        return address.path
    
    @classmethod
    def get_path(cls, sc, hdfs_path):
        path_class = sc._gateway.jvm.org.apache.hadoop.fs.Path
        return path_class(hdfs_path)

    @classmethod
    def get_file_system(cls, sc):
        filesystem_class = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
        hadoop_configuration = sc._jsc.hadoopConfiguration()
        return filesystem_class.get(hadoop_configuration)

    @classmethod
    def get_hadoop_fs(cls, address):
        sc = SparkContext.getOrCreate()
        hdfs_path = StorageTable.generate_hdfs_path(address)
        path = StorageTable.get_path(sc, hdfs_path)
        fs = StorageTable.get_file_system(sc)
        return path, fs

    def close(self):
        pass
