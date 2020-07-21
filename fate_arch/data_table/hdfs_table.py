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
from pyspark import SparkContext
import pickle

from fate_arch.data_table.base import Table, HDFSAddress
from fate_arch.data_table.store_type import StoreEngine
from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()


# noinspection SpellCheckingInspection,PyProtectedMember,PyPep8Naming
class HDFSTable(Table):
    def __init__(self,
                 address=None,
                 partitions: int = 1,
                 **kwargs):
        self.address = address
        self._partitions = partitions

    def get_partitions(self):
        return self._partitions

    def get_storage_engine(self):
        return StoreEngine.HDFS

    def get_address(self):
        return self.address

    def put_all(self, kv_list: Iterable, use_serialize=True, chunk_size=100000):
        path, fs = HDFSTable.get_hadoop_fs(address=self.address)
        if(fs.exists(path)):
            out = fs.append(path)
        else:
            out = fs.create(path)

        counter = 0
        for k, v in kv_list:
            content = u"{}{}{}\n".format(k, HDFSTable.delimiter, pickle.dumps((v)).hex())
            out.write(bytearray(content, "utf-8"))
            counter = counter + 1
        out.flush()
        out.close()
        self.save_schema(count=counter)

    def collect(self, min_chunk_size=0, use_serialize=True) -> list:
        sc = SparkContext.getOrCreate()
        hdfs_path = HDFSTable.generate_hdfs_path(self.address)
        path = HDFSTable.get_path(sc, hdfs_path)
        fs = HDFSTable.get_file_system(sc)
        istream = fs.open(path)
        reader = sc._gateway.jvm.java.io.BufferedReader(sc._jvm.java.io.InputStreamReader(istream))
        while True:
            line = reader.readLine()
            if line is not None:
                fields = line.strip().partition(HDFSTable.delimiter)
                yield fields[0], pickle.loads(bytes.fromhex(fields[2]))
            else:
                break
        istream.close()

    def destroy(self):
        super().destroy()
        path, fs = HDFSTable.get_hadoop_fs(self.address)
        if(fs.exists(path)):
            fs.delete(path)
    
    def count(self):
        meta = self.get_schema(_type='count')
        if meta:
            return meta.f_count
        else:
            return -1

    def save_as(self, address, partition=None, **kwargs):
        sc = SparkContext.getOrCreate()
        src_path = HDFSTable.get_path(sc, HDFSTable.generate_hdfs_path(address))
        dst_path = HDFSTable.get_path(sc, HDFSTable.generate_hdfs_path(address))
        fs = HDFSTable.get_file_system(sc)
        fs.rename(src_path, dst_path)
        return HDFSTable(address=address, partitions=partition)

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
        hdfs_path = HDFSTable.generate_hdfs_path(address)
        path = HDFSTable.get_path(sc, hdfs_path)
        fs = HDFSTable.get_file_system(sc)
        return path, fs