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
import typing
import uuid
from typing import Iterable
from pyspark import SparkContext
import pickle

from arch.api import WorkMode
from arch.api.base.utils.store_type import StoreEngine
from arch.api.data_table.base import Table, HDFSStorage
from arch.api.utils.profile_util import log_elapsed
from fate_flow.db.db_models import DB, TableMeta
from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()



# noinspection SpellCheckingInspection,PyProtectedMember,PyPep8Naming
class HDFSTable(Table):
    def __init__(self,
                 namespace: str = None,
                 name: str = None,
                 partition: int = 1,
                 **kwargs):
        self._name = name or str(uuid.uuid1())
        self._namespace = namespace or str(uuid.uuid1())
        self._partitions = partition
    
    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def get_partitions(self):
        return self._partitions

    def get_storage_engine(self):
        return StoreEngine.HDFS

    def get_address(self):
        return HDFSStorage(HDFSTable.generate_hdfs_path(self._namespace, self._name))

    
    def put_all(self, kv_list: Iterable, use_serialize=True, chunk_size=100000):
        path, fs = HDFSTable.get_hadoop_fs(namespace=self._namespace, name=self._name)
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
        
        HDFSTable.update_table_meta(namespace=self._namespace, name=self._name, partitions=self._partitions, records=counter)

    
    def collect(self, min_chunk_size=0, use_serialize=True) -> list:
        sc = SparkContext.getOrCreate()
        hdfs_path = HDFSTable.generate_hdfs_path(namespace=self._namespace, name=self._sname)
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
        path, fs = HDFSTable.get_hadoop_fs(namespace=self._namespace, name=self._name)
        if(fs.exists(path)):
            fs.delete(path)
        HDFSTable.delete_table_meta(namespace=self._namespace, name=self._name)

    
    def count(self):
        meta = HDFSTable.get_table_meta(namespace=self._namespace, name=self._name)
        if meta:
            return meta.f_records
        else:
            return -1

    
    def save_as(self, name, namespace, partition=None, **kwargs):
        sc = SparkContext.getOrCreate()
        src_path = HDFSTable.get_path(sc, HDFSTable.generate_hdfs_path(namespace=self._namespace, name=self._name))
        dst_path = HDFSTable.get_path(sc, HDFSTable.generate_hdfs_path(namespace=namespace, name=name))
        fs = HDFSTable.get_file_system(sc)
        fs.rename(src_path, dst_path)
        records = self.count()
        HDFSTable.update_table_meta(namespace=namespace, name=name, partitions=partition, records=records)
        return HDFSTable(namespace, name, partition)


    delimiter = '\t'
    
    @classmethod
    def generate_hdfs_path(cls, namespace, name):
        return "/fate/{}/{}".format(namespace, name)
    
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
    def get_hadoop_fs(cls, namespace, name):
        sc = SparkContext.getOrCreate()
        hdfs_path = HDFSTable.generate_hdfs_path(namespace=namespace, name=name)
        path = HDFSTable.get_path(sc, hdfs_path)
        fs = HDFSTable.get_file_system(sc)
        return path, fs

    @classmethod
    def update_table_meta(cls, namespace, name, partitions, records):
        try:
            from arch.api.utils.core_utils import current_timestamp
            with DB.connection_context():
                metas = TableMeta.select().where(TableMeta.f_table_namespace == namespace,
                                                    TableMeta.f_table_name == name)
                is_insert = True
                if metas:
                    meta = metas[0]
                    is_insert = False
                else:
                    meta = TableMeta()
                    meta.f_table_namespace = namespace
                    meta.f_table_name = name
                    meta.f_create_time = current_timestamp()
                    meta.f_records = 0
                    
                meta.f_records = meta.f_records + records
                meta.f_partitions = partitions
                meta.f_update_time = current_timestamp()
                if is_insert:
                    meta.save(force_insert=True)
                else:
                    meta.save()
        except Exception as e:
            LOGGER.error("update_table_meta exception:{}.".format(e))

    @classmethod
    def get_table_meta(cls, namespace, name):
        try:
            with DB.connection_context():
                metas = TableMeta.select().where(TableMeta.f_table_namespace == namespace,
                                                    TableMeta.f_table_name == name)
                if metas:
                    return metas[0]
        except Exception as e:
            LOGGER.error("update_table_meta exception:{}.".format(e))

    @classmethod
    def delete_table_meta(cls, namespace, name):
        try:
            with DB.connection_context():
                TableMeta.delete().where(TableMeta.f_table_namespace == namespace,
                                         TableMeta.f_table_name == name).execute()
        except Exception as e:
            LOGGER.error("delete_table_meta {}, {}, exception:{}.".format(namespace, name, e))
