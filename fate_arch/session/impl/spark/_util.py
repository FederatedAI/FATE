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


RDD_ATTR_NAME = "_rdd"


# noinspection PyUnresolvedReferences
def get_storage_level():
    from pyspark import StorageLevel
    return StorageLevel.MEMORY_AND_DISK


def materialize(rdd):
    rdd = rdd.persist(get_storage_level())
    rdd.count()
    return rdd


def get_partitions(namespace, name) -> int:
    pass


def _generate_hdfs_path(namespace, name):
    return "/fate/{}/{}".format(namespace, name)


def _get_file_system(sc):
    filesystem_class = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hadoop_configuration = sc._jsc.hadoopConfiguration()
    return filesystem_class.get(hadoop_configuration)


def _get_path(sc, hdfs_path):
    path_class = sc._gateway.jvm.org.apache.hadoop.fs.Path
    return path_class(hdfs_path)

 #
 # def write2hdfs(cls, namespace, name, kv_list: Iterable, create_if_missing: bool = True):
 #        hdfs_path = _generate_hdfs_path(namespace=namespace, name=name)
 #        path = _get_path(sc, hdfs_path)
 #        fs = _get_file_system(sc)
 #        if fs.exists(path):
 #            out = fs.append(path)
 #        elif create_if_missing:
 #            out = fs.create(path)
 #        else:
 #            raise AssertionError("hdfs path {} not exists.".format(hdfs_path))
 #
 #        counter = 0
 #        for k, v in kv_list:
 #            content = u"{}{}{}\n".format(k, cls.delimiter, pickle.dumps((v)).hex())
 #            out.write(bytearray(content, "utf-8"))
 #            counter = counter + 1
 #        out.flush()
 #        out.close()
 #
 #        cls.update_table_meta(namespace=namespace, name=name, records=counter)
