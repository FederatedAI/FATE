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
from pyspark import StorageLevel


# noinspection PyUnresolvedReferences
def materialize(rdd):
    rdd = rdd.persist(get_storage_level())
    rdd.count()
    return rdd


# noinspection PyUnresolvedReferences
def get_storage_level():
    return StorageLevel.MEMORY_AND_DISK


def _get_file_system(sc):
    filesystem_class = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    hadoop_configuration = sc._jsc.hadoopConfiguration()
    return filesystem_class.get(hadoop_configuration)


def _get_path(sc, hdfs_path):
    path_class = sc._gateway.jvm.org.apache.hadoop.fs.Path
    return path_class(hdfs_path)
