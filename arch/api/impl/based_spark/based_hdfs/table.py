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

import uuid
from typing import Iterable

# noinspection PyPackageRequirements

from arch.api.base.table import Table
from arch.api.impl.based_spark import util
from arch.api.impl.utils.split import split_put, split_get
from arch.api.utils.profile_util import log_elapsed


class RDDTable(Table):

    # noinspection PyProtectedMember
    @classmethod
    def from_hdfs(cls, session_id: str,
                  namespace: str,
                  name: str = None,
                  partitions: int = 1,
                  create_if_missing: bool = True):
        rdd = rdd_from_hdfs(namespace=namespace, name=name, create_if_missing=create_if_missing)
        return RDDTable(session_id=session_id, namespace=namespace, name=name, partitions=partitions, rdd=rdd)


    @classmethod
    def from_rdd(cls, rdd, job_id: str, namespace: str, name: str):
        partitions = rdd.getNumPartitions()
        return RDDTable(session_id=job_id, namespace=namespace, name=name, partitions=partitions, rdd=rdd)


    @classmethod
    def generate_hdfs_path(cls, namspace, name):
        return "/fate/{}/{}".format(namspace, name)
    
    
    @classmethod
    def get_path(cls, sc, hdfs_path):
        path_class = sc._gateway.jvm.org.apache.hadoop.fs.path
        return path_class(hdfs_path)
    

    @classmethod
    def get_file_system(cls, sc):
        filesystem_class = sc._gateway.jvm.org.apache.hadoop.fs.FilFileSystem
        hadoop_configuration = sc._jsc.hadoopConfiguration()
        return filesystem_class.get(hadoop_configuration)


    @classmethod
    def write2hdfs(cls, sc, hdfs_path, content, create_if_missing: bool = True):
        fs = get_file_system(sc)
        path = get_path(sc, hdfs_path)
        if(fs.exists(path)):
            out = fs.append(path)
        elif create_if_missing:
            out = fs.create(path)
        else:
            raise AssertionError("hdfs path {} not exists.".format(hdfs_path))
        out.write(bytearray(content, "utf-8"))
        out.flush()
        out.close()

    
    @classmethod
    def rdd_from_hdfs(cls, namespace, name, create_if_missing: bool = True):
        from pyspark import SparkContext
        sc = SparkContext.getOrCreate()
        hdfs_path = generate_hdfs_path(namespace=namespace, name=name)
        fs = get_file_system(sc)
        path = get_path(sc, hdfs_path)
        if(fs.exists(path)):
            rdd = sc.textFile(path)
        elif create_if_missing:
            rdd = sc.emptyRDD()
        else:
            raise AssertionError("hdfs path {} not exists.".format(hdfs_path))
        return rdd


    def __init__(self, session_id: str,
                 namespace: str,
                 name: str = None,
                 partitions: int = 1,
                 rdd=None):

        self._valid_param_check(rdd, namespace, partitions)
        setattr(self, util.RDD_ATTR_NAME, rdd)
        self._rdd = rdd
        self._partitions = partitions
        self.schema = {}
        self._name = name or str(uuid.uuid1())
        self._namespace = namespace
        self._session_id = session_id

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def __str__(self):
        return f"{self._namespace}, {self._name}"

    def __repr__(self):
        return f"{self._namespace}, {self._name}"

    def _tmp_table_from_rdd(self, rdd, name=None):
        """
        tmp table, with namespace == job_id
        """
        rdd = util.materialize(rdd)
        name = name or str(uuid.uuid1())
        return RDDTable(session_id=self._session_id,
                        namespace=self._namespace,
                        name=name,
                        partitions=rdd.getNumPartitions(),
                        rdd=rdd)

    # self._rdd should not be pickled(spark requires all transformer/action to be invoked in driver).
    def __getstate__(self):
        state = dict(self.__dict__)
        if "_rdd" in state:
            del state["_rdd"]
        return state

    @staticmethod
    def _valid_param_check(rdd, namespace, partitions):
        assert (rdd is not None), "params rdd is None"
        assert namespace is not None, "namespace is None"
        assert partitions > 0, "invalid partitions={0}".format(partitions)

    def rdd(self):
        if hasattr(self, "_rdd") and self._rdd is not None:
            return self._rdd
        else:
            return _rdd_from_hdfs()


    # noinspection PyProtectedMember,PyUnresolvedReferences
    @log_elapsed
    def _rdd_from_hdfs(self):
        self._rdd = rdd_from_hdfs(namespace=self._namespace, name=self._name, create_if_missing=False)
        return self._rdd


    def get_partitions(self):
        return self._partitions

    @log_elapsed
    def map(self, func, **kwargs):
        from arch.api.impl.based_spark.rdd_func import _map
        rtn_rdd = _map(self.rdd(), func)
        return self._tmp_table_from_rdd(rtn_rdd)

    @log_elapsed
    def mapValues(self, func, **kwargs):
        from arch.api.impl.based_spark.rdd_func import _map_value
        rtn_rdd = _map_value(self.rdd(), func)
        return self._tmp_table_from_rdd(rtn_rdd)

    @log_elapsed
    def mapPartitions(self, func, **kwargs):
        from arch.api.impl.based_spark.rdd_func import _map_partitions
        rtn_rdd = _map_partitions(self.rdd(), func)
        return self._tmp_table_from_rdd(rtn_rdd)

    @log_elapsed
    def mapPartitions2(self, func, **kwargs):
        return self.mapPartitions(func, **kwargs)

    @log_elapsed
    def reduce(self, func, key_func=None, **kwargs):
        if key_func is None:
            return self.rdd().values().reduce(func)

        return dict(self.rdd().map(lambda x: (key_func(x[0]), x[1])).reduceByKey(func).collect())

    def join(self, other, func=None, **kwargs):
        rdd1 = self.rdd()
        rdd2 = other.rdd()

        # noinspection PyUnusedLocal,PyShadowingNames
        @log_elapsed
        def _join(rdda, rddb, **kwargs):
            from arch.api.impl.based_spark.rdd_func import _join
            return self._tmp_table_from_rdd(_join(rdda, rddb, func))

        return _join(rdd1, rdd2, **kwargs)

    @log_elapsed
    def glom(self, **kwargs):
        from arch.api.impl.based_spark.rdd_func import _glom
        return self._tmp_table_from_rdd(_glom(self.rdd()))

    @log_elapsed
    def sample(self, fraction, seed=None, **kwargs):
        from arch.api.impl.based_spark.rdd_func import _sample
        return self._tmp_table_from_rdd(_sample(self.rdd(), fraction, seed))

    @log_elapsed
    def subtractByKey(self, other, **kwargs):
        from arch.api.impl.based_spark.rdd_func import _subtract_by_key
        return self._tmp_table_from_rdd(_subtract_by_key(self.rdd(), other.rdd()))

    @log_elapsed
    def filter(self, func, **kwargs):
        from arch.api.impl.based_spark.rdd_func import _filter
        return self._tmp_table_from_rdd(_filter(self.rdd(), func))

    @log_elapsed
    def union(self, other, func=lambda v1, v2: v1, **kwargs):
        from arch.api.impl.based_spark.rdd_func import _union
        return self._tmp_table_from_rdd(_union(self.rdd(), other.rdd(), func))

    @log_elapsed
    def flatMap(self, func, **kwargs):
        from arch.api.impl.based_spark.rdd_func import _flat_map
        return self._tmp_table_from_rdd(_flat_map(self.rdd(), func))

    @log_elapsed
    def collect(self, min_chunk_size=0, use_serialize=True, **kwargs):
        return iter(self.rdd().collect())

    """
    storage api
    """

    def put(self, k, v, use_serialize=True, maybe_large_value=False):
        self._rdd = None

    def put_all(self, kv_list: Iterable, use_serialize=True, chunk_size=100000):
        self._rdd = None

    def get(self, k, use_serialize=True, maybe_large_value=False):
        return self.rdd().lookup(k)

    def delete(self, k, use_serialize=True):
        self._rdd = None

    def destroy(self):
        self._rdd = None

    def put_if_absent(self, k, v, use_serialize=True):
        self.put(k=k, v=v, use_serialize=use_serialize)

    # noinspection PyPep8Naming
    def take(self, n=1, keysOnly=False, use_serialize=True):
        rtn = self._rdd.take(n)
        if keysOnly:
            rtn = [pair[0] for pair in rtn]
        return rtn

    # noinspection PyPep8Naming
    def first(self, keysOnly=False, use_serialize=True):
        return self.take(1, keysOnly, use_serialize)[0]

    def count(self, **kwargs):
        return self._rdd.count()

    @log_elapsed
    def save_as(self, name, namespace, partition=None, use_serialize=True, persistent=True, **kwargs):
        if partition is None:
            partition = self._partitions
        if self._dtable:
            from arch.api import RuntimeInstance
            persistent_engine = RuntimeInstance.SESSION.get_persistent_engine()
            options = dict(store_type=persistent_engine)
            saved_table = self._dtable.save_as(name=name, namespace=namespace, partition=partition, options=options)
            return RDDTable.from_dtable(session_id=self._session_id, dtable=saved_table)
        else:
            it = self._rdd.toLocalIterator()
            from arch.api import session
            rdd_table = session.table(name=name, namespace=namespace, partition=partition, persistent=persistent)
            rdd_table.put_all(kv_list=it)
            return rdd_table
