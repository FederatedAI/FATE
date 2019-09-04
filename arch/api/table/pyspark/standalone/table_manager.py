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

from arch.api.standalone.eggroll import Standalone
from arch.api.table.pyspark import materialize
from arch.api.table.pyspark.standalone import _to_serializable
from arch.api.table.pyspark.standalone.rddtable import RDDTable
from arch.api.table.table_manager import TableManager as TableManger


# noinspection PyProtectedMember
class RDDTableManager(TableManger):
    """
    manage RDDTable, use EggRoleStorage as storage
    """

    def __init__(self, job_id, eggroll_context):
        self._eggroll = Standalone(job_id=job_id, eggroll_context=eggroll_context)
        self._eggroll = _to_serializable(self._eggroll)

        # init PySpark
        sc = SparkContext.getOrCreate()
        self._sc = sc
        self.job_id = job_id

        # set eggroll info
        import pickle
        from arch.api.table.pyspark import _EGGROLL_CLIENT
        pickled_client = pickle.dumps(self._eggroll).hex()
        sc.setLocalProperty(_EGGROLL_CLIENT, pickled_client)
        TableManger.set_instance(self)

    def table(self,
              name,
              namespace,
              partition,
              persistent,
              in_place_computing,
              create_if_missing,
              error_if_exist):
        dtable = self._eggroll.table(name=name, namespace=namespace, partition=partition,
                                     persistent=persistent, in_place_computing=in_place_computing,
                                     create_if_missing=create_if_missing, error_if_exist=error_if_exist)
        return RDDTable.from_dtable(job_id=self.job_id, dtable=dtable)

    def parallelize(self,
                    data: Iterable,
                    include_key,
                    name,
                    partition,
                    namespace,
                    persistent,
                    chunk_size,
                    in_place_computing,
                    create_if_missing,
                    error_if_exist):
        _iter = data if include_key else enumerate(data)
        rdd = self._sc.parallelize(_iter, partition).partitionBy(partition)
        rdd = materialize(rdd)
        if namespace is None:
            namespace = self.job_id
        rdd_inst = RDDTable.from_rdd(rdd=rdd, job_id=self.job_id, namespace=namespace, name=name)

        return rdd_inst

    def cleanup(self, name, namespace, persistent):
        self._eggroll.cleanup(name=name, namespace=namespace, persistent=persistent)

    def generateUniqueId(self):
        self._eggroll.generateUniqueId()

    def get_job_id(self):
        return self.job_id
