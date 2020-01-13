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

from arch.api import WorkMode
from arch.api.table import eggroll_util
from eggroll.api import StoreType
from arch.api.table.pyspark import materialize
from arch.api.table.pyspark.table_impl import RDDTable
from arch.api.table.session import FateSession


# noinspection PyProtectedMember
class FateSessionImpl(FateSession):
    """
    manage RDDTable, use EggRoleStorage as storage
    """

    def __init__(self, eggroll_session, work_mode: WorkMode, persistent_engine=StoreType.LMDB):
        self._session_id = eggroll_session.get_session_id()
        self._eggroll_session = eggroll_session
        self._persistent_engine = persistent_engine
        self._sc = self._build_spark_context()
        eggroll_util.broadcast_eggroll_session(self._sc, work_mode, eggroll_session)
        self._eggroll = eggroll_util.build_eggroll_runtime(work_mode, eggroll_session)
        FateSession.set_instance(self)

    def get_persistent_engine(self):
        return self._persistent_engine

    @staticmethod
    def _build_spark_context():
        from pyspark import SparkContext
        sc = SparkContext.getOrCreate()
        return sc

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
                                     create_if_missing=create_if_missing, error_if_exist=error_if_exist,
                                     persistent_engine=self._persistent_engine)
        return RDDTable.from_dtable(session_id=self._session_id, dtable=dtable)

    def parallelize(self,
                    data: Iterable,
                    name,
                    namespace,
                    partition,
                    include_key,
                    persistent,
                    chunk_size,
                    in_place_computing,
                    create_if_missing,
                    error_if_exist):
        _iter = data if include_key else enumerate(data)
        rdd = self._sc.parallelize(_iter, partition)
        rdd = materialize(rdd)
        if namespace is None:
            namespace = self._session_id
        return RDDTable.from_rdd(rdd=rdd, job_id=self._session_id, namespace=namespace, name=name)

    def cleanup(self, name, namespace, persistent):
        return self._eggroll.cleanup(name=name, namespace=namespace, persistent=persistent)

    def generateUniqueId(self):
        return self._eggroll.generateUniqueId()

    def get_session_id(self):
        return self._session_id

    def stop(self):
        self._eggroll.stop()
