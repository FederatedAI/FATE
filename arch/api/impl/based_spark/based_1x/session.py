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

from arch.api.base.utils.store_type import StoreTypes

from arch.api.base.session import FateSession
from arch.api.impl.based_spark import util
from arch.api.impl.based_spark.based_1x.table import RDDTable

__all__ = ["FateSessionImpl"]


class FateSessionImpl(FateSession):
    """
    manage RDDTable, use EggRoleStorage as storage
    """

    def __init__(self, session_id, eggroll_runtime, persistent_engine):
        self._session_id = session_id
        self._persistent_engine = persistent_engine
        self._eggroll = eggroll_runtime

        # convert to StoreType class in eggroll v1.x
        from eggroll.api import StoreType as StoreTypeV1
        if persistent_engine == StoreTypes.ROLLPAIR_LMDB:
            self._persistent_engine = StoreTypeV1.LMDB
        elif persistent_engine == StoreTypes.ROLLPAIR_LEVELDB:
            self._persistent_engine = StoreTypeV1.LEVEL_DB
        elif persistent_engine == StoreTypes.ROLLPAIR_IN_MEMORY:
            self._persistent_engine = StoreTypeV1.IN_MEMORY
        else:
            raise ValueError(f"{persistent_engine} not supported, use one of {[e.value for e in StoreTypeV1]}")

        FateSession.set_instance(self)

    def get_persistent_engine(self):
        return self._persistent_engine

    def table(self,
              name,
              namespace,
              partition,
              persistent,
              in_place_computing,
              create_if_missing,
              error_if_exist,
              **kwargs):
        dtable = self._eggroll.table(name=name, namespace=namespace, partition=partition,
                                     persistent=persistent, in_place_computing=in_place_computing,
                                     create_if_missing=create_if_missing, error_if_exist=error_if_exist,
                                     persistent_engine=self._persistent_engine)
        return RDDTable.from_dtable(session_id=self._session_id, dtable=dtable)

    # noinspection PyUnresolvedReferences
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
        from pyspark import SparkContext
        rdd = SparkContext.getOrCreate().parallelize(_iter, partition)
        rdd = util.materialize(rdd)
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

    def kill(self):
        self._eggroll.stop()
