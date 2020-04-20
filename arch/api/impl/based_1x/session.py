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


# noinspection PyProtectedMember
from typing import Iterable

from arch.api import WorkMode
from arch.api.base.session import FateSession
from arch.api.base.utils.store_type import StoreTypes
from arch.api.impl.based_1x.table import DTable


# noinspection PyProtectedMember
def build_eggroll_runtime(work_mode: WorkMode, eggroll_session):
    if work_mode.is_standalone():
        from eggroll.api.standalone.eggroll import Standalone
        return Standalone(eggroll_session)

    elif work_mode.is_cluster():
        from eggroll.api.cluster.eggroll import eggroll_init, _EggRoll
        if _EggRoll.instance is None:
            return eggroll_init(eggroll_session)
    else:
        raise ValueError(f"work_mode: {work_mode} not supported!")


def build_eggroll_session(work_mode: WorkMode, job_id=None, server_conf_path="eggroll/conf/server_conf.json"):
    if work_mode.is_standalone():
        from eggroll.api.core import EggrollSession
        import uuid
        session_id = job_id or str(uuid.uuid1())
        session = EggrollSession(session_id=session_id)
        return session
    elif work_mode.is_cluster():
        from eggroll.api.cluster.eggroll import session_init
        return session_init(session_id=job_id, server_conf_path=server_conf_path)
    raise ValueError(f"work_mode: {work_mode} not supported!")


def build_session(job_id, work_mode: WorkMode, persistent_engine: str):
    eggroll_session = build_eggroll_session(work_mode=work_mode, job_id=job_id)
    session = FateSessionImpl(eggroll_session, work_mode, persistent_engine)
    return session


# noinspection PyProtectedMember
class FateSessionImpl(FateSession):
    """
    manage DTable, use EggRoleStorage as storage
    """

    def __init__(self, eggroll_session, work_mode, persistent_engine: str):
        self._eggroll = build_eggroll_runtime(work_mode=work_mode, eggroll_session=eggroll_session)
        self._session_id = eggroll_session.get_session_id()

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
        return DTable(dtable=dtable, session_id=self._session_id)

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
        dtable = self._eggroll.parallelize(data=data,
                                           include_key=include_key,
                                           name=name,
                                           partition=partition,
                                           namespace=namespace,
                                           persistent=persistent,
                                           chunk_size=chunk_size,
                                           in_place_computing=in_place_computing,
                                           create_if_missing=create_if_missing,
                                           error_if_exist=error_if_exist,
                                           persistent_engine=self._persistent_engine)

        rdd_inst = DTable(dtable, session_id=self._session_id)

        return rdd_inst

    def cleanup(self, name, namespace, persistent):
        self._eggroll.cleanup(name=name, namespace=namespace, persistent=persistent)

    def generateUniqueId(self):
        return self._eggroll.generateUniqueId()

    def get_session_id(self):
        return self._session_id

    def stop(self):
        self._eggroll.stop()

    def kill(self):
        self._eggroll.stop()
