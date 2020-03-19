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
from arch.api.impl.based_2x.table import DTable
from eggroll.core.constants import SerdesTypes
from eggroll.roll_pair.roll_pair import RollPairContext


def build_eggroll_session(job_id, work_mode: WorkMode):
    from eggroll.core.session import session_init
    options = {}
    if work_mode == WorkMode.STANDALONE:
        options['eggroll.session.deploy.mode'] = "standalone"
    elif work_mode == WorkMode.CLUSTER:
        options['eggroll.session.deploy.mode'] = "cluster"

    return session_init(session_id=job_id, options=options)


def build_eggroll_runtime(eggroll_session):
    rpc = RollPairContext(session=eggroll_session)
    return rpc


def build_session(job_id, work_mode: WorkMode, persistent_engine: str):
    eggroll_session = build_eggroll_session(work_mode=work_mode, job_id=job_id)
    session = FateSessionImpl(eggroll_session, persistent_engine)
    return session


class FateSessionImpl(FateSession):
    """
    manage DTable, use EggRoleStorage as storage
    """

    def __init__(self, eggroll_session, persistent_engine: str):
        self._eggroll = build_eggroll_runtime(eggroll_session=eggroll_session)
        self._session_id = eggroll_session.get_session_id()
        self._persistent_engine = persistent_engine
        self._session = eggroll_session
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
        options = kwargs.get("option", {})
        if "use_serialize" in kwargs and not kwargs["use_serialize"]:
            options["serdes"] = SerdesTypes.EMPTY
        options.update(dict(create_if_missing=create_if_missing, total_partitions=partition))
        dtable = self._eggroll.load(namespace=namespace, name=name, options=options)
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
        # dtable = self._eggroll.parallelize(data=data,
        #                                    include_key=include_key,
        #                                    name=name,
        #                                    partition=partition,
        #                                    namespace=namespace,
        #                                    persistent=persistent,
        #                                    chunk_size=chunk_size,
        #                                    in_place_computing=in_place_computing,
        #                                    create_if_missing=create_if_missing,
        #                                    error_if_exist=error_if_exist)
        options = dict()
        options["name"] = name
        options["namespace"] = namespace
        options["create_if_missing"] = create_if_missing
        options["total_partitions"] = partition

        options["include_key"] = include_key
        dtable = self._eggroll.parallelize(data=data, options=options)
        rdd_inst = DTable(dtable, session_id=self._session_id)

        return rdd_inst

    def cleanup(self, name, namespace, persistent):
        self._eggroll.cleanup(name=name, namespace=namespace)

    def generateUniqueId(self):
        return self._eggroll.generateUniqueId()

    def get_session_id(self):
        return self._session_id

    def stop(self):
        return self._session.stop()

    def kill(self):
        return self._session.kill()
