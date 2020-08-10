#
#  Copyright 2019 The Eggroll Authors. All Rights Reserved.
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

from typing import MutableMapping

from arch.standalone import NamingPolicy, ComputingEngine
from arch.standalone.proto.basic_meta_pb2 import SessionInfo
from arch.standalone.utils.log_utils import getLogger
from arch.standalone.standalone.eggroll import Standalone

LOGGER = getLogger()


class EggrollSession(object):
    def __init__(self, session_id, chunk_size=100000, computing_engine_conf: MutableMapping = None,
                 naming_policy: NamingPolicy = NamingPolicy.DEFAULT, tag=None):
        if not computing_engine_conf:
            computing_engine_conf = dict()
        self._session_id = session_id
        self._chunk_size = chunk_size
        self._computing_engine_conf = computing_engine_conf
        self._naming_policy = naming_policy
        self._tag = tag
        self._cleanup_tasks = set()
        self._runtime = dict()
        self._gc_table = None

    def get_session_id(self):
        return self._session_id

    def get_chunk_size(self):
        return self._chunk_size

    def computing_engine_conf(self):
        return self.computing_engine_conf

    def add_conf(self, key, value):
        self._computing_engine_conf[key] = str(value)

    def get_conf(self, key):
        return self._computing_engine_conf.get(key)

    def has_conf(self, key):
        return self.get_conf(key) is not None

    def get_naming_policy(self):
        return self._naming_policy

    def get_tag(self):
        return self._tag

    def clean_duplicated_table(self, eggroll):
        for item in list(self._gc_table.collect()):
            name = item[0]
            if isinstance(eggroll, Standalone):
                eggroll.cleanup(name, self._session_id, False)
            else:
                table = eggroll.table(name=name, namespace=self._session_id, persistent=False)
                if not table.gc_enable:
                    eggroll.destroy(table)

    def add_cleanup_task(self, func):
        self._cleanup_tasks.add(func)

    def run_cleanup_tasks(self, eggroll):
        for func in self._cleanup_tasks:
            func(eggroll)

    def to_protobuf(self):
        return SessionInfo(sessionId=self._session_id,
                           computingEngineConf=self._computing_engine_conf,
                           namingPolicy=self._naming_policy.name,
                           tag=self._tag)

    @staticmethod
    def from_protobuf(session):
        return EggrollSession(session_id=session.get_session_id(),
                              computing_engine_conf=session.get_computing_engine_conf(),
                              naming_policy=session.get_naming_policy(),
                              tag=session.get_tag())

    def set_runtime(self, computing_engine: ComputingEngine, target):
        self._runtime[computing_engine] = target

    def set_gc_table(self, eggroll):
        self._gc_table = eggroll.table(name="__gc_" + self._session_id, namespace=self._session_id)

    def __str__(self):
        return "<EggrollSession: session_id: {}, computing_engine_conf: {}, naming_policy: {}, tag: {}, runtime: {}>" \
            .format(self._session_id, self.computing_engine_conf(), self._naming_policy.name, self._tag, self._runtime)
