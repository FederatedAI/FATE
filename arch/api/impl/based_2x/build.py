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
from arch.api.base import build
from arch.api.base.utils.wrap import FederationWrapped
from arch.api.impl.based_2x.federation import FederationRuntime
from arch.api.impl.based_2x.session import build_session
from arch.api.impl.based_2x.table import DTable
from eggroll.roll_pair.roll_pair import RollPair


class Builder(build.Builder):
    _table_cls = DTable

    def __init__(self, session_id, work_mode, persistent_engine):
        self._session_id = session_id
        self._work_mode = work_mode
        self._persistent_engine = persistent_engine

    def build_session(self):
        return build_session(job_id=self._session_id, work_mode=self._work_mode,
                             persistent_engine=self._persistent_engine)

    def build_federation(self, federation_id, runtime_conf, server_conf_path):
        return FederationRuntime(session_id=federation_id, runtime_conf=runtime_conf)

    def build_wrapper(self):
        return FederationWrapped(session_id=self._session_id, dtable_cls=RollPair, table_cls=self._table_cls)
