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

from arch.api.impl.based_2x import build as build2x
from arch.api.impl.based_spark.based_2x.table import RDDTable
from arch.api.impl.based_spark.util import broadcast_eggroll_session


class Builder(build2x.Builder):
    _table_cls = RDDTable

    def __init__(self, session_id, work_mode, persistent_engine):
        super().__init__(session_id=session_id, work_mode=work_mode, persistent_engine=persistent_engine)

    def build_session(self):
        from arch.api.impl.based_2x.session import build_eggroll_session, build_eggroll_runtime
        from arch.api.impl.based_spark.based_2x.session import FateSessionImpl
        eggroll_session = build_eggroll_session(work_mode=self._work_mode, job_id=self._session_id)
        self._session_id = eggroll_session.get_session_id()
        broadcast_eggroll_session(work_mode=self._work_mode, eggroll_session=eggroll_session)
        eggroll_runtime = build_eggroll_runtime(eggroll_session=eggroll_session)
        return FateSessionImpl(self._session_id, eggroll_runtime, self._persistent_engine)
