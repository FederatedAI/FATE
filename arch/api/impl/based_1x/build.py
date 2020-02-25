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
from arch.api.base.utils.consts import CONF_KEY_SERVER, CONF_KEY_FEDERATION
from arch.api.base.utils.wrap import FederationWrapped
from arch.api.impl.based_1x.session import build_session
from arch.api.impl.based_1x.table import DTable
from arch.api.utils import file_utils


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
        if self._work_mode.is_standalone():
            from arch.api.impl.based_1x.federation_standalone import FederationRuntime
            return FederationRuntime(session_id=federation_id, runtime_conf=runtime_conf)

        elif self._work_mode.is_cluster():
            from arch.api.impl.based_1x.federation_cluster import FederationRuntime
            server_conf = file_utils.load_json_conf(server_conf_path)
            if CONF_KEY_SERVER not in server_conf:
                raise EnvironmentError("server_conf should contain key {}".format(CONF_KEY_SERVER))
            if CONF_KEY_FEDERATION not in server_conf.get(CONF_KEY_SERVER):
                raise EnvironmentError(
                    "The {} should be a json file containing key: {}".format(server_conf_path, CONF_KEY_FEDERATION))
            host = server_conf.get(CONF_KEY_SERVER).get(CONF_KEY_FEDERATION).get("host")
            port = server_conf.get(CONF_KEY_SERVER).get(CONF_KEY_FEDERATION).get("port")
            return FederationRuntime(session_id=federation_id, runtime_conf=runtime_conf, host=host, port=port)

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def build_wrapper(self):
        if self._work_mode.is_standalone():
            from eggroll.api.standalone.eggroll import _DTable
            return FederationWrapped(session_id=self._session_id, dtable_cls=_DTable, table_cls=self._table_cls)

        elif self._work_mode.is_cluster():
            from eggroll.api.cluster.eggroll import _DTable
            return FederationWrapped(session_id=self._session_id, dtable_cls=_DTable, table_cls=self._table_cls)

        raise ValueError(f"work_mode: ${self._work_mode} unknown")
