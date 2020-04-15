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
from arch.api.impl.based_spark.based_hdfs.table import RDDTable
from arch.api.impl.based_spark.based_hdfs.session import FateSessionImpl
from arch.api.impl.based_spark.based_hdfs.federation import FederationRuntime
from arch.api.base.utils.wrap import FederationWrapped


class Builder(build.Builder):
    _table_cls = RDDTable

    def __init__(self, session_id):
        self._session_id = session_id

    def build_session(self):
        return FateSessionImpl(self._session_id)

    def build_federation(self, federation_id, runtime_conf, server_conf_path):
        return FederationRuntime(session_id=federation_id, runtime_conf=runtime_conf)

    def build_wrapper(self):
        from pyspark.rdd import RDD
        return FederationWrapped(session_id=self._session_id, dtable_cls=RDD, table_cls=self._table_cls)

