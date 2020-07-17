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
import typing
import uuid
from typing import Iterable

from arch.api import WorkMode, Backend
from arch.api.base.utils.store_type import StoreEngine
from arch.api.data_table.base import Table, EggRollStorage
from arch.api.utils.profile_util import log_elapsed
from fate_flow.settings import WORK_MODE
from arch.api.data_table import eggroll_session

# noinspection SpellCheckingInspection,PyProtectedMember,PyPep8Naming
class EggRollTable(Table):
    def __init__(self,
                 job_id: str = uuid.uuid1(),
                 mode: typing.Union[int, WorkMode] = WORK_MODE,
                 backend: typing.Union[int, Backend] = Backend.EGGROLL,
                 persistent_engine: str = StoreEngine.LMDB,
                 namespace: str = None,
                 name: str = None,
                 partitions: int = 1,
                 **kwargs):
        self._mode = mode
        self._name = name or str(uuid.uuid1())
        self._namespace = namespace or str(uuid.uuid1())
        self._partitions = partitions
        self._strage_engine = persistent_engine
        self.session = eggroll_session.get_session(session_id=job_id, work_mode=mode)
        self._table = self.session.table(namespace=namespace, name=name, partition=partitions, **kwargs)

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def get_partitions(self):
        return self._table.get_partitions()

    def get_storage_engine(self):
        return self._strage_engine

    def get_address(self):
        return EggRollStorage(namespace=self._namespace, name=self._name)

    def put_all(self, kv_list: Iterable, use_serialize=True, chunk_size=100000):
        return self._table.put_all(kv_list, use_serialize, chunk_size)

    @log_elapsed
    def collect(self, min_chunk_size=0, use_serialize=True, **kwargs) -> list:
        return self._table.get_all(min_chunk_size, use_serialize, **kwargs)

    def destroy(self):
        super().destroy()
        return self._table.destroy()

    @classmethod
    def dtable(cls, session_id, name, namespace, partitions):
        return EggRollTable(session_id=session_id, name=name, namespace=namespace, partitions=partitions)

    @log_elapsed
    def save_as(self, name, namespace, partition=None, **kwargs):

        from arch.api import RuntimeInstance
        options = kwargs.get("options", {})
        store_type = options.get("store_type", RuntimeInstance.SESSION.get_persistent_engine())
        options["store_type"] = store_type

        if partition is None:
            partition = self._partitions
        self._table.save_as(name=name, namespace=namespace, partition=partition, options=options)

        return self.dtable(self._session_id, name, namespace, partition)

    def close(self):
        self.session.stop()

    @log_elapsed
    def count(self, **kwargs):
        return self._table.count()




