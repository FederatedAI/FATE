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

from fate_arch.common.profile import log_elapsed
from fate_arch.data_table import eggroll_session
from fate_arch.data_table.base import Table, AddressABC
from fate_arch.data_table.store_type import StoreEngine
from fate_arch.session import WorkMode
from fate_flow.settings import WORK_MODE


# noinspection SpellCheckingInspection,PyProtectedMember,PyPep8Naming
class EggRollTable(Table):
    def __init__(self,
                 job_id: str = uuid.uuid1(),
                 mode: typing.Union[int, WorkMode] = WORK_MODE,
                 persistent_engine: str = StoreEngine.LMDB,
                 namespace: str = None,
                 name: str = None,
                 partitions: int = 1,
                 address=AddressABC(),
                 **kwargs):
        self._mode = mode
        self._address = address
        self._strage_engine = persistent_engine
        self._session_id = job_id
        self.session = eggroll_session.get_session(session_id=self._session_id, work_mode=mode)
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
        return self._address

    def put_all(self, kv_list: Iterable, **kwargs):
        return self._table.put_all(kv_list)

    @log_elapsed
    def collect(self, **kwargs) -> list:
        return self._table.get_all(**kwargs)

    def destroy(self):
        super().destroy()
        return self._table.destroy()

    @classmethod
    def dtable(cls, session_id, name, namespace, partitions):
        return EggRollTable(session_id=session_id, name=name, namespace=namespace, partitions=partitions)

    @log_elapsed
    def save_as(self, name, namespace, partition=None, **kwargs):

        options = kwargs.get("options", {})
        store_type = options.get("store_type", StoreEngine.LMDB)
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
