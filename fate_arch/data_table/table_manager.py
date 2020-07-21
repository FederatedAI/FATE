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
import json
import typing
import uuid

from arch.api.utils.conf_utils import get_base_config
from fate_arch.data_table.eggroll_table import EggRollTable
from fate_arch.data_table.hdfs_table import HDFSTable
from fate_arch.data_table.mysql_table import MysqlTable
from fate_arch.data_table.store_type import StoreEngine, Relationship
from fate_arch.session import WorkMode, Backend
from fate_flow.manager.table_manager import get_store_info, create
from fate_flow.settings import WORK_MODE


def get_table(job_id: str = uuid.uuid1(),
              mode: typing.Union[int, WorkMode] = WORK_MODE,
              backend: typing.Union[int, Backend] = Backend.EGGROLL,
              persistent_engine: str = StoreEngine.LMDB,
              namespace: str = None,
              name: str = None,
              **kwargs):
    store_engine, address, partitions = get_store_info(name, namespace)
    if not store_engine:
        return None
    if store_engine == 'MYSQL':
        if address:
            database_config = json.loads(address)
        else:
            database_config = get_base_config("data_storage_config", {})
        return MysqlTable(mode=mode, persistent_engine=StoreEngine.MYSQL, namespace=namespace, name=name,
                          partitions=partitions, database_config=database_config)
    if store_engine in Relationship.CompToStore.get(Backend.EGGROLL):
        return EggRollTable(job_id=job_id,  mode=mode, backend=backend, persistent_engine=persistent_engine,
                            namespace=namespace, name=name, partitions=partitions, **kwargs)
    if store_engine in Relationship.CompToStore.get(Backend.SPARK):
        return HDFSTable(namespace, name, partitions)


def create_table(job_id: str = uuid.uuid1(),
                 mode: typing.Union[int, WorkMode] = WORK_MODE,
                 store_engine: str = StoreEngine.LMDB,
                 namespace: str = None,
                 name: str = None,
                 partitions: int = 1,
                 **kwargs):
    if store_engine in Relationship.CompToStore.get(Backend.EGGROLL):
        create(name=name, namespace=namespace, store_engine=store_engine, partitions=partitions)
        return EggRollTable(job_id=job_id, mode=mode, persistent_engine=store_engine,
                            namespace=namespace, name=name, partitions=partitions, **kwargs)

    if store_engine in Relationship.CompToStore.get(Backend.SPARK):
        create(name=name, namespace=namespace, store_engine=store_engine,partitions=partitions)
        return HDFSTable(namespace=namespace, name=name, partitions=partitions, **kwargs)
