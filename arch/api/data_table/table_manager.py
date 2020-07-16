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

from arch.api import WorkMode, Backend
from arch.api.base.utils.store_type import StoreTypes, StoreEngine
from arch.api.data_table.mysql_table import MysqlTable
from arch.api.utils.conf_utils import get_base_config
from arch.api.data_table.eggroll_table import EggRollTable
from fate_flow.settings import WORK_MODE
from fate_flow.utils.data_utils import get_store_info


def get_table(job_id: str = uuid.uuid1(),
              mode: typing.Union[int, WorkMode] = WORK_MODE,
              backend: typing.Union[int, Backend] = Backend.EGGROLL,
              persistent_engine: str = StoreEngine.LMDB,
              namespace: str = None,
              name: str = None,
              init_session: bool = False,
              **kwargs):
    store_engine, address, partition = get_store_info(name, namespace)
    if store_engine == 'MYSQL':
        if address:
            database_config = json.loads(address)
        else:
            database_config = get_base_config("data_storage_config", {})
        return MysqlTable(mode=mode, persistent_engine=StoreEngine.MYSQL, namespace=namespace, name=name,
                          partition=partition, database_config=database_config)
    if store_engine == 'EGGROLL':
        return EggRollTable(job_id=job_id,  mode=mode, backend=backend, persistent_engine=persistent_engine,
                            namespace=namespace, name=name, partition=partition, init_session=init_session, **kwargs)
    else:
        # set default table
        return EggRollTable(job_id=job_id, mode=mode, backend=backend, persistent_engine=persistent_engine,
                            namespace=namespace, name=name, partition=partition, init_session=init_session, **kwargs)

