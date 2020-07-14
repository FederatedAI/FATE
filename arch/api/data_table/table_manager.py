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

import typing
import uuid

from arch.api import WorkMode, Backend
from arch.api.base.utils.store_type import StoreTypes, StoreEngine
from arch.api.data_table.mysql_table import MysqlTable
from arch.api.utils.conf_utils import get_base_config
from arch.api.data_table.eggroll_table import EggRollTable


def get_table(job_id: str = uuid.uuid1(),
              mode: typing.Union[int, WorkMode] = WorkMode.STANDALONE,
              backend: typing.Union[int, Backend] = Backend.EGGROLL,
              persistent_engine: str = StoreTypes.ROLLPAIR_LMDB,
              store_engine: str = StoreEngine.EGGROLL[0],
              namespace: str = None,
              name: str = None,
              partition: int = 1,
              init_session: bool = False,
              **kwargs):

    if store_engine in StoreEngine.MYSQL:
        if 'data_storage_config' in kwargs:
            database_config = kwargs.get('data_storage_config')
        else:
            database_config = get_base_config("data_storage_config", {})
        return MysqlTable(mode, StoreTypes.MYSQL, namespace, name, partition, database_config)

    if store_engine in StoreEngine.EGGROLL:
        return EggRollTable(job_id=job_id,  mode=mode, backend=backend, persistent_engine=persistent_engine,
                            namespace=namespace, name=name, partition=partition, init_session=init_session, **kwargs)

