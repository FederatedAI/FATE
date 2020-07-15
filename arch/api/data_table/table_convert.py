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
from arch.api.base.utils.store_type import StoreEngine
from arch.api.data_table.eggroll_table import EggRollTable
from arch.standalone import WorkMode
from fate_flow.utils.data_utils import create

MAX_NUM = 10000


def convert(table, computing_engine, **kwargs):
    name = table.get_name()
    namespace = table.get_namespace()
    partition = table.get_partition()
    mode = table._mode if table._mode else WorkMode.CLUSTER
    computing_engine = computing_engine
    if computing_engine == 'EGGROLL' and table.get_storage_engine != StoreEngine.EGGROLL:
        _table = EggRollTable(mode=mode, namespace=namespace, name=name, partition=partition)
        count = 0
        data = []
        for line in _table.collect():
            data.append(line)
            count += 1
            if len(data) == MAX_NUM:
                _table.put_all(data)
                count = 0
                data = []
        table.close()
        create(name=name, namespace=namespace, store_engine=StoreEngine.EGGROLL,
               address={'name': name, 'namespace': namespace})
        return _table



