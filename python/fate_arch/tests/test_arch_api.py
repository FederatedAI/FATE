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
import uuid

import numpy as np
from fate_arch import session
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine

work_mode = 0
backend = 0

#sess = session.Session.create(computing=ComputingEngine.STANDALONE, federation=FederationEngine.STANDALONE, storage=StorageEngine.STANDALONE)
#sess = session.Session.create(computing=ComputingEngine.STANDALONE)
#sess = session.Session.create(work_mode=work_mode, backend=backend)
sess = session.Session.create(work_mode=work_mode)
sess.init_computing().as_default()

data = []
for i in range(10):
    features = np.random.random(10)
    features = ",".join([str(x) for x in features])
    data.append((i, features))

c_table = session.get_latest_opened().computing.parallelize(data, include_key=True, partition=4)
for k, v in c_table.collect():
    print(v)
print()

table_meta = sess.persistent(computing_table=c_table, table_namespace="jarvis_experiment", table_name=str(uuid.uuid1()))

storage_session = sess.storage(namespace=table_meta.get_namespace(), name=table_meta.get_name())
s_table = storage_session.get_table()
for k, v in s_table.collect():
    print(v)
print()

t2 = session.get_latest_opened().computing.load(table_meta.get_address(), partitions=table_meta.get_partitions(), schema=table_meta.get_schema())
for k, v in t2.collect():
    print(v)

sess.destroy_all_sessions()

