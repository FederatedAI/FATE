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

from arch.api import session
from arch.api import RuntimeInstance
from arch.api import WorkMode
from arch.api.utils import version_control
import datetime


def save_model(buffer_type, proto_buffer, name, namespace, version_log=None):
    data_table = session.table(name=name, namespace=namespace, partition=get_model_table_partition_count(),
                               create_if_missing=True, error_if_exist=False)
    # todo:  model slice?
    data_table.put(buffer_type, proto_buffer.SerializeToString(), use_serialize=False)
    version_log = "[AUTO] save model at %s." % datetime.datetime.now() if not version_log else version_log
    version_control.save_version(name=name, namespace=namespace, version_log=version_log)


def read_model(buffer_type, proto_buffer, name, namespace):
    data_table = session.table(name=name, namespace=namespace, partition=get_model_table_partition_count(),
                               create_if_missing=False, error_if_exist=False)
    if data_table:
        buffer_bytes = data_table.get(buffer_type, use_serialize=False)
        if buffer_bytes:
            proto_buffer.ParseFromString(buffer_bytes)
        else:
            return 1
        return 0
    else:
        return 2


def get_model_table_partition_count():
    # todo: max size limit?
    return 4 if RuntimeInstance.MODE == WorkMode.CLUSTER else 1
