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
from typing import Iterable
from eggroll.api.utils.core import json_dumps, json_loads
from eggroll.api.version_control import control
from eggroll.api import eggroll
import datetime


def save_data(kv_data: Iterable, name, namespace, partition=1, create_if_missing=True, error_if_exist=False, version_log=None):
    """
    save data into data table
    :param kv_data:
    :param name: table name of data table
    :param namespace: table namespace of data table
    :param partition: number of partition
    :param create_if_missing:
    :param error_if_exist:
    :return:
        data table instance
    """
    data_table = eggroll.table(name=name, namespace=namespace, partition=partition,
                               create_if_missing=create_if_missing, error_if_exist=error_if_exist)
    data_table.put_all(kv_data)
    version_log = "[AUTO] save data at %s." % datetime.datetime.now() if not version_log else version_log
    control.save_version(name=name, namespace=namespace, version_log=version_log)
    return data_table


def get_data_table(name, namespace):
    """
    return data table instance by table name and table name space
    :param name: table name of data table
    :param namespace: table name space of data table
    :return:
        data table instance
    """
    return eggroll.table(name=name, namespace=namespace, create_if_missing=False)


def save_data_table_meta(kv, data_table_name, data_table_namespace):
    """
    save data table meta information
    :param kv: v should be serialized by JSON
    :param data_table_name: table name of this data table
    :param data_table_namespace: table name of this data table
    :return:
    """
    data_meta_table = eggroll.table(name="%s.meta" % data_table_name,
                                    namespace=data_table_namespace,
                                    partition=1,
                                    create_if_missing=True, error_if_exist=False)
    for k, v in kv.items():
        data_meta_table.put(k, json_dumps(v), use_serialize=False)


def get_data_table_meta(key, data_table_name, data_table_namespace):
    """
    get data table meta information
    :param key:
    :param data_table_name: table name of this data table
    :param data_table_namespace: table name of this data table
    :return:
    """
    data_meta_table = eggroll.table(name="%s.meta" % data_table_name,
                                    namespace=data_table_namespace,
                                    create_if_missing=True,
                                    error_if_exist=False)
    if data_meta_table:
        value_bytes = data_meta_table.get(key, use_serialize=False)
        if value_bytes:
            return json_loads(value_bytes)
        else:
            return None
    else:
        return None


if __name__ == '__main__':
    from eggroll.api import eggroll
    import uuid
    import random
    job_id = str(uuid.uuid1().hex)
    eggroll.init(job_id=job_id, mode=0)

    table_name = "test_example"
    table_namespace = "storage_test_example"

    def gen_test_data(row_count, column_count):
        for r in range(row_count):
            k = uuid.uuid1().hex
            v = ','.join([str(random.randint(1, 100)) for i in range(column_count - 1)])
            yield k, v

    data_table = save_data(gen_test_data(5, 10), name=table_name, namespace=table_namespace)

    for k, v in data_table.collect():
        print(k, v)

    save_data_table_meta({"k1": {"t1": [1, 2]}}, data_table_name=table_name, data_table_namespace=table_namespace)
    print(get_data_table_meta("k1", data_table_name=table_name, data_table_namespace=table_namespace))


