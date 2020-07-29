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

from arch.api.utils import log_utils
from fate_arch.data_table.eggroll_table import EggRollTable
from fate_arch.data_table.hdfs_table import HDFSTable
from fate_arch.data_table.store_type import Relationship, StoreEngine
from fate_arch.session import Backend, WorkMode
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.manager.table_manager.table_operation import create

logger = log_utils.getLogger()

MAX_NUM = 10000


def convert(table, name='', namespace='', job_id=uuid.uuid1().hex, force=False, **kwargs):
    partitions = table.get_partitions()
    mode = table._mode if table._mode else WorkMode.CLUSTER
    if RuntimeConfig.BACKEND == Backend.EGGROLL:
        logger.info('backend is eggroll, storage engine is {}'.format(table.get_storage_engine()))
        if table.get_storage_engine() not in Relationship.CompToStore.get(RuntimeConfig.BACKEND, []):
            logger.info('convert {} table  to eggroll table'.format(table.get_storage_engine()))
            address = create(name=name, namespace=namespace, store_engine=StoreEngine.LMDB, partitions=partitions)
            logger.info('table info: name {}, namespace {}, store engine {}, partitions {}'.format(
                name, namespace, StoreEngine.LMDB, partitions))
            _table = EggRollTable(job_id=job_id, mode=mode, address=address, partitions=partitions, name=name, namespace=namespace)
            logger.info('start convert')
            copy_table(table, _table)
            logger.info('convert success')
            return _table
    elif RuntimeConfig.BACKEND == Backend.SPARK:
        logger.info('backend is spark, storage engine is {}'.format(table.get_storage_engine()))
        if table.get_storage_engine() not in Relationship.CompToStore.get(RuntimeConfig.BACKEND, []):
            logger.info('convert {} table to spark table'.format(table.get_storage_engine()))
            address = create(name=name, namespace=namespace, store_engine=StoreEngine.HDFS, partitions=partitions)
            logger.info('table info: name {}, namespace {}, store engine {}, partitions {}'.format(
                name, namespace, StoreEngine.HDFS, partitions))
            _table = HDFSTable(address=address, partitions=partitions, name=name, namespace=namespace)
            logger.info('start convert')
            copy_table(table, _table)
            logger.info('convert success')
            return _table
    else:
        return None


def copy_table(src_table, dest_table):
    count = 0
    data = []
    for k, v in src_table.collect():
        data.append((k, v))
        count += 1
        if len(data) == MAX_NUM:
            dest_table.put_all(data)
            count = 0
            data = []
    dest_table.save_schema(src_table.get_schema(), count=src_table.count())







