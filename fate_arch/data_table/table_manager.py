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

from arch.api.utils.conf_utils import get_base_config
from fate_arch.data_table.base import MysqlAddress
from fate_arch.data_table.eggroll_table import EggRollTable
from fate_arch.data_table.hdfs_table import HDFSTable
from fate_arch.data_table.mysql_table import MysqlTable
from fate_arch.data_table.store_type import StoreEngine, Relationship
from fate_arch.session import WorkMode, Backend
from fate_flow.manager.table_manager import get_store_info, create
from fate_flow.settings import WORK_MODE, data_manager_logger


def get_table(job_id: str = uuid.uuid1(),
              mode: typing.Union[int, WorkMode] = WORK_MODE,
              backend: typing.Union[int, Backend] = Backend.EGGROLL,
              persistent_engine: str = StoreEngine.LMDB,
              namespace: str = None,
              name: str = None,
              **kwargs):
    data_manager_logger.info('start get table by name {} namespace {}'.format(name, namespace))
    store_engine, address, partitions = get_store_info(name, namespace)
    if not store_engine:
        data_manager_logger.error('no find table')
        return None
    data_manager_logger.info('table store engine is {}'.format(store_engine))
    if store_engine == 'MYSQL':
        if not address:
            database_config = get_base_config("data_storage_config", {})
            address = MysqlAddress(user=database_config.get('user'),
                                   passwd=database_config.get('passwd'),
                                   host=database_config.get('host'),
                                   port=database_config.get('port'),
                                   db=namespace, name=name)
        data_manager_logger.info('get mysql table mode {} store_engine {} partition {}'.format(mode, store_engine, partitions))
        return MysqlTable(mode=mode, persistent_engine=StoreEngine.MYSQL, address=address, partitions=partitions,
                          name=name, namespace=namespace)
    if store_engine in Relationship.CompToStore.get(Backend.EGGROLL):
        data_manager_logger.info('get eggroll table mode {} store_engine {} partition {}'.format(mode, store_engine, partitions))
        return EggRollTable(job_id=job_id,  mode=mode, persistent_engine=persistent_engine, name=name,
                            namespace=namespace, partitions=partitions, address=address, **kwargs)
    if store_engine in Relationship.CompToStore.get(Backend.SPARK):
        data_manager_logger.info('get spark table store_engine {} partition {} path {}'.format(store_engine, partitions, address.path))
        return HDFSTable(address=address, partitions=partitions, name=name, namespace=namespace)


def create_table(job_id: str = uuid.uuid1(),
                 mode: typing.Union[int, WorkMode] = WORK_MODE,
                 store_engine: str = StoreEngine.LMDB,
                 namespace: str = None,
                 name: str = None,
                 partitions: int = 1,
                 **kwargs):
    data_manager_logger.info('start create {} table'.format(store_engine))
    if store_engine in Relationship.CompToStore.get(Backend.EGGROLL):
        address = create(name=name, namespace=namespace, store_engine=store_engine, partitions=partitions)
        data_manager_logger.info('create success')
        return EggRollTable(job_id=job_id, mode=mode, persistent_engine=store_engine, namespace=namespace, name=name,
                            address=address, partitions=partitions, **kwargs)

    if store_engine in Relationship.CompToStore.get(Backend.SPARK):
        data_manager_logger.info('create success')
        address = create(name=name, namespace=namespace, store_engine=store_engine, partitions=partitions)
        return HDFSTable(address=address, partitions=partitions, namespace=namespace, name=name, **kwargs)
    else:
        raise Exception('does not support the creation of this type of table :{}'.format(store_engine))
