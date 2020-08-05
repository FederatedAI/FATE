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

from arch.api.utils.core_utils import current_timestamp, serialize_b64, deserialize_b64
from fate_arch.data_table.address import EggRollAddress, HDFSAddress, MysqlAddress
from fate_arch.data_table.simple_table import SimpleTable
from fate_arch.db.db_models import DB, StoreTableMeta
from fate_flow.utils import data_utils

from arch.api.utils.conf_utils import get_base_config
from fate_arch.data_table.hdfs_table import HDFSTable
from fate_arch.data_table.mysql_table import MysqlTable
from fate_arch.data_table.store_type import StoreEngine, Relationship
from fate_arch.common import WorkMode, Backend
from fate_flow.settings import WORK_MODE, data_manager_logger


def create(name, namespace, store_engine, address=None, partitions=1, count=0):
    with DB.connection_context():
        schema = StoreTableMeta.select().where(StoreTableMeta.f_table_name == name,
                                               StoreTableMeta.f_namespace == namespace)
        is_insert = True
        if schema:
            if store_engine != schema[0].f_data_store_engine:
                raise Exception('table {} {} has been created by store engine {} '.format(name, namespace,
                                                                                          schema.f_data_store_engine))
            else:
                return
        else:
            schema = StoreTableMeta()
            schema.f_create_time = current_timestamp()
            schema.f_table_name = name
            schema.f_namespace = namespace
            schema.f_data_store_engine = store_engine
            if not address:
                if store_engine in Relationship.CompToStore.get(Backend.EGGROLL):
                    address = EggRollAddress(name=name, namespace=namespace, store_type=store_engine)
                elif store_engine in Relationship.CompToStore.get(Backend.SPARK):
                    address = HDFSAddress(path=data_utils.generate_hdfs_address())
            schema.f_address = address.__dict__ if address else {}
            schema.f_partitions = partitions
            schema.f_count = count
            schema.f_schema = serialize_b64({}, to_str=True)
            schema.f_part_of_data = serialize_b64([], to_str=True)
        schema.f_update_time = current_timestamp()
        if is_insert:
            schema.save(force_insert=True)
        else:
            schema.save()
        return address


def get_store_info(name, namespace):
    with DB.connection_context():
        schema = StoreTableMeta.select().where(StoreTableMeta.f_table_name == name,
                                               StoreTableMeta.f_namespace == namespace)
        if schema:
            schema = schema[0]
            store_info = schema.f_data_store_engine
            address_dict = schema.f_address
            address = get_address(store_engine=store_info, address_dict=address_dict)
            partitions = schema.f_partitions
        else:
            return None, None, None
    return store_info, address, partitions


def get_table(job_id: str = '',
              mode: typing.Union[int, WorkMode] = WORK_MODE,
              backend: typing.Union[int, Backend] = Backend.EGGROLL,
              persistent_engine: str = StoreEngine.LMDB,
              namespace: str = None,
              name: str = None,
              simple: bool = False,
              **kwargs):
    if not job_id:
        job_id = uuid.uuid1().hex
    if simple:
        return SimpleTable(name=name, namespace=namespace, data_name='')
    data_manager_logger.info('start get table by name {} namespace {}'.format(name, namespace))
    store_engine, address, partitions = get_store_info(name, namespace)
    if 'partition' in kwargs.keys():
        partitions = kwargs.get('partition')
        kwargs.pop('partition')
    if not store_engine:
        data_manager_logger.error('no find table')
        return None
    data_manager_logger.info('table store engine is {}'.format(store_engine))
    if store_engine == 'MYSQL':
        data_manager_logger.info(address)
        if not address:
            database_config = get_base_config("data_storage_address", {})
            address = MysqlAddress(user=database_config.get('user'),
                                   passwd=database_config.get('passwd'),
                                   host=database_config.get('host'),
                                   port=database_config.get('port'),
                                   db=namespace, name=name)
        data_manager_logger.info(
            'get mysql table mode {} store_engine {} partition {}'.format(mode, store_engine, partitions))
        return MysqlTable(mode=mode, persistent_engine=StoreEngine.MYSQL, address=address, partitions=partitions,
                          name=name, namespace=namespace)
    if store_engine in Relationship.CompToStore.get(Backend.EGGROLL):
        data_manager_logger.info(
            'get eggroll table mode {} store_engine {} partition {}'.format(mode, store_engine, partitions))
        from fate_arch.data_table.eggroll_table import EggRollTable
        return EggRollTable(job_id=job_id, mode=mode, persistent_engine=persistent_engine, name=name,
                            namespace=namespace, partitions=partitions, address=address, **kwargs)
    if store_engine in Relationship.CompToStore.get(Backend.SPARK):
        data_manager_logger.info(
            'get spark table store_engine {} partition {} path {}'.format(store_engine, partitions, address.path))
        return HDFSTable(address=address, partitions=partitions, name=name, namespace=namespace)


def create_table(job_id: str = uuid.uuid1().hex,
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
        if mode == WorkMode.CLUSTER:
            from fate_arch.data_table.eggroll_table import EggRollTable
            return EggRollTable(job_id=job_id, mode=mode, persistent_engine=store_engine, namespace=namespace,
                                name=name,
                                address=address, partitions=partitions, **kwargs)
        else:
            from fate_arch.data_table.standalone_table import StandaloneTable
            return StandaloneTable(job_id, persistent_engine=store_engine, namespace=namespace, name=name,
                                   address=address, partitions=partitions)

    if store_engine in Relationship.CompToStore.get(Backend.SPARK):
        data_manager_logger.info('create success')
        address = create(name=name, namespace=namespace, store_engine=store_engine, partitions=partitions)
        return HDFSTable(address=address, partitions=partitions, namespace=namespace, name=name, **kwargs)
    else:
        raise Exception('does not support the creation of this type of table :{}'.format(store_engine))


def get_address(store_engine, address_dict):
    if store_engine in Relationship.CompToStore.get(Backend.EGGROLL):
        address = EggRollAddress(**address_dict)
    elif store_engine in Relationship.CompToStore.get(Backend.SPARK):
        address = HDFSAddress(*address_dict)
    else:
        address = {}
    return address
