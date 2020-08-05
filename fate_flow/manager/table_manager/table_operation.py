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
from fate_arch.storage.address import EggRollAddress, HDFSAddress, MysqlAddress
from fate_arch.storage.simple_table import SimpleTable
from fate_arch.db.db_models import DB, StorageTableMeta
from fate_flow.utils import data_utils

from arch.api.utils.conf_utils import get_base_config
from fate_arch.storage.hdfs_table import HDFSTable
from fate_arch.storage.mysql_table import MysqlTable
from fate_arch.storage.constant import StorageEngine, Relationship
from fate_arch.common import WorkMode, Backend
from fate_flow.settings import WORK_MODE, data_manager_logger


def create(name, namespace, storage_engine, address=None, partitions=1, count=0):
    with DB.connection_context():
        metas = StorageTableMeta.select().where(StorageTableMeta.f_name == name,
                                                StorageTableMeta.f_namespace == namespace)
        is_insert = True
        if metas:
            if storage_engine != metas[0].f_engine:
                raise Exception('table {} {} has been created by storage engine {} '.format(name, namespace,
                                                                                            metas.f_engine))
            else:
                is_insert = False
                meta = metas[0]
        else:
            meta = StorageTableMeta()
            meta.f_create_time = current_timestamp()
            meta.f_name = name
            meta.f_namespace = namespace
            meta.f_engine = storage_engine
            if not address:
                if storage_engine in Relationship.CompToStore.get(Backend.EGGROLL):
                    address = EggRollAddress(name=name, namespace=namespace, storage_type=storage_engine)
                elif storage_engine in Relationship.CompToStore.get(Backend.SPARK):
                    address = HDFSAddress(path=data_utils.generate_hdfs_address())
            meta.f_address = address.__dict__ if address else {}
            meta.f_partitions = partitions
            meta.f_count = count
            meta.f_schema = serialize_b64({}, to_str=True)
            meta.f_part_of_data = serialize_b64([], to_str=True)
        meta.f_update_time = current_timestamp()
        if is_insert:
            meta.save(force_insert=True)
        else:
            meta.save()
        return get_address(engine=meta.f_engine, address_dict=meta.f_address)


def get_storage_info(name, namespace):
    with DB.connection_context():
        metas = StorageTableMeta.select().where(StorageTableMeta.f_name == name,
                                                StorageTableMeta.f_namespace == namespace)
        if metas:
            meta = metas[0]
            engine = meta.f_engine
            address_dict = meta.f_address
            address = get_address(engine=engine, address_dict=address_dict)
            partitions = meta.f_partitions
        else:
            return None, None, None
    return engine, address, partitions


def get_table(job_id: str = '',
              mode: typing.Union[int, WorkMode] = WORK_MODE,
              backend: typing.Union[int, Backend] = Backend.EGGROLL,
              persistent_engine: str = StorageEngine.LMDB,
              namespace: str = None,
              name: str = None,
              simple: bool = False,
              **kwargs):
    if not job_id:
        job_id = uuid.uuid1().hex
    if simple:
        return SimpleTable(name=name, namespace=namespace, data_name='')
    data_manager_logger.info('start get table by name {} namespace {}'.format(name, namespace))
    storage_engine, address, partitions = get_storage_info(name, namespace)
    if 'partition' in kwargs.keys():
        partitions = kwargs.get('partition')
        kwargs.pop('partition')
    if not storage_engine:
        data_manager_logger.error('no find table')
        return None
    data_manager_logger.info('table store engine is {}'.format(storage_engine))
    if storage_engine == 'MYSQL':
        data_manager_logger.info(address)
        if not address:
            database_config = get_base_config("data_storage_address", {})
            address = MysqlAddress(user=database_config.get('user'),
                                   passwd=database_config.get('passwd'),
                                   host=database_config.get('host'),
                                   port=database_config.get('port'),
                                   db=namespace, name=name)
        data_manager_logger.info(
            'get mysql table mode {} store_engine {} partition {}'.format(mode, storage_engine, partitions))
        return MysqlTable(mode=mode, persistent_engine=StorageEngine.MYSQL, address=address, partitions=partitions,
                          name=name, namespace=namespace)
    if storage_engine in Relationship.CompToStore.get(Backend.EGGROLL):
        data_manager_logger.info(
            'get eggroll table mode {} store_engine {} partition {}'.format(mode, storage_engine, partitions))
        from fate_arch.storage.eggroll_table import EggRollTable
        return EggRollTable(job_id=job_id, mode=mode, persistent_engine=persistent_engine, name=name,
                            namespace=namespace, partitions=partitions, address=address, **kwargs)
    if storage_engine in Relationship.CompToStore.get(Backend.SPARK):
        data_manager_logger.info(
            'get spark table store_engine {} partition {} path {}'.format(storage_engine, partitions, address.path))
        return HDFSTable(address=address, partitions=partitions, name=name, namespace=namespace)


def create_table(job_id: str = uuid.uuid1().hex,
                 mode: typing.Union[int, WorkMode] = WORK_MODE,
                 engine: str = StorageEngine.LMDB,
                 namespace: str = None,
                 name: str = None,
                 partitions: int = 1,
                 **kwargs):
    data_manager_logger.info('start create {} table'.format(engine))
    if engine in Relationship.CompToStore.get(Backend.EGGROLL):
        address = create(name=name, namespace=namespace, storage_engine=engine, partitions=partitions)
        data_manager_logger.info('create success')
        if mode == WorkMode.CLUSTER:
            from fate_arch.storage.eggroll_table import EggRollTable
            return EggRollTable(job_id=job_id, mode=mode, persistent_engine=engine, namespace=namespace,
                                name=name,
                                address=address, partitions=partitions, **kwargs)
        else:
            from fate_arch.storage.standalone_table import StandaloneTable
            return StandaloneTable(job_id, persistent_engine=engine, namespace=namespace, name=name,
                                   address=address, partitions=partitions)

    if engine in Relationship.CompToStore.get(Backend.SPARK):
        data_manager_logger.info('create success')
        address = create(name=name, namespace=namespace, storage_engine=engine, partitions=partitions)
        return HDFSTable(address=address, partitions=partitions, namespace=namespace, name=name, **kwargs)
    else:
        raise Exception('does not support the creation of this type of table :{}'.format(engine))


def get_address(engine, address_dict):
    if engine in Relationship.CompToStore.get(Backend.EGGROLL):
        address = EggRollAddress(**address_dict)
    elif engine in Relationship.CompToStore.get(Backend.SPARK):
        address = HDFSAddress(*address_dict)
    else:
        address = None
    return address
