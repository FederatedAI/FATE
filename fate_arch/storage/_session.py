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


import peewee
from fate_arch.abc import StorageSessionABC, StorageTableABC
from fate_arch.common import compatibility_utils
from fate_arch.common.base_utils import fate_uuid, current_timestamp
from fate_arch.common.log import getLogger
from fate_arch.computing import ComputingEngine
from fate_arch.storage._table import StorageTableMeta
from fate_arch.storage._types import StorageEngine, Relationship
from fate_arch.storage.metastore.db_models import DB, StorageTableMetaModel, SessionRecord

MAX_NUM = 10000

LOGGER = getLogger()


class Session(object):
    @classmethod
    def build(cls, session_id=None, storage_engine=None, computing_engine=None, **kwargs):
        session_id = session_id if session_id else fate_uuid()
        address = kwargs.get('address', None)
        # Find the storage engine type
        if storage_engine is None and kwargs.get("name") and kwargs.get("namespace"):
            storage_engine, address, partitions = StorageSessionBase.get_storage_info(name=kwargs.get("name"),
                                                                                      namespace=kwargs.get("namespace"))
        if storage_engine is None and computing_engine is None:
            computing_engine, federation_engine, federation_mode = compatibility_utils.backend_compatibility(**kwargs)
        if storage_engine is None and computing_engine:
            # Gets the computing engine default storage engine
            storage_engine = Relationship.CompToStore.get(computing_engine)[0]

        if storage_engine == StorageEngine.EGGROLL:
            from fate_arch.storage.eggroll import StorageSession
            storage_session = StorageSession(session_id=session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.STANDALONE:
            from fate_arch.storage.standalone import StorageSession
            return StorageSession(session_id=session_id, options=kwargs.get("options", {}))
        if storage_engine == StorageEngine.MYSQL:
            from fate_arch.storage.mysql import StorageSession
            return StorageSession(session_id=session_id, options=kwargs.get("options", {}))
        else:
            raise NotImplementedError(f"can not be initialized with storage engine: {storage_engine}")
        if kwargs.get("name") and kwargs.get("namespace"):
            storage_session.set_default(name=kwargs["name"], namespace=kwargs["namespace"])
        return storage_session

    @classmethod
    def convert(cls, src_name, src_namespace, dest_name, dest_namespace,
                computing_engine: ComputingEngine = ComputingEngine.EGGROLL, force=False):
        # The source and target may be different session types
        src_table_meta = StorageTableMeta.build(name=src_name, namespace=src_namespace)
        if not src_table_meta:
            raise RuntimeError(f"can not found table name: {src_name} namespace: {src_namespace}")
        dest_table_address = None
        dest_table_engine = None
        if src_table_meta.get_engine() not in Relationship.CompToStore.get(computing_engine, []):
            if computing_engine == ComputingEngine.STANDALONE:
                from fate_arch.storage import EggRollStorageType
                dest_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.STANDALONE,
                                                                     address_dict=dict(name=dest_name,
                                                                                       namespace=dest_namespace,
                                                                                       storage_type=EggRollStorageType.ROLLPAIR_LMDB))
                dest_table_engine = StorageEngine.STANDALONE
            elif computing_engine == ComputingEngine.EGGROLL:
                from fate_arch.storage.eggroll import StorageSession
                from fate_arch.storage import EggRollStorageType
                dest_table_address = StorageTableMeta.create_address(storage_engine=StorageEngine.EGGROLL,
                                                                     address_dict=dict(name=dest_name,
                                                                                       namespace=dest_namespace,
                                                                                       storage_type=EggRollStorageType.ROLLPAIR_LMDB))
                dest_table_engine = StorageEngine.EGGROLL
            elif computing_engine == ComputingEngine.SPARK:
                pass
            else:
                raise RuntimeError(f"can not support computing engine {computing_engine}")
            return src_table_meta, dest_table_address, dest_table_engine
        else:
            return src_table_meta, dest_table_address, dest_table_engine

    @classmethod
    def copy_table(cls, src_table, dest_table):
        count = 0
        data = []
        part_of_data = []
        for k, v in src_table.collect():
            data.append((k, v))
            count += 1
            if count < 100:
                part_of_data.append((k, v))
            if len(data) == MAX_NUM:
                dest_table.put_all(data)
                data = []
        if data:
            dest_table.put_all(data)
        dest_table.get_meta().update_metas(schema=src_table.get_meta(meta_type="schema"), count=src_table.count(),
                                           part_of_data=part_of_data)


class StorageSessionBase(StorageSessionABC):
    def __init__(self, session_id):
        self._session_id = session_id
        self._default_name = None
        self._default_namespace = None

    def create(self):
        raise NotImplementedError()

    def create_table(self, address, name, namespace, partitions=1, **kwargs):
        table = self.table(address=address, name=name, namespace=namespace, partitions=partitions, **kwargs)
        meta_info = {}
        meta_info.update(kwargs)
        meta_info["name"] = name
        meta_info["namespace"] = namespace
        meta_info["address"] = table.get_address()
        meta_info["partitions"] = table.get_partitions()
        meta_info["engine"] = table.get_engine()
        meta_info["type"] = table.get_type()
        meta_info["options"] = table.get_options()
        StorageTableMeta.create_metas(**meta_info)
        table.set_meta(StorageTableMeta.build(name=name, namespace=namespace))
        # update count on meta
        table.count()
        return table

    def set_default(self, name, namespace):
        self._default_name = name
        self._default_namespace = namespace

    def get_table(self, name=None, namespace=None):
        if not name or not namespace:
            name = self._default_name
            namespace = self._default_namespace
        meta = StorageTableMeta.build(name=name, namespace=namespace)
        if meta:
            table = self.table(name=meta.get_name(),
                               namespace=meta.get_namespace(),
                               address=meta.get_address(),
                               partitions=meta.get_partitions(),
                               storage_type=meta.get_type(),
                               options=meta.get_options())
            table.set_meta(meta)
            return table
        else:
            return None

    def table(self, name, namespace, address, partitions, storage_type=None, options=None, **kwargs) -> StorageTableABC:
        raise NotImplementedError()

    @classmethod
    def get_storage_info(cls, name, namespace):
        with DB.connection_context():
            metas = StorageTableMetaModel.select().where(StorageTableMetaModel.f_name == name,
                                                         StorageTableMetaModel.f_namespace == namespace)
            if metas:
                meta = metas[0]
                engine = meta.f_engine
                address_dict = meta.f_address
                address = StorageTableMeta.create_address(storage_engine=engine, address_dict=address_dict)
                partitions = meta.f_partitions
            else:
                return None, None, None
        return engine, address, partitions

    def __enter__(self):
        with DB.connection_context():
            session_record = SessionRecord()
            session_record.f_session_id = self._session_id
            session_record.f_engine_type = "storage"
            # TODO: engine address
            session_record.f_engine_address = {}
            session_record.f_create_time = current_timestamp()
            rows = session_record.save(force_insert=True)
            if rows != 1:
                raise Exception(f"create session record {self._session_id} failed")
            LOGGER.info(f"save session {self._session_id} record")
        self.create()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        with DB.connection_context():
            rows = SessionRecord.delete().where(SessionRecord.f_session_id == self._session_id).execute()
            if rows > 0:
                LOGGER.info(f"delete session {self._session_id} record")
            else:
                LOGGER.warning(f"failed delete session {self._session_id} record")

    def close(self):
        try:
            self.stop()
        except Exception as e:
            self.kill()

    def stop(self):
        raise NotImplementedError()

    def kill(self):
        raise NotImplementedError()

    def session_id(self):
        return self._session_id
