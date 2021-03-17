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


from fate_arch.abc import StorageSessionABC, StorageTableABC
from fate_arch.common import compatibility_utils, EngineType
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
            storage_session = StorageSession(session_id=session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.MYSQL:
            from fate_arch.storage.mysql import StorageSession
            storage_session = StorageSession(session_id=session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.HDFS:
            from fate_arch.storage.hdfs import StorageSession
            storage_session = StorageSession(session_id=session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.FILE:
            from fate_arch.storage.file import StorageSession
            storage_session = StorageSession(session_id=session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.PATH:
            from fate_arch.storage.path import StorageSession
            storage_session = StorageSession(session_id=session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.LOCAL:
            from fate_arch.storage.local import StorageSession
            storage_session = StorageSession(session_id=session_id, options=kwargs.get("options", {}))

        else:
            raise NotImplementedError(f"can not be initialized with storage engine: {storage_engine}")
        if kwargs.get("name") and kwargs.get("namespace"):
            storage_session.set_default(name=kwargs["name"], namespace=kwargs["namespace"])
        return storage_session


class StorageSessionBase(StorageSessionABC):
    def __init__(self, session_id, engine_name):
        self._session_id = session_id
        self._engine_name = engine_name
        self._default_name = None
        self._default_namespace = None

    def create(self):
        raise NotImplementedError()

    def create_table(self, address, name, namespace, partitions=None, **kwargs):
        table = self.table(address=address, name=name, namespace=namespace, partitions=partitions, **kwargs)
        table_meta = StorageTableMeta(name=name, namespace=namespace, new=True)
        table_meta.set_metas(**kwargs)
        table_meta.address = table.get_address()
        table_meta.partitions = table.get_partitions()
        table_meta.engine = table.get_engine()
        table_meta.type = table.get_type()
        table_meta.options = table.get_options()
        table_meta.create()
        table.set_meta(table_meta)
        # update count on meta
        # table.count()
        return table

    def set_default(self, name, namespace):
        self._default_name = name
        self._default_namespace = namespace

    def get_table(self, name=None, namespace=None):
        if not name or not namespace:
            name = self._default_name
            namespace = self._default_namespace
        meta = StorageTableMeta(name=name, namespace=namespace)
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
    @DB.connection_context()
    def get_storage_info(cls, name, namespace):
        metas = StorageTableMetaModel.select().where(StorageTableMetaModel.f_name == name,
                                                     StorageTableMetaModel.f_namespace == namespace)
        if metas:
            meta = metas[0]
            engine = meta.f_engine
            address_dict = meta.f_address
            address = StorageTableMeta.create_address(storage_engine=engine, address_dict=address_dict)
            partitions = meta.f_partitions
            return engine, address, partitions
        else:
            return None, None, None

    def __enter__(self):
        with DB.connection_context():
            session_record = SessionRecord()
            session_record.f_session_id = self._session_id
            session_record.f_engine_name = self._engine_name
            session_record.f_engine_type = EngineType.STORAGE
            # TODO: engine address
            session_record.f_engine_address = {}
            session_record.f_create_time = current_timestamp()
            rows = session_record.save(force_insert=True)
            if rows != 1:
                raise Exception(f"create session record {self._session_id} failed")
            LOGGER.debug(f"save session {self._session_id} record")
        self.create()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        with DB.connection_context():
            rows = SessionRecord.delete().where(SessionRecord.f_session_id == self._session_id).execute()
            if rows > 0:
                LOGGER.debug(f"delete session {self._session_id} record")
            else:
                LOGGER.warning(f"failed delete session {self._session_id} record")

    def destroy_session(self):
        try:
            self.close()
        except:
            pass
        with DB.connection_context():
            rows = SessionRecord.delete().where(SessionRecord.f_session_id == self._session_id).execute()
            if rows > 0:
                LOGGER.debug(f"delete session {self._session_id} record")
            else:
                LOGGER.warning(f"failed delete session {self._session_id} record")

    @classmethod
    @DB.connection_context()
    def query_expired_sessions_record(cls, ttl) -> [SessionRecord]:
        sessions_record = SessionRecord.select().where(SessionRecord.f_create_time < (current_timestamp() - ttl))
        return [session_record for session_record in sessions_record]

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
