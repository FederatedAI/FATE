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


from fate_arch.common.log import getLogger
from fate_arch.storage._types import StorageEngine, Relationship
from fate_arch.common.base_utils import current_timestamp, serialize_b64, deserialize_b64, fate_uuid
from fate_arch.storage.metastore.db_models import DB, StorageTableMeta
from fate_arch.computing import ComputingEngine
from fate_arch.abc import StorageSessionABC, StorageTableABC
from fate_arch.storage._table import StorageTableBase
from fate_arch.storage import StorageTableMetaType
from fate_arch.common import Backend, WorkMode

MAX_NUM = 10000

LOGGER = getLogger()


class Session(object):
    @classmethod
    def build(cls, session_id=None, storage_engine=None, computing_engine=None, **kwargs):
        session_id = session_id if session_id else fate_uuid()
        if storage_engine is None and kwargs.get("name") and kwargs.get("namespace"):
            tables_meta = StorageTableBase().get_metas(filter_fields=dict(name=kwargs.get("name"), namespace=kwargs.get("namespace")), query_fields=[StorageTableMetaType.ENGINE])
            if tables_meta:
                storage_engine = tables_meta[0].f_engine
        if storage_engine is None and computing_engine is None:
            if kwargs.get("work_mode") and kwargs.get("backend"):
                # Compatible with previous 1.5 versions
                backend = kwargs.get("backend")
                if backend == Backend.EGGROLL:
                    computing_engine = ComputingEngine.EGGROLL if kwargs.get("work_mode") == WorkMode.CLUSTER else ComputingEngine.STANDALONE
                elif backend == Backend.SPARK:
                    computing_engine = ComputingEngine.SPARK
        if storage_engine is None and computing_engine:
            # Gets the computing engine default storage engine
            storage_engine = Relationship.CompToStore.get(computing_engine)[0]
        if storage_engine == StorageEngine.EGGROLL:
            from fate_arch.storage.eggroll import StorageSession
            return StorageSession(session_id=session_id, options=kwargs.get("options", {}))
        else:
            return StorageSessionBase()

    @classmethod
    def convert(cls, session_id, src_name, src_namespace, dest_name, dest_namespace, computing_engine: ComputingEngine = ComputingEngine.EGGROLL, force=False, **kwargs):
        # The source and target may be different session types
        src_session_id = f"{session_id}_convert_src"
        src_session = cls.build(session_id=src_session_id, name=src_name, namespace=src_namespace)
        src_table = src_session.get_table(name=src_name, namespace=src_namespace)
        if not src_table:
            raise RuntimeError(f"can not found table name: {src_name} namespace: {src_namespace}")
        if src_table.get_storage_engine() not in Relationship.CompToStore.get(computing_engine, []):
            dest_session_id = f"{session_id}_convert_dest"
            if computing_engine == ComputingEngine.STANDALONE:
                pass
            elif computing_engine == ComputingEngine.EGGROLL:
                from fate_arch.storage.eggroll import StorageSession
                dest_session = StorageSession(session_id=dest_session_id, options=kwargs.get("options", {}))
                dest_address = StorageSessionBase().get_address(storage_engine=StorageEngine.EGGROLL, address_dict=dict(name=dest_name, namespace=dest_namespace))
                dest_table = dest_session.create_table(address=dest_address, name=dest_name, namespace=dest_namespace, partitions=src_table.get_partitions(), **kwargs)
            elif computing_engine == ComputingEngine.SPARK:
                pass
            else:
                raise RuntimeError(f"can not support computing engine {computing_engine}")
            with src_session:
                with dest_session:
                    cls.copy_table(src_table, dest_table)
            return src_session, src_table, dest_session, dest_table, True
        else:
            return src_session, src_table, src_session, src_table, False

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
        dest_table.update_metas(schema=src_table.get_meta(meta_type="schema"), count=src_table.count(), part_of_data=part_of_data)


class StorageSessionBase(StorageSessionABC):
    def create(self):
        raise NotImplementedError()

    def create_table(self, address, name, namespace, partitions=1, **kwargs):
        table = self.table(address=address, name=name, namespace=namespace, partitions=partitions, **kwargs)
        table.create_metas()
        return table

    def get_table(self, name, namespace):
        with DB.connection_context():
            metas = StorageTableMeta.select().where(StorageTableMeta.f_name == name,
                                                    StorageTableMeta.f_namespace == namespace)
            if metas:
                meta = metas[0]
                return self.table(address=self.get_address(storage_engine=meta.f_engine, address_dict=meta.f_address),
                                  name=meta.f_name,
                                  namespace=meta.f_namespace,
                                  partitions=meta.f_partitions,
                                  storage_type=meta.f_type,
                                  options=meta.f_options)
            else:
                return None

    def table(self, address, name, namespace, partitions, storage_type=None, options=None, **kwargs) -> StorageTableABC:
        raise NotImplementedError()

    def get_storage_info(self, name, namespace):
        with DB.connection_context():
            metas = StorageTableMeta.select().where(StorageTableMeta.f_name == name,
                                                    StorageTableMeta.f_namespace == namespace)
            if metas:
                meta = metas[0]
                engine = meta.f_engine
                address_dict = meta.f_address
                address = self.get_address(storage_engine=engine, address_dict=address_dict)
                partitions = meta.f_partitions
            else:
                return None, None, None
        return engine, address, partitions

    def get_address(self, storage_engine, address_dict):
        return Relationship.EngineToAddress.get(storage_engine)(*address_dict)

    def __enter__(self):
        self.create()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        try:
            self.stop()
        except Exception as e:
            self.kill()

    def stop(self):
        raise NotImplementedError()

    def kill(self):
        raise NotImplementedError()
