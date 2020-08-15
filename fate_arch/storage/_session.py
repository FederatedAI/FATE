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
from fate_arch.common.core_utils import current_timestamp, serialize_b64, deserialize_b64, fate_uuid
from fate_arch.metastore.db_models import DB, StorageTableMeta
from fate_arch.computing import ComputingEngine
from fate_arch.abc import StorageSessionABC, StorageTableABC

MAX_NUM = 10000

LOGGER = getLogger()


class Session(object):
    @classmethod
    def build(cls, session_id=None, storage_engine=None, **kwargs):
        session_id = session_id if session_id else fate_uuid()
        if storage_engine == StorageEngine.EGGROLL:
            from fate_arch.storage.eggroll import StorageSession
            return StorageSession(session_id=session_id, options=kwargs.get("options", {}))
        else:
            return StorageSessionBase()

    @classmethod
    def storage_engine_for_computing(cls):
        pass


class StorageSessionBase(StorageSessionABC):
    def create(self):
        raise NotImplementedError()

    def create_table(self, address, name, namespace, partitions=1, **kwargs):
        with DB.connection_context():
            metas = StorageTableMeta.select().where(StorageTableMeta.f_name == name,
                                                    StorageTableMeta.f_namespace == namespace)
            if metas:
                    raise Exception('table {} {} has been created by storage engine {} '.format(name, namespace,
                                                                                                metas.f_engine))
            else:
                table = self.table(address=address, name=name, namespace=namespace, partitions=partitions, **kwargs)
                meta = StorageTableMeta()
                meta.f_create_time = current_timestamp()
                meta.f_name = name
                meta.f_namespace = namespace
                meta.f_address = address.__dict__ if address else {}
                meta.f_engine = table.get_storage_engine()
                meta.f_type = table.get_storage_type()
                meta.f_partitions = partitions
                meta.f_options = serialize_b64(table.get_options())
                meta.f_count = table.get_options().get("count", None)
                meta.f_schema = serialize_b64({}, to_str=True)
                meta.f_part_of_data = serialize_b64([], to_str=True)
            meta.f_update_time = current_timestamp()
            meta.save(force_insert=True)
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
                                  options=(deserialize_b64(meta.f_options) if meta.f_options else None))
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

    def convert(self, src_table, dest_name, dest_namespace, session_id, computing_engine: ComputingEngine, force=False, **kwargs):
        partitions = src_table.get_partitions()
        if src_table.get_storage_engine() not in Relationship.CompToStore.get(computing_engine, []):
            if computing_engine == ComputingType.STANDALONE:
                from fate_arch.storage.standalone._table import StorageTable
                address = self.create_table(name=dest_name, namespace=dest_namespace, storage_engine=StorageEngine.STANDALONE, partitions=partitions)
                _table = StorageTable(session_id, storage_type=StorageEngine.LMDB, namespace=dest_namespace, name=dest_name,
                                      address=address, partitions=partitions)
            elif computing_engine == ComputingType.EGGROLL:
                from fate_arch.storage.eggroll._table import StorageTable
                address = self.create_table(name=dest_name, namespace=dest_namespace, storage_engine=StorageEngine.EGGROLL, partitions=partitions)
                _table = StorageTable(session_id=session_id, address=address, partitions=partitions, name=dest_name, namespace=dest_namespace)
            elif computing_engine == ComputingType.SPARK:
                from fate_arch.storage.hdfs._table import StorageTable
                address = self.create_table(name=dest_name, namespace=dest_namespace, storage_engine=StorageEngine.HDFS, partitions=partitions)
                _table = StorageTable(address=address, partitions=partitions, name=dest_name, namespace=dest_namespace)
            else:
                raise RuntimeError("can not convert table")
            self.copy_table(src_table, _table)
            return _table

    def copy_table(self, src_table, dest_table):
        count = 0
        data = []
        party_of_data = []
        for k, v in src_table.collect():
            data.append((k, v))
            count += 1
            if count < 100:
                party_of_data.append((k, v))
            if len(data) == MAX_NUM:
                dest_table.put_all(data)
                data = []
        if data:
            dest_table.put_all(data)
        dest_table.update_metas(schema=src_table.get_meta(meta_type="schema"), count=src_table.count(), party_of_data=party_of_data)

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
