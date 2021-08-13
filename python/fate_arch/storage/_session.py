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
import os.path
import typing

from fate_arch.abc import StorageSessionABC, StorageTableABC, CTableABC
from fate_arch.common import EngineType, engine_utils
from fate_arch.common.log import getLogger
from fate_arch.storage._table import StorageTableMeta
from fate_arch.storage._types import StorageEngine, EggRollStoreType, StandaloneStoreType
from fate_arch.relation_ship import Relationship
from fate_arch.storage.metastore.db_models import DB, StorageTableMetaModel
from fate_arch.common.base_utils import current_timestamp

MAX_NUM = 10000

LOGGER = getLogger()


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
        table_meta.store_type = table.get_store_type()
        table_meta.options = table.get_options()
        table_meta.create()
        table.meta = table_meta
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
                               store_type=meta.get_store_type(),
                               options=meta.get_options())
            table.meta = meta
            return table
        else:
            return None

    def table(self, name, namespace, address, partitions, store_type=None, options=None, **kwargs) -> StorageTableABC:
        raise NotImplementedError()

    @classmethod
    def persistent(cls, computing_table: CTableABC, table_namespace, table_name, schema=None, engine=None, engine_address=None, store_type=None, token: typing.Dict = None) -> StorageTableMeta:
        if engine:
            if engine not in Relationship.Computing.get(computing_table.engine, {}).get(EngineType.STORAGE, {}).get("support", []):
                raise Exception(f"storage engine {engine} not supported with computing engine {computing_table.engine}")
        else:
            engine = Relationship.Computing.get(computing_table.engine, {}).get(EngineType.STORAGE, {}).get("default", None)
            if not engine:
                raise Exception(f"can not found {computing_table.engine} default storage engine")
        if engine_address is None:
            # find engine address from service_conf.yaml
            engine_address = engine_utils.get_engines_config_from_conf().get(EngineType.STORAGE, {}).get(engine, None)
        if engine_address is None:
            raise Exception("no engine address")
        address_dict = engine_address.copy()
        partitions = computing_table.partitions
        if engine == StorageEngine.EGGROLL:
            address_dict.update({"name": table_name, "namespace": table_namespace})
            store_type = EggRollStoreType.ROLLPAIR_LMDB if store_type is None else store_type
        elif engine == StorageEngine.STANDALONE:
            address_dict.update({"name": table_name, "namespace": table_namespace})
            store_type = StandaloneStoreType.ROLLPAIR_LMDB if store_type is None else store_type
        elif engine == StorageEngine.HIVE:
            address_dict.update({"name": table_name, "database": table_namespace})
        elif engine == StorageEngine.LINKIS_HIVE:
            address_dict.update({"database": None, "name": f"{table_namespace}_{table_name}", "username": token.get("username", "")})
        elif engine == StorageEngine.HDFS:
            address_dict.update({"path": os.path.join(address_dict.get("path_prefix", ""), table_namespace, table_name)})
        else:
            raise RuntimeError(f"{engine} storage is not supported")
        address = StorageTableMeta.create_address(storage_engine=engine, address_dict=address_dict)
        schema = schema if schema else {}
        computing_table.save(address, schema=schema, partitions=partitions, store_type=store_type)
        part_of_data = []
        part_of_limit = 100
        for k, v in computing_table.collect():
            part_of_data.append((k, v))
            part_of_limit -= 1
            if part_of_limit == 0:
                break
        table_count = computing_table.count()
        table_meta = StorageTableMeta(name=table_name, namespace=table_namespace, new=True)
        table_meta.address = address
        table_meta.partitions = computing_table.partitions
        table_meta.engine = engine
        table_meta.store_type = store_type
        table_meta.schema = schema
        table_meta.part_of_data = part_of_data
        table_meta.count = table_count
        table_meta.write_access_time = current_timestamp()
        table_meta.create()
        return table_meta

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
        self.create()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.destroy()

    def destroy(self):
        try:
            self.stop()
        except Exception as e:
            LOGGER.warning(f"stop storage session {self._session_id} failed, try to kill", e)
            self.kill()

    def stop(self):
        raise NotImplementedError()

    def kill(self):
        raise NotImplementedError()

    @property
    def session_id(self):
        return self._session_id
