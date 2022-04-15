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

from fate_arch.abc import StorageSessionABC, CTableABC
from fate_arch.common import EngineType, engine_utils
from fate_arch.common.data_utils import default_output_fs_path
from fate_arch.common.log import getLogger
from fate_arch.storage._table import StorageTableMeta
from fate_arch.storage._types import StorageEngine, EggRollStoreType, StandaloneStoreType, HDFSStoreType, HiveStoreType, \
    LinkisHiveStoreType, LocalFSStoreType, PathStoreType, StorageTableOrigin
from fate_arch.relation_ship import Relationship
from fate_arch.common.base_utils import current_timestamp


LOGGER = getLogger()


class StorageSessionBase(StorageSessionABC):
    def __init__(self, session_id, engine):
        self._session_id = session_id
        self._engine = engine

    def create_table(self, address, name, namespace, partitions=None, **kwargs):
        table = self.table(address=address, name=name, namespace=namespace, partitions=partitions, **kwargs)
        table.create_meta(**kwargs)
        return table

    def get_table(self, name, namespace):
        meta = StorageTableMeta(name=name, namespace=namespace)
        if meta and meta.exists():
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

    @classmethod
    def get_table_meta(cls, name, namespace):
        meta = StorageTableMeta(name=name, namespace=namespace)
        if meta and meta.exists():
            return meta
        else:
            return None

    @classmethod
    def persistent(cls, computing_table: CTableABC, namespace, name, schema=None,
                   part_of_data=None, engine=None, engine_address=None,
                   store_type=None, token: typing.Dict = None) -> StorageTableMeta:
        if engine:
            if engine != StorageEngine.PATH and engine not in Relationship.Computing.get(
                    computing_table.engine, {}).get(EngineType.STORAGE, {}).get("support", []):
                raise Exception(f"storage engine {engine} not supported with computing engine {computing_table.engine}")
        else:
            engine = Relationship.Computing.get(
                computing_table.engine,
                {}).get(
                EngineType.STORAGE,
                {}).get(
                "default",
                None)
            if not engine:
                raise Exception(f"can not found {computing_table.engine} default storage engine")
        if engine_address is None:
            # find engine address from service_conf.yaml
            engine_address = engine_utils.get_engines_config_from_conf().get(EngineType.STORAGE, {}).get(engine, {})
        address_dict = engine_address.copy()
        partitions = computing_table.partitions

        if engine == StorageEngine.STANDALONE:
            address_dict.update({"name": name, "namespace": namespace})
            store_type = StandaloneStoreType.ROLLPAIR_LMDB if store_type is None else store_type

        elif engine == StorageEngine.EGGROLL:
            address_dict.update({"name": name, "namespace": namespace})
            store_type = EggRollStoreType.ROLLPAIR_LMDB if store_type is None else store_type

        elif engine == StorageEngine.HIVE:
            address_dict.update({"database": namespace, "name": f"{name}"})
            store_type = HiveStoreType.DEFAULT if store_type is None else store_type

        elif engine == StorageEngine.LINKIS_HIVE:
            address_dict.update({"database": None, "name": f"{namespace}_{name}",
                                 "username": token.get("username", "")})
            store_type = LinkisHiveStoreType.DEFAULT if store_type is None else store_type

        elif engine == StorageEngine.HDFS:
            if not address_dict.get("path"):
                address_dict.update({"path": default_output_fs_path(
                    name=name, namespace=namespace, prefix=address_dict.get("path_prefix"))})
            store_type = HDFSStoreType.DISK if store_type is None else store_type

        elif engine == StorageEngine.LOCALFS:
            if not address_dict.get("path"):
                address_dict.update({"path": default_output_fs_path(
                    name=name, namespace=namespace, storage_engine=StorageEngine.LOCALFS)})
            store_type = LocalFSStoreType.DISK if store_type is None else store_type

        elif engine == StorageEngine.PATH:
            store_type = PathStoreType.PICTURE if store_type is None else store_type

        else:
            raise RuntimeError(f"{engine} storage is not supported")
        address = StorageTableMeta.create_address(storage_engine=engine, address_dict=address_dict)
        schema = schema if schema else {}
        computing_table.save(address, schema=schema, partitions=partitions, store_type=store_type)
        table_count = computing_table.count()
        table_meta = StorageTableMeta(name=name, namespace=namespace, new=True)
        table_meta.address = address
        table_meta.partitions = computing_table.partitions
        table_meta.engine = engine
        table_meta.store_type = store_type
        table_meta.schema = schema
        table_meta.part_of_data = part_of_data if part_of_data else {}
        table_meta.count = table_count
        table_meta.write_access_time = current_timestamp()
        table_meta.origin = StorageTableOrigin.OUTPUT
        table_meta.create()
        return table_meta

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.destroy()

    def destroy(self):
        try:
            self.stop()
        except Exception as e:
            LOGGER.warning(f"stop storage session {self._session_id} failed, try to kill", e)
            self.kill()

    def table(self, name, namespace, address, store_type, partitions=None, **kwargs):
        raise NotImplementedError()

    def stop(self):
        raise NotImplementedError()

    def kill(self):
        raise NotImplementedError()

    @property
    def session_id(self):
        return self._session_id

    @property
    def engine(self):
        return self._engine
