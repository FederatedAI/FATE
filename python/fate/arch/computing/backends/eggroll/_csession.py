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

import logging

from fate.arch.computing.api import KVTableContext
from fate.arch.unify import URI, uuid
from ._table import Table

try:
    from eggroll.session import session_init
    from eggroll.computing import runtime_init
except ImportError:
    raise EnvironmentError("eggroll not found in pythonpath")

logger = logging.getLogger(__name__)


class CSession(KVTableContext):
    def __init__(
        self,
        session_id,
        host: str = None,
        port: int = None,
        options: dict = None,
        config=None,
        config_options=None,
        config_properties_file=None,
    ):
        if options is None:
            options = {}
        self._eggroll_session = session_init(
            session_id=session_id,
            host=host,
            port=port,
            options=options,
            config=config,
            config_options=config_options,
            config_properties_file=config_properties_file,
        )
        self._rpc = runtime_init(session=self._eggroll_session)
        self._session_id = self._eggroll_session.get_session_id()

    def get_rpc(self):
        return self._rpc

    @property
    def session_id(self):
        return self._session_id

    def _load(self, uri: URI, schema: dict, options: dict) -> Table:
        from ._type import EggRollStoreType

        if uri.scheme != "eggroll":
            raise ValueError(f"uri scheme {uri.scheme} not supported with eggroll backend")
        try:
            _, namespace, name = uri.path_splits()
        except Exception as e:
            raise ValueError(f"uri {uri} not valid, demo format: eggroll:///namespace/name") from e

        if options is None:
            options = {}
        store_type = options.get("store_type", EggRollStoreType.ROLLPAIR_LMDB)
        rp = self._rpc.load_rp(namespace=namespace, name=name, store_type=store_type)
        if rp is None or rp.get_partitions() == 0:
            raise RuntimeError(f"no exists: {name}, {namespace}")

        if store_type != EggRollStoreType.ROLLPAIR_IN_MEMORY:
            rp = rp.copy_as(
                name=f"{name}_{uuid()}",
                namespace=self.session_id,
                store_type=EggRollStoreType.ROLLPAIR_IN_MEMORY,
            )

            table = Table(rp=rp)
            table.schema = schema
            return table

    def _parallelize(
        self,
        data,
        total_partitions,
        key_serdes,
        key_serdes_type,
        value_serdes,
        value_serdes_type,
        partitioner,
        partitioner_type,
    ):
        rp = self._rpc.parallelize(
            data=data,
            total_partitions=total_partitions,
            partitioner=partitioner,
            partitioner_type=partitioner_type,
            key_serdes_type=key_serdes_type,
            value_serdes_type=value_serdes_type,
        )
        return Table(rp)

    def _info(self):
        if hasattr(self._rpc, "info"):
            return self._rpc.info()
        else:
            return {
                "session_id": self.session_id,
            }

    def cleanup(self, name, namespace):
        self._rpc.cleanup(name=name, namespace=namespace)

    def stop(self):
        return self._eggroll_session.stop()

    def kill(self):
        return self._eggroll_session.kill()

    def _destroy(self):
        try:
            logger.info(f"clean table namespace {self.session_id}")
            self.cleanup(namespace=self.session_id, name="*")
        except Exception:
            logger.warning(f"no found table namespace {self.session_id}")

        try:
            self.stop()
        except Exception as e:
            logger.exception(f"stop storage session {self.session_id} failed, try to kill")
            self.kill()
        else:
            logger.info(f"stop storage session {self.session_id} success")
