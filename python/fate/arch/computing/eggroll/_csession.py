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

from fate.arch.abc import CSessionABC

from ...unify import URI, uuid
from .._profile import computing_profile
from ._table import Table

try:
    from eggroll.core.session import session_init
    from eggroll.roll_pair.roll_pair import runtime_init
except ImportError:
    raise EnvironmentError("eggroll not found in pythonpath")

LOGGER = logging.getLogger(__name__)


class CSession(CSessionABC):
    def __init__(self, session_id, options: dict = None):
        if options is None:
            options = {}
        if "eggroll.session.deploy.mode" not in options:
            options["eggroll.session.deploy.mode"] = "cluster"
        if "eggroll.rollpair.inmemory_output" not in options:
            options["eggroll.rollpair.inmemory_output"] = True
        self._rp_session = session_init(session_id=session_id, options=options)
        self._rpc = runtime_init(session=self._rp_session)
        self._session_id = self._rp_session.get_session_id()

    def get_rpc(self):
        return self._rpc

    @property
    def session_id(self):
        return self._session_id

    @computing_profile
    def load(self, uri: URI, schema: dict, options: dict = None) -> Table:
        from ._type import EggRollStoreType

        if uri.scheme != "eggroll":
            raise ValueError(f"uri scheme {uri.scheme} not supported with eggroll backend")
        try:
            _, namespace, name = uri.path_splits()
        except Exception as e:
            raise ValueError(f"uri {uri} not valid, demo format: eggroll:///namespace/name") from e

        if options is None:
            options = {}
        if "store_type" not in options:
            options["store_type"] = EggRollStoreType.ROLLPAIR_LMDB
        options["create_if_missing"] = False
        rp = self._rpc.load(namespace=namespace, name=name, options=options)
        if rp is None or rp.get_partitions() == 0:
            raise RuntimeError(f"no exists: {name}, {namespace}")

        if options["store_type"] != EggRollStoreType.ROLLPAIR_IN_MEMORY:
            rp = rp.save_as(
                name=f"{name}_{uuid()}",
                namespace=self.session_id,
                partition=rp.get_partitions(),
                options={"store_type": EggRollStoreType.ROLLPAIR_IN_MEMORY},
            )

            table = Table(rp=rp)
            table.schema = schema
            return table

    @computing_profile
    def parallelize(self, data, partition: int, include_key: bool, **kwargs) -> Table:
        options = dict()
        options["total_partitions"] = partition
        options["include_key"] = include_key
        rp = self._rpc.parallelize(data=data, options=options)
        return Table(rp)

    def cleanup(self, name, namespace):
        self._rpc.cleanup(name=name, namespace=namespace)

    def stop(self):
        return self._rp_session.stop()

    def kill(self):
        return self._rp_session.kill()

    def destroy(self):
        try:
            LOGGER.info(f"clean table namespace {self.session_id}")
            self.cleanup(namespace=self.session_id, name="*")
        except Exception:
            LOGGER.warning(f"no found table namespace {self.session_id}")

        try:
            self.stop()
        except Exception as e:
            LOGGER.warning(f"stop storage session {self.session_id} failed, try to kill", e)
            self.kill()
