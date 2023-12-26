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

import logging
from typing import Optional

from fate.arch.computing.api import KVTableContext, generate_computing_uuid
from fate.arch.unify import URI, uuid
from ._standalone import Session
from ._table import Table

logger = logging.getLogger(__name__)


class CSession(KVTableContext):
    def __init__(
        self,
        session_id: Optional[str] = None,
        data_dir=None,
        logger_config: Optional[dict] = None,
        options: Optional[dict] = None,
    ):
        if session_id is None:
            session_id = generate_computing_uuid()
        if data_dir is None:
            raise ValueError("data_dir is None")
        if options is None:
            options = {}
        max_workers = options.get("task_cores", None)
        self._session = Session(session_id, data_dir=data_dir, max_workers=max_workers, logger_config=logger_config)

    def get_standalone_session(self):
        return self._session

    @property
    def session_id(self):
        return self._session.session_id

    def _load(
        self,
        uri: URI,
        schema: dict,
        options: dict,
    ):
        if uri.scheme != "standalone":
            raise ValueError(f"uri scheme `{uri.scheme}` not supported with standalone backend")
        try:
            *database, namespace, name = uri.path_splits()
        except Exception as e:
            raise ValueError(f"uri `{uri}` not valid, demo format: standalone://database_path/namespace/name") from e

        raw_table = self._session.load(name=name, namespace=namespace)
        raw_table = raw_table.copy_as(
            name=f"{name}_{uuid()}",
            namespace=namespace,
            need_cleanup=True,
        )
        table = Table(raw_table)
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
        table = self._session.parallelize(
            data=data,
            partition=total_partitions,
            partitioner=partitioner,
            key_serdes_type=key_serdes_type,
            value_serdes_type=value_serdes_type,
            partitioner_type=partitioner_type,
        )
        return Table(table)

    def _info(self, level=0):
        if level == 0:
            return f"Standalone<session_id={self.session_id}, max_workers={self._session.max_workers}, data_dir={self._session.data_dir}>"

        elif level == 1:
            import inspect

            return {
                "session_id": self.session_id,
                "data_dir": self._session.data_dir,
                "max_workers": self._session.max_workers,
                "code_path": inspect.getfile(self._session.__class__),
            }

    def cleanup(self, name, namespace):
        return self._session.cleanup(name=name, namespace=namespace)

    def stop(self):
        return self._session.stop()

    def kill(self):
        return self._session.kill()

    def _destroy(self):
        try:
            logger.debug(f"clean table namespace {self.session_id}")
            self.cleanup(namespace=self.session_id, name="*")
        except:
            logger.warning(f"no found table namespace {self.session_id}")

        try:
            self.stop()
        except Exception as e:
            logger.warning(f"stop storage session {self.session_id} failed, try to kill", e)
            self.kill()
