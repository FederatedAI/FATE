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
from collections.abc import Iterable
from typing import Optional

from fate.arch.abc import CSessionABC

from ..._standalone import Session
from ...unify import URI, generate_computing_uuid, uuid
from ._table import Table

LOGGER = logging.getLogger(__name__)


class CSession(CSessionABC):
    def __init__(
        self, session_id: Optional[str] = None, logger_config: Optional[dict] = None, options: Optional[dict] = None
    ):
        if session_id is None:
            session_id = generate_computing_uuid()
        if options is None:
            options = {}
        max_workers = options.get("task_cores", None)
        self._session = Session(session_id, max_workers=max_workers, logger_config=logger_config)

    def get_standalone_session(self):
        return self._session

    @property
    def session_id(self):
        return self._session.session_id

    def load(self, uri: URI, schema: dict, options: dict = None):
        if uri.scheme != "standalone":
            raise ValueError(f"uri scheme `{uri.scheme}` not supported with standalone backend")
        try:
            *database, namespace, name = uri.path_splits()
        except Exception as e:
            raise ValueError(f"uri `{uri}` not valid, demo format: standalone://database_path/namespace/name") from e

        raw_table = self._session.load(name=name, namespace=namespace)
        partitions = raw_table.partitions
        raw_table = raw_table.save_as(
            name=f"{name}_{uuid()}",
            namespace=namespace,
            partition=partitions,
            need_cleanup=True,
        )
        table = Table(raw_table)
        table.schema = schema
        return table

    def parallelize(self, data: Iterable, partition: int, include_key: bool, **kwargs):
        table = self._session.parallelize(data=data, partition=partition, include_key=include_key, **kwargs)
        return Table(table)

    def cleanup(self, name, namespace):
        return self._session.cleanup(name=name, namespace=namespace)

    def stop(self):
        return self._session.stop()

    def kill(self):
        return self._session.kill()

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
