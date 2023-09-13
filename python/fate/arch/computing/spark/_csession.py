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
import typing
from typing import Iterable

from fate.arch.abc import CSessionABC

from ...unify import URI
from ._table import from_hdfs, from_hive, from_localfs, from_rdd

if typing.TYPE_CHECKING:
    from ._table import Table
LOGGER = logging.getLogger(__name__)


class CSession(CSessionABC):
    """
    manage RDDTable
    """

    def __init__(self, session_id):
        self._session_id = session_id

    def load(self, uri: URI, schema, options: dict = None) -> "Table":
        if not options:
            options = {}
        partitions = options.get("partitions", None)

        if uri.scheme == "hdfs":
            in_serialized = (options.get("in_serialized", True),)
            id_delimiter = (options.get("id_delimiter", ","),)
            table = from_hdfs(
                paths=uri.original_uri,
                partitions=partitions,
                in_serialized=in_serialized,
                id_delimiter=id_delimiter,
            )
            table.schema = schema
            return table

        if uri.scheme == "hive":
            try:
                (path,) = uri.path_splits()
                database_name, table_name = path.split(".")
            except Exception as e:
                raise ValueError(f"invalid hive uri {uri}, demo uri: hive://localhost:10000/database.table") from e
            table = from_hive(
                tb_name=table_name,
                db_name=database_name,
                partitions=partitions,
            )
            table.schema = schema
            return table

        if uri.scheme == "file":
            in_serialized = (options.get("in_serialized", True),)
            id_delimiter = (options.get("id_delimiter", ","),)
            table = from_localfs(
                paths=uri.path,
                partitions=partitions,
                in_serialized=in_serialized,
                id_delimiter=id_delimiter,
            )
            table.schema = schema
            return table

        raise NotImplementedError(f"uri type {uri} not supported with spark backend")

    def parallelize(self, data: Iterable, partition: int, include_key: bool, **kwargs):
        # noinspection PyPackageRequirements
        from pyspark import SparkContext

        _iter = data if include_key else enumerate(data)
        rdd = SparkContext.getOrCreate().parallelize(_iter, partition)
        return from_rdd(rdd)

    @property
    def session_id(self):
        return self._session_id

    def cleanup(self, name, namespace):
        pass

    def stop(self):
        pass

    def kill(self):
        pass

    def destroy(self):
        pass
