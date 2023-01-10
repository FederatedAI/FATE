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
from typing import Iterable

from .._computing import Address, CSessionABC
from ._table import from_hdfs, from_hive, from_localfs, from_rdd

LOGGER = logging.getLogger(__name__)


class CSession(CSessionABC):
    """
    manage RDDTable
    """

    def __init__(self, session_id):
        self._session_id = session_id

    def load(self, address: Address, partitions, schema, **kwargs):
        from .._address import HDFSAddress

        if isinstance(address, HDFSAddress):
            table = from_hdfs(
                paths=f"{address.name_node}/{address.path}",
                partitions=partitions,
                in_serialized=kwargs.get("in_serialized", True),
                id_delimiter=kwargs.get("id_delimiter", ","),
            )
            table.schema = schema
            return table

        from .._address import HiveAddress, LinkisHiveAddress

        if isinstance(address, (HiveAddress, LinkisHiveAddress)):
            table = from_hive(
                tb_name=address.name,
                db_name=address.database,
                partitions=partitions,
            )
            table.schema = schema
            return table

        from .._address import LocalFSAddress

        if isinstance(address, LocalFSAddress):
            table = from_localfs(
                paths=address.path,
                partitions=partitions,
                in_serialized=kwargs.get("in_serialized", True),
                id_delimiter=kwargs.get("id_delimiter", ","),
            )
            table.schema = schema
            return table

        raise NotImplementedError(f"address type {type(address)} not supported with spark backend")

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
