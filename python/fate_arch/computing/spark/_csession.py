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

from typing import Iterable

from fate_arch.abc import AddressABC
from fate_arch.abc import CSessionABC
from fate_arch.computing.spark._table import from_hdfs, from_rdd, from_local_file
from fate_arch.common import log

LOGGER = log.getLogger()


class CSession(CSessionABC):
    """
    manage RDDTable
    """

    def __init__(self, session_id):
        self._session_id = session_id

    def load(self, address: AddressABC, partitions, schema, **kwargs):
        from fate_arch.common.address import HDFSAddress
        if isinstance(address, HDFSAddress):
            table = from_hdfs(
                paths=f"{address.name_node}/{address.path}", partitions=partitions)
            table.schema = schema
            return table

        from fate_arch.common.address import FileAddress
        if isinstance(address, FileAddress):
            return address

        from fate_arch.common.address import PathAddress
        if isinstance(address, PathAddress):
            table = from_local_file(path=address.path, partitions=partitions)
            table.schema = schema
            return table

        raise NotImplementedError(
            f"address type {type(address)} not supported with spark backend")

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
