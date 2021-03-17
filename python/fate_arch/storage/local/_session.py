########################################################
# Copyright 2020-2021 program was created VMware, Inc. #
# SPDX-License-Identifier: Apache-2.0                  #
########################################################

from fate_arch.storage import StorageSessionBase, StorageEngine
from fate_arch.abc import AddressABC
from fate_arch.common.address import PathAddress


class StorageSession(StorageSessionBase):
    def __init__(self, session_id, options=None):
        super(StorageSession, self).__init__(session_id=session_id, engine_name=StorageEngine.LOCAL)

    def create(self):
        pass

    def table(self, address: AddressABC, name, namespace, partitions, storage_type=None, options=None, **kwargs):
        if isinstance(address, PathAddress):
            from fate_arch.storage.local._table import StorageTable
            return StorageTable(address=address, name=name, namespace=namespace,
                                partitions=partitions, storage_type=storage_type, options=options)
        raise NotImplementedError(f"address type {type(address)} not supported with hdfs storage")

    def cleanup(self, name, namespace):
        pass

    def stop(self):
        pass

    def kill(self):
        pass