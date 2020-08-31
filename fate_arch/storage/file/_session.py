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

from fate_arch.abc import AddressABC
from fate_arch.common.address import FileAddress
from fate_arch.common.profile import log_elapsed
from fate_arch.storage import StorageSessionBase


class StorageSession(StorageSessionBase):
    def __init__(self, session_id, options=None):
        super(StorageSession, self).__init__(session_id=session_id)
        self.options = options

    def create(self):
        pass

    def table(self, address: AddressABC, name, namespace, partitions, storage_type=None, options=None, **kwargs):
        if isinstance(address, FileAddress):
            from fate_arch.storage.file._table import StorageTable
            if not options:
                options = self.options
            return StorageTable(address=address, name=name, namespace=namespace, partitions=partitions, storage_type=storage_type, options=options)
        raise NotImplementedError(f"address type {type(address)} not supported with hdfs storage")

    @log_elapsed
    def cleanup(self, name, namespace):
        pass

    @log_elapsed
    def stop(self):
        pass

    @log_elapsed
    def kill(self):
        pass
