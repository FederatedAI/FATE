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
import os
import shutil
import traceback

from fate_arch.common import file_utils
from fate_arch.storage import StorageSessionBase, StorageEngine
from fate_arch.abc import AddressABC
from fate_arch.common.address import ApiAddress


class StorageSession(StorageSessionBase):
    def __init__(self, session_id, options=None):
        super(StorageSession, self).__init__(session_id=session_id, engine=StorageEngine.PATH)
        self.base_dir = os.path.join(file_utils.get_project_base_directory(), "api_data", session_id)

    def table(self, address: AddressABC, name, namespace, partitions, store_type=None, options=None, **kwargs):
        if isinstance(address, ApiAddress):
            from fate_arch.storage.api._table import StorageTable
            return StorageTable(path=os.path.join(self.base_dir, namespace, name),
                                address=address,
                                name=name,
                                namespace=namespace,
                                partitions=partitions, store_type=store_type, options=options)
        raise NotImplementedError(f"address type {type(address)} not supported with api storage")

    def cleanup(self, name, namespace):
        # path = os.path.join(self.base_dir, namespace, name)
        # try:
        #     os.remove(path)
        # except Exception as e:
        #     traceback.print_exc()
        pass

    def stop(self):
        # try:
        #     shutil.rmtree(self.base_dir)
        # except Exception as e:
        #     traceback.print_exc()
        pass

    def kill(self):
        # try:
        #     shutil.rmtree(self.base_dir)
        # except Exception as e:
        #     traceback.print_exc()
        pass
