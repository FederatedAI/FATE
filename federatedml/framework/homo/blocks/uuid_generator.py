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
import hashlib

from federatedml.framework.homo.blocks.base import _BlockBase, TransferInfo, HomoTransferBase


class UUIDTransferVariable(HomoTransferBase):
    def __init__(self, info: TransferInfo = None):
        super().__init__(info)
        self.uuid = self.create_server_to_client_variable(name="uuid")


class Server(_BlockBase):

    def __init__(self, transfer_variable: UUIDTransferVariable = UUIDTransferVariable()):
        super().__init__(transfer_variable)
        self._uuid_transfer = transfer_variable.uuid
        self._uuid_set = set()
        self._ind = -1
        self._parties = self._uuid_transfer.roles_to_parties(self._uuid_transfer.authorized_dst_roles)

    # noinspection PyUnusedLocal
    @staticmethod
    def generate_id(ind, *args, **kwargs):
        return hashlib.md5(f"{ind}".encode("ascii")).hexdigest()

    def _next_uuid(self):
        while True:
            self._ind += 1
            uid = Server.generate_id(self._ind)
            if uid in self._uuid_set:
                continue
            self._uuid_set.add(uid)
            return uid

    def validate_uuid(self):
        for party in self._parties:
            uid = self._next_uuid()
            self._uuid_transfer.remote_parties(obj=uid, parties=[party])


class Client(_BlockBase):

    def __init__(self, transfer_variable: UUIDTransferVariable = UUIDTransferVariable()):
        super().__init__(transfer_variable)
        self._uuid_variable = transfer_variable.uuid

    def generate_uuid(self):
        uid = self._uuid_variable.get(0)
        return uid
