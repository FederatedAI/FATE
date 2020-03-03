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

from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.util import consts


class UUIDTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.uuid = self.create_server_to_client_variable(name="uuid")


class Server(object):

    def __init__(self, trans_var: UUIDTransVar = UUIDTransVar()):
        self._uuid_transfer = trans_var.uuid
        self._uuid_set = set()
        self._ind = -1
        self.client_parties = trans_var.client_parties

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
        for party in self.client_parties:
            uid = self._next_uuid()
            self._uuid_transfer.remote_parties(obj=uid, parties=[party])


class Client(object):

    def __init__(self, trans_var: UUIDTransVar = UUIDTransVar()):
        self._uuid_variable = trans_var.uuid
        self._server_parties = trans_var.server_parties

    def generate_uuid(self):
        uid = self._uuid_variable.get_parties(parties=self._server_parties)[0]
        return uid
