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
from arch.api import RuntimeInstance
from federatedml.transfer_variable.base_transfer_variable import Variable, BaseTransferVariables
from federatedml.util import consts


class HomoTransferBase(BaseTransferVariables):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__()
        if prefix is None:
            self.prefix = self.__class__.__name__
        else:
            self.prefix = f"{prefix}.{self.__class__.__name__}"
        self.server = server
        self.clients = clients

    def create_client_to_server_variable(self, name):
        return Variable(name=f"{self.prefix}.{name}", src=self.clients, dst=self.server)

    def create_server_to_client_variable(self, name):
        return Variable(name=f"{self.prefix}.{name}", src=self.server, dst=self.clients)

    @staticmethod
    def get_parties(roles):
        return RuntimeInstance.FEDERATION.roles_to_parties(roles=roles)

    @property
    def client_parties(self):
        return self.get_parties(roles=self.clients)

    @property
    def server_parties(self):
        return self.get_parties(roles=self.server)
