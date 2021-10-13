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
from fate_arch.session import get_parties
from federatedml.transfer_variable.base_transfer_variable import Variable, BaseTransferVariables
from federatedml.util import consts


class HomoTransferBase(BaseTransferVariables):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__()
        if prefix is None:
            self.prefix = f"{self.__class__.__module__}.{self.__class__.__name__}."
        else:
            self.prefix = f"{prefix}{self.__class__.__name__}."
        self.server = server
        self.clients = clients

    def create_client_to_server_variable(self, name):
        name = f"{self.prefix}{name}"
        return Variable.get_or_create(name, lambda: Variable(name, self.clients, self.server))

    def create_server_to_client_variable(self, name):
        name = f"{self.prefix}{name}"
        return Variable.get_or_create(name, lambda: Variable(name, self.server, self.clients))

    @staticmethod
    def get_parties(roles):
        return get_parties().roles_to_parties(roles=roles)

    @property
    def client_parties(self):
        return self.get_parties(roles=self.clients)

    @property
    def server_parties(self):
        return self.get_parties(roles=self.server)
