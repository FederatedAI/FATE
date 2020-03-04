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
from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.util import consts


class HasConvergedTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.has_converged = self.create_server_to_client_variable(name="has_converged")


class Server(object):

    def __init__(self, trans_var: HasConvergedTransVar = HasConvergedTransVar()):
        self._broadcaster = trans_var.has_converged
        self._client_parties = trans_var.client_parties

    def remote_converge_status(self, is_converge, suffix=tuple()):
        self._broadcaster.remote_parties(obj=is_converge, parties=self._client_parties, suffix=suffix)
        return is_converge


class Client(object):
    def __init__(self, trans_var: HasConvergedTransVar = HasConvergedTransVar()):
        self._broadcaster = trans_var.has_converged
        self._server_parties = trans_var.server_parties

    def get_converge_status(self, suffix=tuple()):
        return self._broadcaster.get_parties(parties=self._server_parties, suffix=suffix)[0]
