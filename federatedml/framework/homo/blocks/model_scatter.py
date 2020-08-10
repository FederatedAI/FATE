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


class ModelScatterTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.client_model = self.create_client_to_server_variable(name="client_model")


class Server(object):

    def __init__(self, trans_var: ModelScatterTransVar = ModelScatterTransVar()):
        self._scatter = trans_var.client_model
        self._client_parties = trans_var.client_parties

    def get_models(self, suffix=tuple()):
        models = self._scatter.get_parties(parties=self._client_parties, suffix=suffix)
        return models


class Client(object):
    def __init__(self, trans_var: ModelScatterTransVar = ModelScatterTransVar()):
        self._scatter = trans_var.client_model
        self._server_parties = trans_var.server_parties

    def send_model(self, model, suffix=tuple()):
        return self._scatter.remote_parties(obj=model, parties=self._server_parties, suffix=suffix)
