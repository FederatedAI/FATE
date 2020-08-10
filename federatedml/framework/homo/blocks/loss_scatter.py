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


class LossScatterTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.loss = self.create_client_to_server_variable(name="loss")


class Server(object):

    def __init__(self, trans_var: LossScatterTransVar = LossScatterTransVar()):
        self._scatter = trans_var.loss
        self._client_parties = trans_var.client_parties

    def get_losses(self, parties=None, suffix=tuple()):
        parties = self._client_parties if parties is None else parties
        return self._scatter.get_parties(parties=parties, suffix=suffix)

    def weighted_loss_mean(self, suffix):
        losses = self.get_losses(suffix=suffix)
        total_loss = 0.0
        total_weight = 0.0
        for loss, weight in losses:
            total_loss += loss * weight
            total_weight += weight
        return total_loss / total_weight


class Client(object):
    def __init__(self, trans_var: LossScatterTransVar = LossScatterTransVar()):
        self._scatter = trans_var.loss
        self._server_parties = trans_var.server_parties

    def send_loss(self, loss, suffix=tuple()):
        return self._scatter.remote_parties(obj=loss, parties=self._server_parties, suffix=suffix)
