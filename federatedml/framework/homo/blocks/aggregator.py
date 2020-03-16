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
from arch.api.utils import log_utils
from federatedml.framework.homo.blocks import model_broadcaster, model_scatter
from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.framework.homo.blocks.model_broadcaster import ModelBroadcasterTransVar
from federatedml.framework.homo.blocks.model_scatter import ModelScatterTransVar
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class AggregatorTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.model_scatter = ModelScatterTransVar(server=server, clients=clients, prefix=self.prefix)
        self.model_broadcaster = ModelBroadcasterTransVar(server=server, clients=clients, prefix=self.prefix)


class Server(object):
    def __init__(self, trans_var: AggregatorTransVar = AggregatorTransVar()):
        self._model_broadcaster = model_broadcaster.Server(trans_var=trans_var.model_broadcaster)
        self._model_scatter = model_scatter.Server(trans_var=trans_var.model_scatter)

    def get_models(self, suffix=tuple()):
        return self._model_scatter.get_models(suffix=suffix)

    def send_aggregated_model(self, model, suffix=tuple()):
        self._model_broadcaster.send_model(model=model, suffix=suffix)


class Client(object):
    def __init__(self, trans_var: AggregatorTransVar = AggregatorTransVar()):
        self._model_broadcaster = model_broadcaster.Client(trans_var=trans_var.model_broadcaster)
        self._model_scatter = model_scatter.Client(trans_var=trans_var.model_scatter)

    def send_model(self, model, suffix=tuple()):
        self._model_scatter.send_model(model=model, suffix=suffix)

    def get_aggregated_model(self, suffix=tuple()):
        return self._model_broadcaster.get_model(suffix=suffix)
