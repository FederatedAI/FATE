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
from federatedml.framework.homo.blocks.base import _BlockBase, HomoTransferBase, TransferInfo
from federatedml.framework.homo.blocks.model_broadcaster import ModelBroadcasterTransferVariable
from federatedml.framework.homo.blocks.model_scatter import ModelScatterTransferVariable

LOGGER = log_utils.getLogger()


class AggregatorTransferVariable(HomoTransferBase):
    def __init__(self, info: TransferInfo = None):
        super().__init__(info)
        self.model_scatter = ModelScatterTransferVariable(self.info)
        self.model_broadcaster = ModelBroadcasterTransferVariable(self.info)


class Server(_BlockBase):
    def __init__(self, transfer_variable: AggregatorTransferVariable = AggregatorTransferVariable()):
        super().__init__(transfer_variable)
        self._model_broadcaster = model_broadcaster.Server(transfer_variable.model_broadcaster)
        self._model_scatter = model_scatter.Server(transfer_variable.model_scatter)

    def get_models(self, suffix=tuple()):
        return self._model_scatter.get_models(suffix=suffix)

    def send_aggregated_model(self, model, suffix=tuple()):
        self._model_broadcaster.send_model(model=model, suffix=suffix)


class Client(_BlockBase):
    def __init__(self, transfer_variable: AggregatorTransferVariable = AggregatorTransferVariable()):
        super().__init__(transfer_variable)
        self._model_broadcaster = model_broadcaster.Client(transfer_variable.model_broadcaster)
        self._model_scatter = model_scatter.Client(transfer_variable.model_scatter)

    def send_model(self, model, suffix=tuple()):
        self._model_scatter.send_model(model=model, suffix=suffix)

    def get_aggregated_model(self, suffix=tuple()):
        return self._model_broadcaster.get_model(suffix=suffix)
