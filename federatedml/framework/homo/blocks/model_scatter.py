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

from federatedml.framework.homo.blocks.base import _BlockBase, HomoTransferBase, TransferInfo


class ModelScatterTransferVariable(HomoTransferBase):
    def __init__(self, info: TransferInfo = None):
        super().__init__(info)
        self.client_model = self.create_client_to_server_variable(name="client_model")


class Server(_BlockBase):

    def __init__(self, transfer_variable: ModelScatterTransferVariable = ModelScatterTransferVariable()):
        super().__init__(transfer_variable)
        self._scatter = transfer_variable.client_model

    def get_models(self, suffix=tuple()):
        parties = self._scatter.roles_to_parties(self._scatter.authorized_src_roles)
        models = self._scatter.get_parties(parties, suffix)
        return models


class Client(_BlockBase):
    def __init__(self, transfer_variable: ModelScatterTransferVariable = ModelScatterTransferVariable()):
        super().__init__(transfer_variable)
        self._scatter = transfer_variable.client_model

    def send_model(self, model, suffix=tuple()):
        return self._scatter.remote(obj=model, suffix=suffix)
