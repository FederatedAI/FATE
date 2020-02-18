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


class HasConvergedTransferVariable(HomoTransferBase):
    def __init__(self, info: TransferInfo = None):
        super().__init__(info)
        self.has_converged = self.create_server_to_client_variable(name="has_converged")


class Server(_BlockBase):

    def __init__(self, transfer_variable: HasConvergedTransferVariable = HasConvergedTransferVariable()):
        super().__init__(transfer_variable)
        self._broadcaster = transfer_variable.has_converged

    def remote_converge_status(self, is_converge, suffix=tuple()):
        parties = self._broadcaster.roles_to_parties(self._broadcaster.authorized_dst_roles)
        self._broadcaster.remote_parties(obj=is_converge, parties=parties, suffix=suffix)
        return is_converge


class Client(_BlockBase):
    def __init__(self, transfer_variable: HasConvergedTransferVariable = HasConvergedTransferVariable()):
        super().__init__(transfer_variable)
        self._broadcaster = transfer_variable.has_converged

    def get_converge_status(self, suffix=tuple()):
        return self._broadcaster.get(suffix=suffix)[0]
