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
import typing

from federatedml.transfer_variable.base_transfer_variable import FlowID, Variable, BaseTransferVariables
from federatedml.util import consts


class _BlockBase(object):
    def __init__(self, transfer_variable):
        self._transfer_variable = transfer_variable

    def set_flowid(self, flowid):
        self._transfer_variable.set_flowid(flowid)
        return self


class TransferInfo(object):
    def __init__(self, server: typing.Tuple, clients: typing.Tuple, flowid: FlowID, name_prefix=None):
        self.server = server
        self.clients = clients
        self.flowid = flowid
        self.name_prefix = name_prefix


class HomoTransferBase(BaseTransferVariables):
    def __init__(self, patten=None, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST)):
        super().__init__()
        class_name = self.__class__.__name__
        if patten is None:
            self.info = TransferInfo(server=server,
                                     clients=clients,
                                     flowid=self._flowid,
                                     name_prefix=class_name)
        else:
            self.info = TransferInfo(server=patten.server,
                                     clients=patten.clients,
                                     flowid=patten.flowid,
                                     name_prefix=f"{patten.name_prefix}.{class_name}")

    def create_client_to_server_variable(self, name):
        return Variable(name=f"{self.info.name_prefix}.{name}",
                        src=self.info.clients,
                        dst=self.info.server,
                        flowid=self.info.flowid)

    def create_server_to_client_variable(self, name):
        return Variable(name=f"{self.info.name_prefix}.{name}",
                        src=self.info.server,
                        dst=self.info.clients,
                        flowid=self.info.flowid)
