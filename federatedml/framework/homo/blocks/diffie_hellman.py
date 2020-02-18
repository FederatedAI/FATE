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

from federatedml.framework.homo.blocks.base import _BlockBase, TransferInfo, HomoTransferBase
from federatedml.secureprotol.diffie_hellman import DiffieHellman


class DHTransferVariable(HomoTransferBase):
    def __init__(self, info: TransferInfo = None):
        super().__init__(info)
        self.p_power_r = self.create_client_to_server_variable(name="p_power_r")
        self.p_power_r_bc = self.create_server_to_client_variable(name="p_power_r_bc")
        self.pubkey = self.create_server_to_client_variable(name="pubkey")


class Server(_BlockBase):

    def __init__(self, transfer_variable: DHTransferVariable = DHTransferVariable()):
        super().__init__(transfer_variable)
        self._p_power_r = transfer_variable.p_power_r
        self._p_power_r_bc = transfer_variable.p_power_r_bc
        self._pubkey = transfer_variable.pubkey

    def key_exchange(self):
        p, g = DiffieHellman.key_pair()
        self._pubkey.remote(obj=(int(p), int(g)))
        pubkey = dict(self._p_power_r.get())
        self._p_power_r_bc.remote(obj=pubkey)


class Client(_BlockBase):

    def __init__(self, transfer_variable: DHTransferVariable = DHTransferVariable()):
        super().__init__(transfer_variable)
        self._p_power_r = transfer_variable.p_power_r
        self._p_power_r_bc = transfer_variable.p_power_r_bc
        self._pubkey = transfer_variable.pubkey

    def key_exchange(self, uuid: str):
        p, g = self._pubkey.get()[0]
        r = DiffieHellman.generate_secret(p)
        gr = DiffieHellman.encrypt(g, r, p)
        self._p_power_r.remote((uuid, gr))
        cipher_texts = self._p_power_r_bc.get()[0]
        share_secret = {uid: DiffieHellman.decrypt(gr, r, p) for uid, gr in cipher_texts.items() if uid != uuid}
        return share_secret
