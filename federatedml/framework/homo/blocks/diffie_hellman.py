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
from federatedml.secureprotol.diffie_hellman import DiffieHellman
from federatedml.util import consts


class DHTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.p_power_r = self.create_client_to_server_variable(name="p_power_r")
        self.p_power_r_bc = self.create_server_to_client_variable(name="p_power_r_bc")
        self.pubkey = self.create_server_to_client_variable(name="pubkey")


class Server(object):

    def __init__(self, trans_var: DHTransVar = DHTransVar()):
        self._p_power_r = trans_var.p_power_r
        self._p_power_r_bc = trans_var.p_power_r_bc
        self._pubkey = trans_var.pubkey
        self._client_parties = trans_var.client_parties

    def key_exchange(self):
        p, g = DiffieHellman.key_pair()
        self._pubkey.remote_parties(obj=(int(p), int(g)), parties=self._client_parties)
        pubkey = dict(self._p_power_r.get_parties(parties=self._client_parties))
        self._p_power_r_bc.remote_parties(obj=pubkey, parties=self._client_parties)


class Client(object):

    def __init__(self, trans_var: DHTransVar = DHTransVar()):
        self._p_power_r = trans_var.p_power_r
        self._p_power_r_bc = trans_var.p_power_r_bc
        self._pubkey = trans_var.pubkey
        self._server_parties = trans_var.server_parties

    def key_exchange(self, uuid: str):
        p, g = self._pubkey.get_parties(parties=self._server_parties)[0]
        r = DiffieHellman.generate_secret(p)
        gr = DiffieHellman.encrypt(g, r, p)
        self._p_power_r.remote_parties(obj=(uuid, gr), parties=self._server_parties)
        cipher_texts = self._p_power_r_bc.get_parties(parties=self._server_parties)[0]
        share_secret = {uid: DiffieHellman.decrypt(gr, r, p) for uid, gr in cipher_texts.items() if uid != uuid}
        return share_secret
