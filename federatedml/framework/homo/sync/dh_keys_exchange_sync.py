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

from federatedml.framework.homo.util.scatter import Scatter
from federatedml.secureprotol.diffie_hellman import DiffieHellman
from federatedml.transfer_variable.transfer_class.base_transfer_variable import Variable
from federatedml.util import consts


class Arbiter(object):

    # noinspection PyAttributeOutsideInit
    def register_dh_key_exchange(self,
                                 dh_pubkey_trv: Variable,
                                 dh_ciphertext_host_trv: Variable,
                                 dh_ciphertext_guest_trv: Variable,
                                 dh_ciphertext_bc_trv: Variable):
        self._dh_pubkey_trv = dh_pubkey_trv
        self._dh_pubkey_scatter = Scatter(dh_ciphertext_host_trv, dh_ciphertext_guest_trv)
        self._dh_ciphertext_bc_trv = dh_ciphertext_bc_trv
        return self

    def key_exchange(self):
        p, g = DiffieHellman.key_pair()
        self._dh_pubkey_trv.remote(obj=(int(p), int(g)), role=None, idx=-1)
        pubkey = dict(self._dh_pubkey_scatter.get())
        self._dh_ciphertext_bc_trv.remote(obj=pubkey, role=None, idx=-1)


class Client(object):

    # noinspection PyAttributeOutsideInit
    def register_dh_key_exchange(self,
                                 dh_pubkey_trv: Variable,
                                 dh_ciphertext_trv: Variable,
                                 dh_ciphertext_bc_trv: Variable):
        self._dh_pubkey_trv = dh_pubkey_trv
        self._dh_ciphertext_trv = dh_ciphertext_trv
        self._dh_ciphertext_bc_trv = dh_ciphertext_bc_trv
        return self

    def key_exchange(self, uuid):
        p, g = self._dh_pubkey_trv.get(idx=0)
        r = DiffieHellman.generate_secret(p)
        gr = DiffieHellman.encrypt(g, r, p)
        self._dh_ciphertext_trv.remote((uuid, gr), role=consts.ARBITER, idx=0)
        cipher_texts = self._dh_ciphertext_bc_trv.get(idx=0)
        share_secret = {uid: DiffieHellman.decrypt(gr, r, p) for uid, gr in cipher_texts.items() if uid != uuid}
        return share_secret


Guest = Client
Host = Client
