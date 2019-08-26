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
from federatedml.homo.algo_flow.basic import Arbiter, Guest, Host
from federatedml.util.transfer_variable.base_transfer_variable import Variable
from federatedml.homo.procedure import paillier_cipher


class PaillierArbiter(Arbiter):

    def __init__(self):
        super().__init__()
        self.paillier_cipher = None

    def register_paillier_cipher(self,
                                 use_encrypt_trv: Variable,
                                 paillier_pubkey_trv: Variable,
                                 re_encrypt_times_trv: Variable,
                                 model_to_re_encrypt_trv: Variable,
                                 model_re_encrypted_trv: Variable):
        self.paillier_cipher = paillier_cipher.arbiter(use_encrypt_trv,
                                                       paillier_pubkey_trv,
                                                       re_encrypt_times_trv,
                                                       model_to_re_encrypt_trv,
                                                       model_re_encrypted_trv)

    def initialize(self, key_length=None):
        if self.paillier_cipher:
            self.ciphers = self.paillier_cipher.maybe_gen_pubkey(key_length)
            self.paillier_cipher.set_re_cipher_time()
        super().initialize()


class PaillierHost(Host):
    def __init__(self):
        super().__init__()
        self.paillier_cipher = None

    def register_paillier_cipher(self,
                                 use_encrypt_trv: Variable,
                                 paillier_pubkey_trv: Variable,
                                 re_encrypt_times_trv: Variable,
                                 model_to_re_encrypt_trv: Variable,
                                 model_re_encrypted_trv: Variable):
        self.paillier_cipher = paillier_cipher.host(use_encrypt_trv,
                                                    paillier_pubkey_trv,
                                                    re_encrypt_times_trv,
                                                    model_to_re_encrypt_trv,
                                                    model_re_encrypted_trv)

    def initialize(self, party_weight, enable_paillier_cipher=True, paillier_re_cipher_time=None):
        if self.paillier_cipher:
            self.paillier_cipher.maybe_gen_pubkey(enable_paillier_cipher)
            self.paillier_cipher.set_re_cipher_time(paillier_re_cipher_time)
        super().initialize(party_weight)


class PaillierGuest(Guest):
    pass


def arbiter():
    return PaillierArbiter()


def guest():
    return PaillierHost()


def host():
    return PaillierGuest()
