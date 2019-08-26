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
from federatedml.homo.algo_flow.basic import Arbiter, Host, Guest
from federatedml.util.transfer_variable.base_transfer_variable import Variable
from federatedml.homo.procedure import random_padding_cipher


class RandomPaddingArbiter(Arbiter):
    def __init__(self):
        super().__init__()
        self.random_padding_cipher = None

    def register_random_padding_cipher(self,
                                       guest_uuid_trv: Variable,
                                       host_uuid_trv: Variable,
                                       conflict_flag_trv: Variable,
                                       dh_pubkey_trv: Variable,
                                       dh_ciphertext_host_trv: Variable,
                                       dh_ciphertext_guest_trv: Variable,
                                       dh_ciphertext_bc_trv: Variable):
        self.random_padding_cipher = random_padding_cipher.arbiter(guest_uuid_trv,
                                                                   host_uuid_trv,
                                                                   conflict_flag_trv,
                                                                   dh_pubkey_trv,
                                                                   dh_ciphertext_host_trv,
                                                                   dh_ciphertext_guest_trv,
                                                                   dh_ciphertext_bc_trv)

    def initialize(self):
        if self.random_padding_cipher:
            self.random_padding_cipher.exchange_secret_keys()
        super().initialize()


class RandomPaddingGuest(Guest):
    def __init__(self):
        super().__init__()
        self.random_padding_cipher = None

    def register_random_padding_cipher(self,
                                       guest_uuid_trv: Variable,
                                       conflict_flag_trv: Variable,
                                       dh_pubkey_trv: Variable,
                                       dh_ciphertext_guest_trv: Variable,
                                       dh_ciphertext_bc_trv: Variable):
        self.random_padding_cipher = random_padding_cipher.guest(guest_uuid_trv,
                                                                 conflict_flag_trv,
                                                                 dh_pubkey_trv,
                                                                 dh_ciphertext_guest_trv,
                                                                 dh_ciphertext_bc_trv)

    def initialize(self, party_weight):
        if self.random_padding_cipher:
            self.random_padding_cipher.exchange_secret_keys()
        super().initialize(party_weight)


class RandomPaddingHost(Host):
    def __init__(self):
        super().__init__()
        self.random_padding_cipher = None

    def register_random_padding_cipher(self,
                                       guest_uuid_trv: Variable,
                                       conflict_flag_trv: Variable,
                                       dh_pubkey_trv: Variable,
                                       dh_ciphertext_host_trv: Variable,
                                       dh_ciphertext_bc_trv: Variable):
        self.random_padding_cipher = random_padding_cipher.host(guest_uuid_trv,
                                                                conflict_flag_trv,
                                                                dh_pubkey_trv,
                                                                dh_ciphertext_host_trv,
                                                                dh_ciphertext_bc_trv)

    def initialize(self, party_weight):
        if self.random_padding_cipher:
            self.random_padding_cipher.exchange_secret_keys()
        super().initialize(party_weight)


def arbiter():
    return RandomPaddingArbiter()


def guest():
    return RandomPaddingGuest()


def host():
    return RandomPaddingHost()
