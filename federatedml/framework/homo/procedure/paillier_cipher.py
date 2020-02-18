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

from deprecated import deprecated

from arch.api.utils import log_utils
from federatedml.framework.homo.blocks import paillier_cipher
from federatedml.secureprotol import PaillierEncrypt
from federatedml.transfer_variable.base_transfer_variable import Variable

LOGGER = log_utils.getLogger()


class Host(paillier_cipher.Client):

    @deprecated(reason="could be remove")
    def register_paillier_cipher(self, transfer_variables):
        return self


class Arbiter(paillier_cipher.Server):

    @deprecated(reason="could be remove")
    def register_paillier_cipher(self, transfer_variables):
        return self

    def __init__(self):
        super().__init__()
        re_encrypt_times_variable: Variable = self._transfer_variable.re_encrypt_times
        self._parties = re_encrypt_times_variable.roles_to_parties(re_encrypt_times_variable.authorized_src_roles)
        self._party_idx_map = {party: idx for idx, party in enumerate(self._parties)}

    def paillier_keygen(self, key_length, suffix=tuple()) -> typing.Mapping[int, typing.Union[PaillierEncrypt, None]]:
        ciphers = super().keygen(key_length, suffix)
        return {self._party_idx_map[party]: cipher for party, cipher in ciphers.items()}

    def set_re_cipher_time(self, ciphers: typing.Mapping[int, typing.Union[PaillierEncrypt, None]], suffix=tuple()):
        _ciphers = {self._parties[idx]: cipher for idx, cipher in ciphers.items()}
        recipher_times = super().set_re_cipher_time(_ciphers, suffix)
        return {self._party_idx_map[party]: time for party, time in recipher_times.items()}

    def re_cipher(self, iter_num, re_encrypt_times, ciphers, re_encrypt_batches, suffix=tuple()):
        _ciphers = {self._parties[idx]: cipher for idx, cipher in ciphers.items()}
        _re_encrypt_times = {self._parties[idx]: time for idx, time in re_encrypt_times.items()}
        return super().re_cipher(iter_num, _re_encrypt_times, _ciphers, re_encrypt_batches, suffix)
