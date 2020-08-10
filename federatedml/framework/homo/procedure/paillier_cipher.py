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

from arch.api.utils import log_utils
from federatedml.framework.homo.blocks import paillier_cipher
from federatedml.framework.homo.blocks.paillier_cipher import PaillierCipherTransVar
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fate_paillier import PaillierPublicKey

LOGGER = log_utils.getLogger()


class Host(object):

    def __init__(self, trans_var=PaillierCipherTransVar()):
        self._paillier = paillier_cipher.Client(trans_var=trans_var)

    def register_paillier_cipher(self, transfer_variables):
        pass

    def gen_paillier_pubkey(self, enable, suffix=tuple()) -> typing.Union[PaillierPublicKey, None]:
        return self._paillier.gen_paillier_pubkey(enable=enable, suffix=suffix)

    def set_re_cipher_time(self, re_encrypt_times, suffix=tuple()):
        return self._paillier.set_re_cipher_time(re_encrypt_times=re_encrypt_times, suffix=suffix)

    def re_cipher(self, w, iter_num, batch_iter_num, suffix=tuple()):
        return self._paillier.re_cipher(w=w, iter_num=iter_num, batch_iter_num=batch_iter_num, suffix=suffix)


class Arbiter(object):

    def register_paillier_cipher(self, transfer_variables):
        pass

    def __init__(self, trans_var=PaillierCipherTransVar()):
        self._paillier = paillier_cipher.Server(trans_var=trans_var)
        self._client_parties = trans_var.client_parties
        self._party_idx_map = {party: idx for idx, party in enumerate(self._client_parties)}

    def paillier_keygen(self, key_length, suffix=tuple()) -> typing.Mapping[int, typing.Union[PaillierEncrypt, None]]:
        ciphers = self._paillier.keygen(key_length, suffix)
        return {self._party_idx_map[party]: cipher for party, cipher in ciphers.items()}

    def set_re_cipher_time(self, ciphers: typing.Mapping[int, typing.Union[PaillierEncrypt, None]],
                           suffix=tuple()):
        _ciphers = {self._client_parties[idx]: cipher for idx, cipher in ciphers.items()}
        recipher_times = self._paillier.set_re_cipher_time(_ciphers, suffix)
        return {self._party_idx_map[party]: time for party, time in recipher_times.items()}

    def re_cipher(self, iter_num, re_encrypt_times, host_ciphers_dict, re_encrypt_batches, suffix=tuple()):
        _ciphers = {self._client_parties[idx]: cipher for idx, cipher in host_ciphers_dict.items()}
        _re_encrypt_times = {self._client_parties[idx]: time for idx, time in re_encrypt_times.items()}
        return self._paillier.re_cipher(iter_num, _re_encrypt_times, _ciphers, re_encrypt_batches, suffix)
