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


from federatedml.secureprotol.encrypt import PaillierEncrypt
from federatedml.secureprotol.fate_paillier import PaillierPublicKey
from federatedml.util import consts


class Arbiter(object):

    # noinspection PyAttributeOutsideInit
    def _register_paillier_keygen(self, use_encrypt_transfer, pubkey_transfer):
        self._use_encrypt_transfer = use_encrypt_transfer
        self._pubkey_transfer = pubkey_transfer
        return self

    def paillier_keygen(self, key_length, suffix=tuple()) -> dict:
        hosts_use_cipher = self._use_encrypt_transfer.get(suffix=suffix)
        host_ciphers = dict()
        for idx, use_encryption in enumerate(hosts_use_cipher):
            if not use_encryption:
                host_ciphers[idx] = None
            else:
                cipher = PaillierEncrypt()
                cipher.generate_key(key_length)
                pub_key = cipher.get_public_key()
                self._pubkey_transfer.remote(obj=pub_key, role=consts.HOST, idx=idx, suffix=suffix)
                host_ciphers[idx] = cipher
        return host_ciphers


class Host(object):

    # noinspection PyAttributeOutsideInit
    def _register_paillier_keygen(self, use_encrypt_transfer, pubkey_transfer):
        self._use_encrypt_transfer = use_encrypt_transfer
        self._pubkey_transfer = pubkey_transfer
        return self

    def gen_paillier_pubkey(self, enable, suffix=tuple()) -> PaillierPublicKey:
        self._use_encrypt_transfer.remote(obj=enable, role=consts.ARBITER, idx=0, suffix=suffix)
        if enable:
            return self._pubkey_transfer.get(idx=0, suffix=suffix)
