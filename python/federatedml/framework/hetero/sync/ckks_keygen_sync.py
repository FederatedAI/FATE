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


from federatedml.secureprotol.encrypt import CKKSEncrypt
from federatedml.util import consts


class Arbiter(object):
    # noinspection PyAttributeOutsideInit
    def _register_ckks_keygen(self, pubkey_transfer):
        self._pubkey_transfer = pubkey_transfer

    def ckks_keygen(self, poly_modulus_degree, coeff_mod_bit_sizes, global_scale, suffix=tuple()):
        # Initialize CKKSEncrypt class
        cipher = CKKSEncrypt()
        cipher.generate_key(poly_modulus_degree, coeff_mod_bit_sizes, global_scale)

        # Remote public key to transfer variables
        pub_key = cipher.get_public_key()
        self._pubkey_transfer.remote(obj=pub_key, role=consts.HOST, idx=-1, suffix=suffix)
        self._pubkey_transfer.remote(obj=pub_key, role=consts.GUEST, idx=-1, suffix=suffix)

        # Return a CKKSEncrypt object
        return cipher


class _Client(object):
    # noinspection PyAttributeOutsideInit
    def _register_ckks_keygen(self, pubkey_transfer):
        self._pubkey_transfer = pubkey_transfer

    def gen_ckks_cipher_operator(self, suffix=tuple()):
        # Get public key from transfer variables
        pubkey = self._pubkey_transfer.get(idx=0, suffix=suffix)

        # Create a CKKSEncrypt object with public key loaded
        cipher = CKKSEncrypt()
        cipher.set_public_key(pubkey)

        # Return CKKSEncrypt object with public key loaded
        return cipher


Host = _Client
Guest = _Client
