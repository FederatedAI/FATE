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

from federatedml.framework.homo.sync import paillier_keygen_sync

from arch.api.utils import log_utils
from federatedml.framework.homo.sync import paillier_re_cipher_sync

LOGGER = log_utils.getLogger()


class Host(paillier_keygen_sync.Host, paillier_re_cipher_sync.Host):

    def register_paillier_cipher(self, transfer_variables):
        self._register_paillier_keygen(use_encrypt_transfer=transfer_variables.use_encrypt,
                                       pubkey_transfer=transfer_variables.paillier_pubkey)
        self._register_paillier_re_cipher(re_encrypt_times_transfer=transfer_variables.re_encrypt_times,
                                          model_to_re_encrypt_transfer=transfer_variables.to_encrypt_model,
                                          model_re_encrypted_transfer=transfer_variables.re_encrypted_model)


class Arbiter(paillier_keygen_sync.Arbiter, paillier_re_cipher_sync.Arbiter):

    def register_paillier_cipher(self, transfer_variables):
        self._register_paillier_keygen(use_encrypt_transfer=transfer_variables.use_encrypt,
                                       pubkey_transfer=transfer_variables.paillier_pubkey)
        self._register_paillier_re_cipher(re_encrypt_times_transfer=transfer_variables.re_encrypt_times,
                                          model_to_re_encrypt_transfer=transfer_variables.to_encrypt_model,
                                          model_re_encrypted_transfer=transfer_variables.re_encrypted_model)
