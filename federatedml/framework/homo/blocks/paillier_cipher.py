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
from typing import Union

from arch.api import RuntimeInstance
from arch.api.utils import log_utils
from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fate_paillier import PaillierPublicKey
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class PaillierCipherTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.HOST,), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.use_encrypt = self.create_client_to_server_variable(name="use_encrypt")
        self.pailler_pubkey = self.create_server_to_client_variable(name="pailler_pubkey")
        self.re_encrypt_times = self.create_client_to_server_variable(name="re_encrypt_times")
        self.model_to_re_encrypt = self.create_client_to_server_variable(name="model_to_re_encrypt")
        self.model_re_encrypted = self.create_server_to_client_variable(name="model_re_encrypted")


def _get_parties(roles):
    return RuntimeInstance.FEDERATION.roles_to_parties(roles=roles)


class Server(object):

    def __init__(self, trans_var: PaillierCipherTransVar = PaillierCipherTransVar()):
        self._use_encrypt = trans_var.use_encrypt
        self._pailler_pubkey = trans_var.pailler_pubkey
        self._re_encrypt_times = trans_var.re_encrypt_times
        self._model_to_re_encrypt = trans_var.model_to_re_encrypt
        self._model_re_encrypted = trans_var.model_re_encrypted

        self._client_parties = trans_var.client_parties

    def keygen(self, key_length, suffix=tuple()) -> dict:
        use_cipher = self._use_encrypt.get_parties(parties=self._client_parties, suffix=suffix)
        ciphers = dict()
        for party, use_encryption in zip(self._client_parties, use_cipher):
            if not use_encryption:
                ciphers[party] = None
            else:
                cipher = PaillierEncrypt()
                cipher.generate_key(key_length)
                pub_key = cipher.get_public_key()
                self._pailler_pubkey.remote_parties(obj=pub_key, parties=[party], suffix=suffix)
                ciphers[party] = cipher
        return ciphers

    def set_re_cipher_time(self, ciphers, suffix=tuple()):
        re_encrypt_times = dict()
        for party, cipher in ciphers.items():
            if cipher is not None:
                re_encrypt_times[party] = self._re_encrypt_times.get_parties(parties=[party], suffix=suffix)[0]
            else:
                re_encrypt_times[party] = 0
        return re_encrypt_times

    def re_cipher(self, iter_num, re_encrypt_times, ciphers, re_encrypt_batches, suffix=tuple()):
        LOGGER.debug("Get in re_cipher, re_encrypt_times: {}".format(re_encrypt_times))

        left_re_encrypt_times = re_encrypt_times.copy()
        total = sum(left_re_encrypt_times.values())
        batch_iter_num = 0
        while total > 0:
            party_remained = [party for party, left_times in left_re_encrypt_times.items() if left_times > 0]
            LOGGER.debug("Current party_remind: {}, left_re_encrypt_times: {}, total: {}".format(party_remained,
                                                                                                 left_re_encrypt_times,
                                                                                                 total))

            for party in party_remained:
                LOGGER.debug("Before accept re_encrypted_model, batch_iter_num: {}".format(batch_iter_num))
                re_encrypt_model = self._model_to_re_encrypt \
                    .get_parties(parties=[party], suffix=(*suffix, iter_num, batch_iter_num))[0]
                cipher = ciphers[party]
                decrypt_model = cipher.decrypt_list(re_encrypt_model)
                LOGGER.debug("Decrypted host model is : {}".format(decrypt_model))
                re_encrypt_model = cipher.encrypt_list(decrypt_model)
                self._model_re_encrypted.remote_parties(obj=re_encrypt_model,
                                                        parties=[party],
                                                        suffix=(*suffix, iter_num, batch_iter_num))
                left_re_encrypt_times[party] -= 1
                total -= 1
            batch_iter_num += re_encrypt_batches


class Client(object):

    def __init__(self, trans_var: PaillierCipherTransVar = PaillierCipherTransVar()):
        self._use_encrypt = trans_var.use_encrypt
        self._pailler_pubkey = trans_var.pailler_pubkey
        self._re_encrypt_times = trans_var.re_encrypt_times
        self._model_to_re_encrypt = trans_var.model_to_re_encrypt
        self._model_re_encrypted = trans_var.model_re_encrypted

        self._server_parties = trans_var.server_parties

    def gen_paillier_pubkey(self, enable, suffix=tuple()) -> Union[PaillierPublicKey, None]:
        self._use_encrypt.remote_parties(obj=enable, parties=self._server_parties, suffix=suffix)
        if enable:
            ciphers = self._pailler_pubkey.get_parties(parties=self._server_parties, suffix=suffix)
            return ciphers[0]
        return None

    def set_re_cipher_time(self, re_encrypt_times, suffix=tuple()):
        self._re_encrypt_times.remote_parties(obj=re_encrypt_times, parties=self._server_parties, suffix=suffix)
        return re_encrypt_times

    def re_cipher(self, w, iter_num, batch_iter_num, suffix=tuple()):
        self._model_to_re_encrypt.remote_parties(obj=w,
                                                 parties=self._server_parties,
                                                 suffix=(*suffix, iter_num, batch_iter_num))
        ws = self._model_re_encrypted.get_parties(parties=self._server_parties,
                                                  suffix=(*suffix, iter_num, batch_iter_num))
        return ws[0]
