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
import numpy as np

from arch.api.utils.log_utils import LoggerFactory
from federatedml.homo.transfer import arbiter2host, host2arbiter
from federatedml.secureprotol.encrypt import PaillierEncrypt

LOGGER = LoggerFactory.get_logger()


def _tag_suffix(epoch_iter, batch_iter):
    return f"{epoch_iter}.{batch_iter}"


def _get_tag(transfer_variable, name, suffix=None):
    if None:
        return transfer_variable.generate_transferid(getattr(transfer_variable, name))
    else:
        return transfer_variable.generate_transferid(getattr(transfer_variable, name), suffix)


class _Host(object):

    def __init__(self, h2a_use_encrypt,
                 paillier_pubkey_broadcast,
                 h2a_re_encrypt_times,
                 h2a_model_to_re_encrypt,
                 a2h_model_re_encrypted):
        self._h2a_use_encrypt = h2a_use_encrypt
        self._paillier_pubkey_broadcast = paillier_pubkey_broadcast
        self._h2a_re_encrypt_times = h2a_re_encrypt_times
        self._h2a_model_to_re_encrypt = h2a_model_to_re_encrypt
        self._a2h_model_re_encrypted = a2h_model_re_encrypted

    def maybe_gen_pubkey(self, enable):
        self._h2a_use_encrypt.remote(enable)
        if enable:
            return self._paillier_pubkey_broadcast.get()

    def set_re_cipher_time(self, re_encrypt_times):
        self._h2a_re_encrypt_times.remote(re_encrypt_times, idx=0)
        LOGGER.info("sent re_encrypt_times: {}".format(re_encrypt_times))

    def re_cipher(self, w, iter_num, batch_num):
        self._h2a_model_to_re_encrypt.remote(value=w, suffix=_tag_suffix(iter_num, batch_num))
        _w = self._a2h_model_re_encrypted.get(idx=0, suffix=_tag_suffix(iter_num, batch_num))
        return np.array(_w)


class _Arbiter(object):

    def __init__(self, h2a_use_encrypt,
                 paillier_pubkey_broadcast,
                 h2a_re_encrypt_times,
                 h2a_model_to_re_encrypt,
                 a2h_model_re_encrypted):
        self._h2a_use_encrypt = h2a_use_encrypt
        self._paillier_pubkey_broadcast = paillier_pubkey_broadcast
        self._h2a_re_encrypt_times = h2a_re_encrypt_times
        self._h2a_model_to_re_encrypt = h2a_model_to_re_encrypt
        self._a2h_model_re_encrypted = a2h_model_re_encrypted

        self._hosts_use_cipher = None
        self._host_ciphers = dict()
        self._re_encrypt_times = dict()

    def maybe_gen_pubkey(self, key_length):
        self._hosts_use_cipher = self._h2a_use_encrypt.get()

        for idx, use_encryption in enumerate(self._hosts_use_cipher):
            if not use_encryption:
                continue
            else:
                cipher = PaillierEncrypt()
                cipher.generate_key(key_length)
                pub_key = cipher.get_public_key()
                self._paillier_pubkey_broadcast.remote(pub_key)
                self._host_ciphers[idx] = cipher
        LOGGER.info(f"hosts that enable paillier cipher: {self._hosts_use_cipher}")
        return self._host_ciphers

    def set_re_cipher_time(self):
        for idx, use_encryption in self._host_ciphers.items():
            self._re_encrypt_times[idx] = self._h2a_re_encrypt_times.get(idx=idx)
        LOGGER.info("re encrypt times for all parties: {}".format(self._re_encrypt_times))

    def re_cipher(self, iter_num, re_encrypt_batches):
        left_re_encrypt_times = self._re_encrypt_times.copy()
        total = sum(left_re_encrypt_times.values())
        batch_iter_num = 0
        while total > 0:
            batch_iter_num += re_encrypt_batches
            idx_remind = [idx for idx, left_times in left_re_encrypt_times.items() if left_times > 0]
            for idx in idx_remind:
                re_encrypt_model = self._h2a_model_to_re_encrypt.get(idx=idx,
                                                                     suffix=_tag_suffix(iter_num, batch_iter_num))
                cipher = self._host_ciphers[idx]
                decrypt_model = cipher.decrypt_list(re_encrypt_model)
                re_encrypt_model = cipher.encrypt_list(decrypt_model)
                self._a2h_model_re_encrypted.remote(value=re_encrypt_model,
                                                    idx=idx,
                                                    suffix=_tag_suffix(iter_num, batch_iter_num))
                left_re_encrypt_times[idx] -= 1
                total -= 1


def _parse_transfer_variable(transfer_variable, suffix="train"):
    h2a_use_encrypt = host2arbiter(name=transfer_variable.use_encrypt.name,
                                   tag=_get_tag(transfer_variable, "use_encrypt"))
    paillier_pubkey_broadcast = arbiter2host(name=transfer_variable.paillier_pubkey.name,
                                             tag=_get_tag(transfer_variable, "paillier_pubkey", suffix))
    h2a_model_to_re_encrypt = host2arbiter(name=transfer_variable.to_encrypt_model.name,
                                           tag=_get_tag(transfer_variable, "to_encrypt_model", suffix))
    a2h_model_re_encrypted = arbiter2host(name=transfer_variable.re_encrypt_model.name,
                                          tag=_get_tag(transfer_variable, "re_encrypt_model", suffix))
    h2a_re_encrypt_times = host2arbiter(name=transfer_variable.re_encrypt_times.name,
                                        tag=_get_tag(transfer_variable, "re_encrypt_times", suffix))
    return (h2a_use_encrypt,
            paillier_pubkey_broadcast,
            h2a_re_encrypt_times,
            h2a_model_to_re_encrypt,
            a2h_model_re_encrypted)


class PaillierCipherProcedure(object):

    @staticmethod
    def host(transfer_variable, suffix="train"):
        return _Host(*_parse_transfer_variable(transfer_variable, suffix))

    @staticmethod
    def arbiter(transfer_variable, suffix="train"):
        return _Arbiter(*_parse_transfer_variable(transfer_variable, suffix))
