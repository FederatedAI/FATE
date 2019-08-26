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
from federatedml.secureprotol.encrypt import PaillierEncrypt
from federatedml.util import consts
from federatedml.util.transfer_variable.base_transfer_variable import Variable

LOGGER = LoggerFactory.get_logger()


class _Host(object):

    def __init__(self,
                 use_encrypt_trv: Variable,
                 paillier_pubkey_trv: Variable,
                 re_encrypt_times_trv: Variable,
                 model_to_re_encrypt_trv: Variable,
                 model_re_encrypted_trv: Variable,
                 suffix):
        self._use_encrypt_trv = use_encrypt_trv
        self._paillier_pubkey_trv = paillier_pubkey_trv
        self._re_encrypt_times_trv = re_encrypt_times_trv
        self._model_to_re_encrypt_trv = model_to_re_encrypt_trv
        self._model_re_encrypted_trv = model_re_encrypted_trv
        self._suffix = suffix

    def maybe_gen_pubkey(self, enable):
        self._use_encrypt_trv.remote(obj=enable, role=consts.ARBITER, idx=0, suffix=self._suffix)
        return self._paillier_pubkey_trv.get(idx=0, suffix=self._suffix) if enable else None

    def set_re_cipher_time(self, re_encrypt_times):
        self._re_encrypt_times_trv.remote(obj=re_encrypt_times, role=consts.ARBITER, idx=0, suffix=self._suffix)
        LOGGER.info("sent re_encrypt_times: {}".format(re_encrypt_times))

    def re_cipher(self, w, *suffix):
        self._model_to_re_encrypt_trv.remote(obj=w, role=consts.ARBITER, suffix=(self._suffix, *suffix))
        _w = self._model_re_encrypted_trv.get(idx=0, suffix=suffix)
        return np.array(_w)


class _Arbiter(object):

    def __init__(self,
                 use_encrypt_trv: Variable,
                 paillier_pubkey_trv: Variable,
                 re_encrypt_times_trv: Variable,
                 model_to_re_encrypt_trv: Variable,
                 model_re_encrypted_trv: Variable,
                 suffix):
        self._use_encrypt_trv = use_encrypt_trv
        self._paillier_pubkey_trv = paillier_pubkey_trv
        self._re_encrypt_times_trv = re_encrypt_times_trv
        self._model_to_re_encrypt_trv = model_to_re_encrypt_trv
        self._model_re_encrypted_trv = model_re_encrypted_trv
        self._suffix = suffix

        self._hosts_use_cipher = None
        self._host_ciphers = dict()
        self._re_encrypt_times = dict()

    def maybe_gen_pubkey(self, key_length):
        self._hosts_use_cipher = self._use_encrypt_trv.get(suffix=self._suffix)

        for idx, use_encryption in enumerate(self._hosts_use_cipher):
            if not use_encryption:
                continue
            else:
                cipher = PaillierEncrypt()
                cipher.generate_key(key_length)
                pub_key = cipher.get_public_key()
                self._paillier_pubkey_trv.remote(obj=pub_key, role=consts.HOST, idx=idx, suffix=self._suffix)
                self._host_ciphers[idx] = cipher
        LOGGER.info(f"hosts that enable paillier cipher: {self._hosts_use_cipher}")
        return self._host_ciphers

    def set_re_cipher_time(self):
        for idx, use_encryption in self._host_ciphers.items():
            self._re_encrypt_times[idx] = self._re_encrypt_times_trv.get(idx=idx, suffix=self._suffix)
        LOGGER.info("re encrypt times for all parties: {}".format(self._re_encrypt_times))

    def re_cipher(self, iter_num, re_encrypt_batches):
        left_re_encrypt_times = self._re_encrypt_times.copy()
        total = sum(left_re_encrypt_times.values())
        batch_iter_num = 0
        while total > 0:
            batch_iter_num += re_encrypt_batches
            idx_remind = [idx for idx, left_times in left_re_encrypt_times.items() if left_times > 0]
            for idx in idx_remind:
                re_encrypt_model = self._model_to_re_encrypt_trv.get(idx=idx,
                                                                     suffix=(self._suffix, iter_num, batch_iter_num))
                cipher = self._host_ciphers[idx]
                decrypt_model = cipher.decrypt_list(re_encrypt_model)
                re_encrypt_model = cipher.encrypt_list(decrypt_model)
                self._model_re_encrypted_trv.remote(obj=re_encrypt_model,
                                                    role=consts.HOST,
                                                    idx=idx,
                                                    suffix=(self._suffix, iter_num, batch_iter_num))
                left_re_encrypt_times[idx] -= 1
                total -= 1


def host(use_encrypt_trv: Variable,
         paillier_pubkey_trv: Variable,
         re_encrypt_times_trv: Variable,
         model_to_re_encrypt_trv: Variable,
         model_re_encrypted_trv: Variable,
         suffix="train"):
    return _Host(use_encrypt_trv,
                 paillier_pubkey_trv,
                 re_encrypt_times_trv,
                 model_to_re_encrypt_trv,
                 model_re_encrypted_trv,
                 suffix)


def arbiter(use_encrypt_trv, paillier_pubkey_trv,
            re_encrypt_times_trv, model_to_re_encrypt_trv, model_re_encrypted_trv,
            suffix="train"):
    return _Arbiter(use_encrypt_trv,
                    paillier_pubkey_trv,
                    re_encrypt_times_trv,
                    model_to_re_encrypt_trv,
                    model_re_encrypted_trv,
                    suffix)
