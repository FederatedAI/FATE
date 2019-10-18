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

from arch.api.utils import log_utils
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class Arbiter(object):
    # noinspection PyAttributeOutsideInit
    def _register_paillier_re_cipher(self, re_encrypt_times_transfer,
                                     model_to_re_encrypt_transfer,
                                     model_re_encrypted_transfer):
        self._re_encrypt_times_transfer = re_encrypt_times_transfer
        self._model_to_re_encrypt_transfer = model_to_re_encrypt_transfer
        self._model_re_encrypted_transfer = model_re_encrypted_transfer
        return self

    def set_re_cipher_time(self, host_ciphers_dict, suffix=tuple()):
        re_encrypt_times = dict()
        for idx, cipher in host_ciphers_dict.items():
            if cipher:
                re_encrypt_times[idx] = self._re_encrypt_times_transfer.get(idx=idx, suffix=suffix)
            else:
                re_encrypt_times[idx] = 0
        return re_encrypt_times

    def re_cipher(self, iter_num, re_encrypt_times, host_ciphers_dict, re_encrypt_batches, suffix=tuple()):

        LOGGER.debug("Get in re_cipher, re_encrypt_times: {}".format(re_encrypt_times))

        left_re_encrypt_times = re_encrypt_times.copy()
        total = sum(left_re_encrypt_times.values())
        batch_iter_num = 0
        while total > 0:
            idx_remind = [idx for idx, left_times in left_re_encrypt_times.items() if left_times > 0]
            LOGGER.debug("Current idx_remind: {}, left_re_encrypt_times: {}, total: {}".format(idx_remind,
                                                                                               left_re_encrypt_times,
                                                                                               total))

            for idx in idx_remind:
                re_encrypt_model = self._model_to_re_encrypt_transfer.get(idx=idx,
                                                                          suffix=(*suffix, iter_num, batch_iter_num))
                cipher = host_ciphers_dict[idx]
                decrypt_model = cipher.decrypt_list(re_encrypt_model)
                LOGGER.debug("Decrypted host model is : {}".format(decrypt_model))
                re_encrypt_model = cipher.encrypt_list(decrypt_model)
                self._model_re_encrypted_transfer.remote(obj=re_encrypt_model,
                                                         role=consts.HOST,
                                                         idx=idx,
                                                         suffix=(*suffix, iter_num, batch_iter_num))
                left_re_encrypt_times[idx] -= 1
                total -= 1
            batch_iter_num += re_encrypt_batches



class Host(object):
    # noinspection PyAttributeOutsideInit
    def _register_paillier_re_cipher(self, re_encrypt_times_transfer,
                                     model_to_re_encrypt_transfer,
                                     model_re_encrypted_transfer):
        self._re_encrypt_times_transfer = re_encrypt_times_transfer
        self._model_to_re_encrypt_transfer = model_to_re_encrypt_transfer
        self._model_re_encrypted_transfer = model_re_encrypted_transfer
        return self

    def set_re_cipher_time(self, re_encrypt_times, suffix=tuple()):
        self._re_encrypt_times_transfer.remote(obj=re_encrypt_times, role=consts.ARBITER, idx=0, suffix=suffix)
        return re_encrypt_times

    def re_cipher(self, w, iter_num, batch_iter_num, suffix=tuple()):
        self._model_to_re_encrypt_transfer.remote(obj=w,
                                                  role=consts.ARBITER,
                                                  suffix=(*suffix, iter_num, batch_iter_num))
        _w = self._model_re_encrypted_transfer.get(idx=0,
                                                   suffix=(*suffix, iter_num, batch_iter_num))
        return _w
