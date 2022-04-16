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

import functools
from federatedml.secureprotol import PaillierEncrypt
from federatedml.util import LOGGER


class EncryptModeCalculator(object):
    """
    Encyprt Mode module, a balance of security level and speed.

    Parameters
    ----------
    encrypter: object, fate-paillier object, object to encrypt numbers

    mode: str, accpet 'strict', 'fast', 'balance'. "confusion_opt", "confusion_opt_balance"
          'strict': means that re-encrypted every function call.

    """

    def __init__(self, encrypter=None, mode="strict", re_encrypted_rate=1):
        self.encrypter = encrypter
        self.mode = mode
        self.re_encrypted_rate = re_encrypted_rate
        self.prev_data = None
        self.prev_encrypted_data = None
        self.enc_zeros = None

        self.align_to_input_data = True

        if self.mode != "strict":
            self.mode = "strict"
            LOGGER.warning("encrypted_mode_calculator will be remove in later version, "
                           "but in current version user can still use it, but it only supports strict mode, "
                           "other mode will be reset to strict for compatibility")

    @staticmethod
    def add_enc_zero(obj, enc_zero):
        pass

    def encrypt_data(self, input_data, enc_func):
        return input_data.mapValues(enc_func)

    def get_enc_func(self, encrypter, raw_enc=False, exponent=0):
        if not raw_enc:
            return encrypter.recursive_encrypt
        else:
            if isinstance(self.encrypter, PaillierEncrypt):
                raw_en_func = functools.partial(self.encrypter.recursive_raw_encrypt, exponent=exponent)
            else:
                raw_en_func = self.encrypter.recursive_raw_encrypt

            return raw_en_func

    def encrypt(self, input_data):
        """
        Encrypt data according to different mode

        Parameters
        ----------
        input_data: Table

        Returns
        -------
        new_data: Table, encrypted result of input_data

        """
        encrypt_func = self.get_enc_func(self.encrypter, raw_enc=False)
        new_data = self.encrypt_data(input_data, encrypt_func)
        return new_data

    def raw_encrypt(self, input_data, exponent=0):
        raw_en_func = self.get_enc_func(self.encrypter, raw_enc=True, exponent=exponent)
        new_data = self.encrypt_data(input_data, raw_en_func)
        return new_data

    def init_enc_zero(self, input_data, raw_en=False, exponent=0):
        pass

    def recursive_encrypt(self, input_data):
        return self.encrypter.recursive_encrypt(input_data)

    def distribute_encrypt(self, input_data):
        return self.encrypt(input_data)

    def distribute_decrypt(self, input_data):
        return self.encrypter.distribute_decrypt(input_data)

    def recursive_decrypt(self, input_data):
        return self.encrypter.recursive_decrypt(input_data)
