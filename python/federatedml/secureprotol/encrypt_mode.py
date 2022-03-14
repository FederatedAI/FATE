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

        if self.mode != "strict":
            self.mode = "strict"
            LOGGER.warning("encrypted_mode_calculator will be remove in later version, "
                           "but in current version user can still use it, but it only supports strict mode, "
                           "other mode will be reset to strict for compatibility")

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
        new_data = input_data.mapValues(self.encrypter.recursive_encrypt)
        return new_data
