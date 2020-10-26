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
from federatedml.framework.homo.blocks import uuid_generator, diffie_hellman
from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.framework.homo.blocks.diffie_hellman import DHTransVar
from federatedml.framework.homo.blocks.uuid_generator import UUIDTransVar
from federatedml.secureprotol.encrypt import TablePadsCipher
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.framework.homo.blocks import random_padding_cipher


class RandomPaddingCipherTransVar(random_padding_cipher.RandomPaddingCipherTransVar):
    pass


class Server(random_padding_cipher.Server):
    pass


class Client(random_padding_cipher.Client):

    def create_cipher(self) -> TablePadsCipher:
        LOGGER.info("synchronizing uuid")
        uuid = self._uuid.generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self._dh.key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = TablePadsCipher()
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)
        self._cipher = cipher
        return cipher
