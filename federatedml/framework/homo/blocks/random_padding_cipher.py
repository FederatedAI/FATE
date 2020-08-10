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
from federatedml.framework.homo.blocks import uuid_generator, diffie_hellman
from federatedml.framework.homo.blocks.base import HomoTransferBase
from federatedml.framework.homo.blocks.diffie_hellman import DHTransVar
from federatedml.framework.homo.blocks.uuid_generator import UUIDTransVar
from federatedml.secureprotol.encrypt import PadsCipher
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class RandomPaddingCipherTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.uuid_transfer_variable = UUIDTransVar(server=server, clients=clients, prefix=self.prefix)
        self.dh_transfer_variable = DHTransVar(server=server, clients=clients, prefix=self.prefix)


class Server(object):

    def __init__(self, trans_var: RandomPaddingCipherTransVar = RandomPaddingCipherTransVar()):
        self._uuid = uuid_generator.Server(trans_var=trans_var.uuid_transfer_variable)
        self._dh = diffie_hellman.Server(trans_var=trans_var.dh_transfer_variable)

    def exchange_secret_keys(self):
        LOGGER.info("synchronizing uuid")
        self._uuid.validate_uuid()

        LOGGER.info("Diffie-Hellman keys exchanging")
        self._dh.key_exchange()


class Client(object):

    def __init__(self, trans_var: RandomPaddingCipherTransVar = RandomPaddingCipherTransVar()):
        self._uuid = uuid_generator.Client(trans_var=trans_var.uuid_transfer_variable)
        self._dh = diffie_hellman.Client(trans_var=trans_var.dh_transfer_variable)
        self._cipher = None

    def create_cipher(self) -> PadsCipher:
        LOGGER.info("synchronizing uuid")
        uuid = self._uuid.generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self._dh.key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = PadsCipher()
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)
        self._cipher = cipher
        return cipher

    def encrypt(self, transfer_weights):
        return self._cipher.encrypt(transfer_weights)
