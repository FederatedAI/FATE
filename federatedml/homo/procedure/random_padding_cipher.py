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

from arch.api.utils.log_utils import LoggerFactory
from federatedml.homo.sync.synchronized_uuid import SynchronizedUUIDProcedure
from federatedml.homo.sync.diffie_hellman_keys_exchange import DHKeysExchange
from federatedml.secureprotol.encrypt import PadsCipher

LOGGER = LoggerFactory.get_logger()


class _Arbiter(object):
    def __init__(self, transfer_variable):
        self._transfer_variable = transfer_variable

    def create_cipher(self):
        LOGGER.info("synchronizing uuid")
        SynchronizedUUIDProcedure.arbiter(self._transfer_variable).validate_uuid()

        LOGGER.info("Diffie-Hellman keys exchanging")
        DHKeysExchange.arbiter(self._transfer_variable).key_exchange()


class _Guest(object):
    def __init__(self, transfer_variable):
        self._transfer_variable = transfer_variable
        self._cipher = None

    def create_cipher(self) -> PadsCipher:
        LOGGER.info("synchronizing uuid")
        uuid = SynchronizedUUIDProcedure.guest(self._transfer_variable).generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = DHKeysExchange.guest(self._transfer_variable).key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = PadsCipher()
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)
        self._cipher = cipher
        return cipher

    def encrypt(self, transfer_weights):
        transfer_weights.encrypted(self._cipher)


class _Host(object):
    def __init__(self, transfer_variable):
        self._transfer_variable = transfer_variable
        self._cipher = None

    def create_cipher(self) -> PadsCipher:
        LOGGER.info("synchronizing uuid")
        uuid = SynchronizedUUIDProcedure.host(self._transfer_variable).generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = DHKeysExchange.host(self._transfer_variable).key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = PadsCipher()
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)
        self._cipher = cipher
        return cipher

    def encrypt(self, transfer_weights):
        transfer_weights.encrypted(self._cipher)


class RandomPadding(object):
    @staticmethod
    def arbiter(transfer_variable):
        return _Arbiter(transfer_variable)\


    @staticmethod
    def guest(transfer_variable):
        return _Guest(transfer_variable)\


    @staticmethod
    def host(transfer_variable):
        return _Host(transfer_variable)
