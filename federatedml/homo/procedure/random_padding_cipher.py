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
from federatedml.homo.sync import dh_keys_exchange
from federatedml.homo.sync import identify_uuid
from federatedml.secureprotol.encrypt import PadsCipher
from federatedml.util.transfer_variable.base_transfer_variable import Variable

LOGGER = LoggerFactory.get_logger()


class _Arbiter(object):
    def __init__(self, uuid_sync: identify_uuid, dh_sync):
        self._uuid_sync = uuid_sync
        self._dh_sync = dh_sync

    def exchange_secret_keys(self):
        LOGGER.info("synchronizing uuid")
        self._uuid_sync.validate_uuid()

        LOGGER.info("Diffie-Hellman keys exchanging")
        self._dh_sync.key_exchange()


class _Client(object):
    def __init__(self, uuid_sync, dh_sync):
        self._uuid_sync = uuid_sync
        self._dh_sync = dh_sync

        self._cipher = None

    def create_cipher(self) -> PadsCipher:
        LOGGER.info("synchronizing uuid")
        uuid = self._uuid_sync.generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self._dh_sync.key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = PadsCipher()
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)
        self._cipher = cipher
        return cipher

    def encrypt(self, transfer_weights):
        transfer_weights.encrypted(self._cipher)


def arbiter(guest_uuid_trv: Variable,
            host_uuid_trv: Variable,
            conflict_flag_trv: Variable,
            dh_pubkey_trv: Variable,
            dh_ciphertext_host_trv: Variable,
            dh_ciphertext_guest_trv: Variable,
            dh_ciphertext_bc_trv: Variable):
    return _Arbiter(uuid_sync=identify_uuid.arbiter(guest_uuid_trv, host_uuid_trv, conflict_flag_trv),
                    dh_sync=dh_keys_exchange.arbiter(dh_pubkey_trv,
                                                     dh_ciphertext_host_trv,
                                                     dh_ciphertext_guest_trv,
                                                     dh_ciphertext_bc_trv))


def guest(guest_uuid_trv: Variable,
          conflict_flag_trv: Variable,
          dh_pubkey_trv: Variable,
          dh_ciphertext_guest_trv: Variable,
          dh_ciphertext_bc_trv: Variable):
    return _Client(uuid_sync=identify_uuid.guest(guest_uuid_trv, conflict_flag_trv),
                   dh_sync=dh_keys_exchange.guest(dh_pubkey_trv,
                                                  dh_ciphertext_guest_trv,
                                                  dh_ciphertext_bc_trv))


def host(host_uuid_trv: Variable,
         conflict_flag_trv: Variable,
         dh_pubkey_trv: Variable,
         dh_ciphertext_host_trv: Variable,
         dh_ciphertext_bc_trv: Variable):
    return _Client(uuid_sync=identify_uuid.guest(host_uuid_trv, conflict_flag_trv),
                   dh_sync=dh_keys_exchange.guest(dh_pubkey_trv,
                                                  dh_ciphertext_host_trv,
                                                  dh_ciphertext_bc_trv))
