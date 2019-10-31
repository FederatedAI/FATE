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
from federatedml.framework.homo.sync import identify_uuid_sync, dh_keys_exchange_sync
from federatedml.secureprotol.encrypt import PadsCipher
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class Arbiter(identify_uuid_sync.Arbiter,
              dh_keys_exchange_sync.Arbiter):

    def register_random_padding_cipher(self, transfer_variables):
        """
        register transfer of uuid and dh-key-exchange.
        Args:
            transfer_variables: assuming transfer_variable has variables:
                1. guest_uuid,  host_uuid and uuid_conflict_flag for uuid generate transfer
                2. dh_pubkey, dh_guest_ciphertext,  dh_host_ciphertext, dh_bc_ciphertext for dh key exchange
        """
        self.register_identify_uuid(guest_uuid_trv=transfer_variables.guest_uuid,
                                    host_uuid_trv=transfer_variables.host_uuid,
                                    conflict_flag_trv=transfer_variables.uuid_conflict_flag)

        self.register_dh_key_exchange(dh_pubkey_trv=transfer_variables.dh_pubkey,
                                      dh_ciphertext_guest_trv=transfer_variables.dh_ciphertext_guest,
                                      dh_ciphertext_host_trv=transfer_variables.dh_ciphertext_host,
                                      dh_ciphertext_bc_trv=transfer_variables.dh_ciphertext_bc)
        return self

    def exchange_secret_keys(self):
        LOGGER.info("synchronizing uuid")
        self.validate_uuid()

        LOGGER.info("Diffie-Hellman keys exchanging")
        self.key_exchange()


class _Client(identify_uuid_sync.Client,
              dh_keys_exchange_sync.Client):

    # noinspection PyAttributeOutsideInit
    def create_cipher(self) -> PadsCipher:
        LOGGER.info("synchronizing uuid")
        uuid = self.generate_uuid()
        LOGGER.info(f"local uuid={uuid}")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self.key_exchange(uuid)
        LOGGER.info(f"Diffie-Hellman exchanged keys {exchanged_keys}")

        cipher = PadsCipher()
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)
        self._cipher = cipher
        return cipher

    def encrypt(self, transfer_weights):
        return self._cipher.encrypt(transfer_weights)


class Guest(_Client):

    def register_random_padding_cipher(self, transfer_variables):
        self.register_identify_uuid(uuid_transfer_variable=transfer_variables.guest_uuid,
                                    conflict_flag_transfer_variable=transfer_variables.uuid_conflict_flag)
        self.register_dh_key_exchange(dh_pubkey_trv=transfer_variables.dh_pubkey,
                                      dh_ciphertext_trv=transfer_variables.dh_ciphertext_guest,
                                      dh_ciphertext_bc_trv=transfer_variables.dh_ciphertext_bc)
        return self


class Host(_Client):

    def register_random_padding_cipher(self, transfer_variables):
        self.register_identify_uuid(uuid_transfer_variable=transfer_variables.host_uuid,
                                    conflict_flag_transfer_variable=transfer_variables.uuid_conflict_flag)
        self.register_dh_key_exchange(dh_pubkey_trv=transfer_variables.dh_pubkey,
                                      dh_ciphertext_trv=transfer_variables.dh_ciphertext_host,
                                      dh_ciphertext_bc_trv=transfer_variables.dh_ciphertext_bc)
        return self


def with_role(role, transfer_variable):
    if role == consts.GUEST:
        return Guest().register_random_padding_cipher(transfer_variable)
    elif role == consts.HOST:
        return Host().register_random_padding_cipher(transfer_variable)
    elif role == consts.ARBITER:
        return Arbiter().register_random_padding_cipher(transfer_variable)
    else:
        raise ValueError(f"role {role} not found")
