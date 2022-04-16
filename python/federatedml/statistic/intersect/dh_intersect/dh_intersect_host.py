#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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

import uuid

from federatedml.secureprotol.symmetric_encryption.cryptor_executor import CryptoExecutor
from federatedml.secureprotol.symmetric_encryption.pohlig_hellman_encryption import PohligHellmanCipherKey
from federatedml.statistic.intersect.dh_intersect.dh_intersect_base import DhIntersect
from federatedml.util import consts, LOGGER


class DhIntersectionHost(DhIntersect):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST
        self.id_list_local_first = None

    def _sync_commutative_cipher_public_knowledge(self):
        self.commutative_cipher = self.transfer_variable.commutative_cipher_public_knowledge.get(idx=0)

        LOGGER.info(f"got commutative cipher public knowledge from guest")

    def _exchange_id_list(self, id_list):
        id_only = id_list.mapValues(lambda v: None)
        self.transfer_variable.id_ciphertext_list_exchange_h2g.remote(id_only,
                                                                      role=consts.GUEST,
                                                                      idx=0)
        LOGGER.info("sent id 1st ciphertext list to guest")

        id_list_guest = self.transfer_variable.id_ciphertext_list_exchange_g2h.get(idx=0)
        LOGGER.info("got id 1st ciphertext list from guest")

        return id_list_guest

    def _sync_doubly_encrypted_id_list(self, id_list):
        self.transfer_variable.doubly_encrypted_id_list.remote(id_list,
                                                               role=consts.GUEST,
                                                               idx=0)
        LOGGER.info("sent doubly encrypted id list to guest")

    def get_intersect_ids(self):
        first_cipher_intersect_ids = self.transfer_variable.intersect_ids.get(idx=0)
        LOGGER.info(f"obtained cipher intersect ids from guest")
        intersect_ids = self.map_encrypt_id_to_raw_id(first_cipher_intersect_ids,
                                                      self.id_list_local_first,
                                                      keep_encrypt_id=False)
        return intersect_ids

    def get_intersect_doubly_encrypted_id(self, data_instances):
        self._sync_commutative_cipher_public_knowledge()
        self.commutative_cipher.init()

        # 1st ID encrypt: (Eh, (h, Instance))
        self.id_list_local_first = self._encrypt_id(data_instances,
                                                    self.commutative_cipher,
                                                    reserve_original_key=True,
                                                    hash_operator=self.hash_operator,
                                                    salt=self.salt,
                                                    reserve_original_value=True)
        LOGGER.info("encrypted local id for the 1st time")
        # send (Eh, -1), get (Eg, -1)
        id_list_remote_first = self._exchange_id_list(self.id_list_local_first)

        # 2nd ID encrypt & send doubly encrypted guest ID list to guest
        id_list_remote_second = self._encrypt_id(id_list_remote_first,
                                                 self.commutative_cipher,
                                                 reserve_original_key=True)  # (EEg, Eg)
        LOGGER.info("encrypted guest id for the 2nd time")
        self._sync_doubly_encrypted_id_list(id_list_remote_second)

    def decrypt_intersect_doubly_encrypted_id(self, id_list_intersect_cipher_cipher=None):
        """
        if self.cardinality_only:
            cardinality = None
            if self.sync_cardinality:
                cardinality = self.transfer_variable.cardinality.get(cardinality, role=consts.GUEST, idx=0)
                LOGGER.info(f"Got intersect cardinality from guest.")
            return cardinality
        """

        intersect_ids = None
        if self.sync_intersect_ids:
            intersect_ids = self.get_intersect_ids()

        return intersect_ids

    def get_intersect_key(self, party_id=None):
        cipher_core = self.commutative_cipher.cipher_core

        intersect_key = {"mod_base": str(cipher_core.mod_base),
                         "exponent": str(cipher_core.exponent)}

        return intersect_key

    def load_intersect_key(self, cache_meta):
        intersect_key = cache_meta[str(self.guest_party_id)]["intersect_key"]

        mod_base = int(intersect_key["mod_base"])
        exponent = int(intersect_key["exponent"])

        ph_key = PohligHellmanCipherKey(mod_base, exponent)
        self.commutative_cipher = CryptoExecutor(ph_key)

    def generate_cache(self, data_instances):
        self._sync_commutative_cipher_public_knowledge()
        self.commutative_cipher.init()

        cache_id = str(uuid.uuid4())
        self.cache_id = {self.guest_party_id: cache_id}
        # id_only.schema = cache_schema
        self.cache_transfer_variable.remote(cache_id, role=consts.GUEST, idx=0)
        LOGGER.info(f"remote cache_id to guest")

        # 1st ID encrypt: (Eh, (h, Instance))
        id_list_local_first = self._encrypt_id(data_instances,
                                               self.commutative_cipher,
                                               reserve_original_key=True,
                                               hash_operator=self.hash_operator,
                                               salt=self.salt,
                                               reserve_original_value=True)
        LOGGER.info("encrypted local id for the 1st time")

        # cache_schema = {"cache_id": cache_id}
        # id_list_local_first.schema = cache_schema

        id_only = id_list_local_first.mapValues(lambda v: None)
        self.transfer_variable.id_ciphertext_list_exchange_h2g.remote(id_only,
                                                                      role=consts.GUEST,
                                                                      idx=0)
        LOGGER.info("sent id 1st ciphertext list to guest")
        """
        cache_set = {
            self.guest_party_id: {
                "data": id_list_local_first,
                "cache_id": cache_id,
                "intersect_meta": self.get_intersect_method_meta(),
                "intersect_key": self.get_intersect_key()
            }}
        """
        cache_data = {self.guest_party_id: id_list_local_first}
        cache_meta = {self.guest_party_id: {"cache_id": cache_id,
                                            "intersect_meta": self.get_intersect_method_meta(),
                                            "intersect_key": self.get_intersect_key()}}
        return cache_data, cache_meta

    def get_intersect_doubly_encrypted_id_from_cache(self, data_instances, cache_data):
        id_list_remote_first = self.transfer_variable.id_ciphertext_list_exchange_g2h.get(idx=0)
        LOGGER.info("got id 1st ciphertext list from guest")

        # 2nd ID encrypt & send doubly encrypted guest ID list to guest
        id_list_remote_second = self._encrypt_id(id_list_remote_first,
                                                 self.commutative_cipher,
                                                 reserve_original_key=True)  # (EEg, Eg)
        LOGGER.info("encrypted guest id for the 2nd time")
        self.id_list_local_first = self.extract_cache_list(cache_data, self.guest_party_id)[0]
        self._sync_doubly_encrypted_id_list(id_list_remote_second)
