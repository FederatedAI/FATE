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

from federatedml.statistic.intersect.ph_intersect.ph_intersect_base import PhIntersect
from federatedml.util import consts, LOGGER


class PhIntersectionHost(PhIntersect):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST
        self.id_list_local_first = None

    def _sync_commutative_cipher_public_knowledge(self):
        self.commutative_cipher = self.transfer_variable.commutative_cipher_public_knowledge.get(idx=0)

        LOGGER.info(f"got commutative cipher public knowledge from guest")

    def _exchange_id_list(self, id_list):
        id_only = id_list.map(lambda k, v: (k, -1))
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
        intersect_ids = self.map_encrypt_id_to_raw_id(first_cipher_intersect_ids, self.id_list_local_first)
        return intersect_ids

    def get_intersect_doubly_encrypted_id(self, data_instances):
        LOGGER.info("Start PH intersection")
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
        intersect_ids = None
        if self.sync_intersect_ids:
            intersect_ids = self.get_intersect_ids()

        return intersect_ids