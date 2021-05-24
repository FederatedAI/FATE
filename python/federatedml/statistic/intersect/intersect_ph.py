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

from federatedml.statistic.intersect import PhIntersect
from federatedml.util import consts, LOGGER


class PhIntersectionGuest(PhIntersect):
    def __init__(self):
        super().__init__()
        self.role = consts.GUEST

    def _sync_commutative_cipher_public_knowledge(self):
        for i, _ in enumerate(self.host_party_id_list):
            self.transfer_variable.commutative_cipher_public_knowledge.remote(self.commutative_cipher[i],
                                                                              role=consts.HOST,
                                                                              idx=i)
            LOGGER.info(f"sent commutative cipher public knowledge to {i}th host")

    def _exchange_id_list(self, id_list):
        for i, _ in enumerate(self.host_party_id_list):
            self.transfer_variable.id_ciphertext_list_exchange_g2h.remote(id_list[i],
                                                                          role=consts.HOST,
                                                                          idx=i)
            LOGGER.info(f"sent id 1st ciphertext list to {i} th host")
        id_list_remote = self.transfer_variable.id_ciphertext_list_exchange_h2g.get(idx=-1)

        LOGGER.info("got id ciphertext list from all host")
        return id_list_remote

    def _sync_doubly_encrypted_id_list(self, id_list=None):
        id_list_guest = self.transfer_variable.doubly_encrypted_id_list.get(idx=-1)
        LOGGER.info("got doubly encrypted id list from all host")
        return id_list_guest

    def get_intersect_id(self, data_instances):
        self._init_commutative_cipher()
        self._sync_commutative_cipher_public_knowledge()
        #@TODO: init for multiple guest
        for cipher in self.commutative_cipher:
            cipher.init()
        LOGGER.info("commutative cipher key generated")

        # 1st ID encrypt: # (Eg, -1)
        id_list_local_first = [self._encrypt_id(data_instances, cipher) for cipher in self.commutative_cipher]
        LOGGER.info("encrypted guest id for the 1st time")
        id_list_remote_first = self._exchange_id_list(id_list_local_first)

        # 2nd ID encrypt & receive doubly encrypted ID list
        id_list_remote_second = [self._encrypt_id(id_list_remote_first[i],
                                                  self.commutive_cipher[i],
                                                  reserve_original_key=True) for i in range(len(id_list_remote_first))]
        LOGGER.info("encrypted remote id for the 2nd time")

        # receive doubly encrypted ID list from all host: # (Eh, EEh)
        id_list_remote_second_only = []
        for id_list in id_list_remote_second:
            id_list_remote_second_only.append(id_list.map(lambda k, v: (v, -1)))  # (EEh, -1)
        id_list_local_second = self._sync_doubly_encrypted_id_list()  # get (EEg, -1)

        # find intersection
        id_list_intersect = self._find_intersection(id_list_local_second, id_list_remote_second_only)  # (EEi, -1)
        LOGGER.info("intersection found")


class PhIntersectionHost(PhIntersect):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST

    def _sync_commutative_cipher_public_knowledge(self):
        self.commutative_cipher = self.transfer_variable.commutative_cipher_public_knowledge.get(idx=0)

        LOGGER.info(f"got commutative cipher public knowledge from guest")

    def _exchange_id_list(self, id_list):
        self.transfer_variable.id_ciphertext_list_exchange_h2g.remote(id_list,
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

    def find_intersect(self, data_instances):
        LOGGER.info("Start PH intersection")
        self._sync_commutative_cipher_public_knowledge()
        self.commutative_cipher.init()

        # 1st ID encrypt: (h, (Eh, Instance))
        id_list_host_first = self._encrypt_id(data_instances, self.commutative_cipher, reserve_value=True)
        LOGGER.info("encrypted host id for the 1st time")
        # send (Eh, -1), get (Eg, -1)
        id_list_guest_first = self._exchange_id_list(id_list_host_first.map(lambda k, v: (v[0], -1)))

        # 2nd ID encrypt & send doubly encrypted guest ID list to guest
        id_list_guest_second = self._encrypt_id(id_list_guest_first)  # (EEg, -1)
        LOGGER.info("encrypted guest id for the 2nd time")
        self._sync_doubly_encrypted_id_list(id_list_guest_second)

        id_list_intersect_cipher = self._decrypt_id(
            id_list_intersect_cipher_cipher, reserve_value=True)
    pass
