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


class PhIntersectionGuest(PhIntersect):
    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
        self.id_list_local_first = None
        self.id_list_remote_second = None
        self.id_list_local_second = None
        self.host_count = None
        # self.recorded_k_data = None

    def _sync_commutative_cipher_public_knowledge(self):
        for i, _ in enumerate(self.host_party_id_list):
            self.transfer_variable.commutative_cipher_public_knowledge.remote(self.commutative_cipher[i],
                                                                              role=consts.HOST,
                                                                              idx=i)
            LOGGER.info(f"sent commutative cipher public knowledge to {i}th host")

    def _exchange_id_list(self, id_list):
        for i, id in enumerate(id_list):
            id_only = id.map(lambda k, v: (k, -1))
            self.transfer_variable.id_ciphertext_list_exchange_g2h.remote(id_only,
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

    def sync_intersect_cipher_cipher(self, id_list):
        for i in range(len(id_list)):
            self.transfer_variable.intersect_cipher_cipher.remote(id_list[i],
                                                                  role=consts.HOST,
                                                                  idx=i)
            LOGGER.info(f"sent intersect cipher cipher to {i}th host")

    def sync_intersect_cipher(self, id_list=None):
        id_list_intersect_cipher = self.transfer_variable.intersect_cipher.get(idx=-1)
        LOGGER.info("got intersect cipher from all host")
        return id_list_intersect_cipher

    @staticmethod
    def map_cipher_cipher_to_cipher_id(id_cipher_cipher, id_cipher):
        common_cipher_id = id_cipher.join(id_cipher_cipher, lambda r, e: r)
        cipher_id = common_cipher_id.map(lambda k, v: (v, k))
        return cipher_id

    def get_intersect_cipher(self, id_list_intersect_cipher_cipher):
        id_list_intersect_cipher = [
            self.map_cipher_cipher_to_cipher_id(id_list_intersect_cipher_cipher[i],
                                                self.id_list_local_second[i]) for i in range(self.host_count)]
        return id_list_intersect_cipher

    def _get_remote_cipher_intersect_ids(self, intersect_ids):
        host_count = len(self.id_list_local_first)
        first_cipher_local_ids_list = [intersect_ids.map(lambda k, v: (v[i], 1)) for i in range(host_count)] # Ei
        doubly_encrypted_ids_list = [self.map_raw_id_to_encrypt_id(first_cipher_local_ids_list[i],           # EEi
                                                                   self.id_list_local_second[i]) for i in
                                     range(host_count)]
        # EEi to Eh
        first_remote_cipher_ids_list = [self.map_cipher_cipher_to_cipher_id(doubly_encrypted_ids_list[i],
                                                                            self.id_list_remote_second[i]) for i in
                                       range(host_count)]
        return first_remote_cipher_ids_list

    def send_intersect_ids(self, intersect_ids):
        remote_intersect_id_list = self._get_remote_cipher_intersect_ids(intersect_ids)
        for i, host_party_id in enumerate(self.host_party_id_list):
            self.transfer_variable.intersect_ids.remote(remote_intersect_id_list[i],
                                                        role=consts.HOST,
                                                        idx=i)
            LOGGER.info(f"Remote intersect ids to Host {host_party_id}!")

    def get_intersect_doubly_encrypted_id(self, data_instances):
        self._generate_commutative_cipher()
        self._sync_commutative_cipher_public_knowledge()
        self.host_count = len(self.commutative_cipher)

        for cipher in self.commutative_cipher:
            cipher.init()
        LOGGER.info("commutative cipher key generated")

        # 1st ID encrypt: # (Eg, -1)
        self.id_list_local_first = [self._encrypt_id(data_instances,
                                                     cipher,
                                                     reserve_original_key=True,
                                                     hash_operator=self.hash_operator,
                                                     salt=self.salt) for cipher in self.commutative_cipher]
        LOGGER.info("encrypted guest id for the 1st time")
        id_list_remote_first = self._exchange_id_list(self.id_list_local_first)

        # 2nd ID encrypt & receive doubly encrypted ID list: # (EEh, Eh)
        self.id_list_remote_second = [self._encrypt_id(id_list_remote_first[i],
                                                       self.commutative_cipher[i],
                                                       reserve_original_key=True) for i in range(self.host_count)]
        LOGGER.info("encrypted remote id for the 2nd time")

        # receive doubly encrypted ID list from all host:
        self.id_list_local_second = self._sync_doubly_encrypted_id_list()  # get (EEg, Eg)

        # find intersection per host
        id_list_intersect_cipher_cipher = [self.extract_intersect_ids(self.id_list_remote_second[i],
                                                                      self.id_list_local_second[i])
                                           for i in range(self.host_count)]  # (EEi, -1)
        LOGGER.info("encrypted intersection ids found")

        return id_list_intersect_cipher_cipher

    def decrypt_intersect_doubly_encrypted_id(self, id_list_intersect_cipher_cipher):
        # EEi -> Ei from Eg
        id_list_intersect_cipher = self.get_intersect_cipher(id_list_intersect_cipher_cipher)

        # find intersect ids: (Ei, original key)
        encrypt_intersect_ids = [
            self.extract_intersect_ids(id_list_intersect_cipher[i],
                                       self.id_list_local_first[i]) for i in range(len(self.id_list_local_first))
        ]
        # map encrypted intersect ids to original ids
        intersect_ids = self.filter_intersect_ids(encrypt_intersect_ids, keep_encrypt_ids=True)
        LOGGER.info(f"intersection found")

        if self.sync_intersect_ids:
            self.send_intersect_ids(intersect_ids)
        else:
            LOGGER.info("Skip sync intersect ids with Host(s).")

        return intersect_ids
