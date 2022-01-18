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

from federatedml.secureprotol.symmetric_encryption.cryptor_executor import CryptoExecutor
from federatedml.secureprotol.symmetric_encryption.pohlig_hellman_encryption import PohligHellmanCipherKey
from federatedml.statistic.intersect.dh_intersect.dh_intersect_base import DhIntersect
from federatedml.util import consts, LOGGER


class DhIntersectionGuest(DhIntersect):
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
            id_only = id.mapValues(lambda v: None)
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

    """
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
    """

    def send_intersect_ids(self, encrypt_intersect_ids_list, intersect_ids):
        if len(self.host_party_id_list) > 1:
            for i, host_party_id in enumerate(self.host_party_id_list):
                remote_intersect_id = intersect_ids.map(lambda k, v: (v[i], 1))
                self.transfer_variable.intersect_ids.remote(remote_intersect_id,
                                                            role=consts.HOST,
                                                            idx=i)
                LOGGER.info(f"Remote intersect ids to Host {host_party_id}!")
        else:
            remote_intersect_id = encrypt_intersect_ids_list[0].mapValues(lambda v: 1)
            self.transfer_variable.intersect_ids.remote(remote_intersect_id,
                                                        role=consts.HOST,
                                                        idx=0)
            LOGGER.info(f"Remote intersect ids to Host!")

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
                                                                      self.id_list_local_second[i],
                                                                      keep_both=True)
                                           for i in range(self.host_count)]  # (EEi, [Eh, Eg])
        LOGGER.info("encrypted intersection ids found")

        return id_list_intersect_cipher_cipher

    def decrypt_intersect_doubly_encrypted_id(self, id_list_intersect_cipher_cipher):
        # EEi -> (Eg, Eh)
        # id_list_intersect_cipher = self.get_intersect_cipher(id_list_intersect_cipher_cipher)
        id_list_intersect_cipher = [ids.map(lambda k, v: (v[1], v[0])) for ids in id_list_intersect_cipher_cipher]

        # find intersect ids: (Eg, [Eh, original id])
        encrypt_intersect_ids = [
            self.extract_intersect_ids(id_list_intersect_cipher[i],
                                       self.id_list_local_first[i],
                                       keep_both=True) for i in range(len(self.id_list_local_first))
        ]
        # (Eh, original id)
        encrypt_intersect_ids = [ids.map(lambda k, v: (v[0], v[1])) for ids in encrypt_intersect_ids]
        # map encrypted intersect ids to original ids: (original id, Eh)
        intersect_ids = self.filter_intersect_ids(encrypt_intersect_ids, keep_encrypt_ids=True)
        LOGGER.info(f"intersection found")

        if self.sync_intersect_ids:
            self.send_intersect_ids(encrypt_intersect_ids, intersect_ids)
        else:
            LOGGER.info("Skip sync intersect ids with Host(s).")
        intersect_ids = intersect_ids.mapValues(lambda v: 1)
        return intersect_ids

    def get_intersect_key(self, party_id):
        idx = self.host_party_id_list.index(party_id)
        cipher_core = self.commutative_cipher[idx].cipher_core

        intersect_key = {"mod_base": str(cipher_core.mod_base),
                         "exponent": str(cipher_core.exponent)}

        return intersect_key

    def load_intersect_key(self, cache_meta):
        commutative_cipher = []
        for host_party in self.host_party_id_list:
            intersect_key = cache_meta[str(host_party)]["intersect_key"]

            mod_base = int(intersect_key["mod_base"])
            exponent = int(intersect_key["exponent"])

            ph_key = PohligHellmanCipherKey(mod_base, exponent)
            commutative_cipher.append(CryptoExecutor(ph_key))

        self.commutative_cipher = commutative_cipher

    def generate_cache(self, data_instances):
        self._generate_commutative_cipher()
        self._sync_commutative_cipher_public_knowledge()
        self.host_count = len(self.commutative_cipher)

        for cipher in self.commutative_cipher:
            cipher.init()
        LOGGER.info("commutative cipher key generated")

        cache_id_list = self.cache_transfer_variable.get(idx=-1)
        LOGGER.info(f"got cache_id from all host")

        id_list_remote_first = self.transfer_variable.id_ciphertext_list_exchange_h2g.get(idx=-1)
        LOGGER.info("Get id ciphertext list from all host")

        # 2nd ID encrypt & receive doubly encrypted ID list: # (EEh, Eh)
        id_list_remote_second = [self._encrypt_id(id_list_remote_first[i],
                                                  self.commutative_cipher[i],
                                                  reserve_original_key=True) for i in range(self.host_count)]
        LOGGER.info("encrypted remote id for the 2nd time")

        cache_data, cache_meta = {}, {}
        intersect_meta = self.get_intersect_method_meta()
        for i, party_id in enumerate(self.host_party_id_list):
            meta = {"cache_id": cache_id_list[i],
                    "intersect_meta": intersect_meta,
                    "intersect_key": self.get_intersect_key(party_id)}
            cache_meta[party_id] = meta
            cache_data[party_id] = id_list_remote_second[i]

        # self.cache_id = cache_id_dict
        return cache_data, cache_meta

    def get_intersect_doubly_encrypted_id_from_cache(self, data_instances, cache_data):
        self.host_count = len(self.commutative_cipher)
        self.id_list_local_first = [self._encrypt_id(data_instances,
                                                     cipher,
                                                     reserve_original_key=True,
                                                     hash_operator=self.hash_operator,
                                                     salt=self.salt) for cipher in self.commutative_cipher]
        LOGGER.info("encrypted guest id for the 1st time")

        for i, id in enumerate(self.id_list_local_first):
            id_only = id.mapValues(lambda v: None)
            self.transfer_variable.id_ciphertext_list_exchange_g2h.remote(id_only,
                                                                          role=consts.HOST,
                                                                          idx=i)
            LOGGER.info(f"sent id 1st ciphertext list to {i} th host")

        # receive doubly encrypted ID list from all host:
        self.id_list_local_second = self._sync_doubly_encrypted_id_list()  # get (EEg, Eg)

        # find intersection per host
        cache_list = self.extract_cache_list(cache_data, self.host_party_id_list)
        id_list_intersect_cipher_cipher = [self.extract_intersect_ids(cache_list[i],
                                                                      self.id_list_local_second[i],
                                                                      keep_both=True)
                                           for i in range(self.host_count)]  # (EEi, [Eh, Eg])
        LOGGER.info("encrypted intersection ids found")
        self.id_list_remote_second = cache_list

        return id_list_intersect_cipher_cipher
