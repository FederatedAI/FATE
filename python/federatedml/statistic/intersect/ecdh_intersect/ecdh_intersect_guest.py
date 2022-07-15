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

from federatedml.statistic.intersect.ecdh_intersect.ecdh_intersect_base import EcdhIntersect
from federatedml.util import consts, LOGGER


class EcdhIntersectionGuest(EcdhIntersect):
    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
        self.id_local_first = None
        self.id_remote_second = None
        self.id_local_second = None

    def _exchange_id(self, id, replace_val=True):
        if replace_val:
            id_only = id.mapValues(lambda v: None)
        else:
            id_only = id
        self.transfer_variable.id_ciphertext_exchange_g2h.remote(id_only,
                                                                 role=consts.HOST,
                                                                 idx=0)
        LOGGER.info(f"sent id 1st ciphertext to host")
        id_remote = self.transfer_variable.id_ciphertext_exchange_h2g.get(idx=-1)

        LOGGER.info("got id ciphertext from host")
        return id_remote

    def _sync_doubly_encrypted_id(self, id=None):
        id_guest = self.transfer_variable.doubly_encrypted_id.get(idx=-1)
        LOGGER.info("got doubly encrypted id list from host")
        return id_guest

    def send_intersect_ids(self, intersect_ids):
        remote_intersect_id = intersect_ids.map(lambda k, v: (v, None))
        self.transfer_variable.intersect_ids.remote(remote_intersect_id,
                                                    role=consts.HOST,
                                                    idx=0)
        LOGGER.info(f"Remote intersect ids to Host!")

    def get_intersect_doubly_encrypted_id(self, data_instances, keep_key=True):
        self.init_curve()
        LOGGER.info(f"curve instance obtained")

        # 1st ID encrypt: # (Eg, -1)
        self.id_local_first = self._encrypt_id(data_instances,
                                               self.curve_instance,
                                               reserve_original_key=keep_key,
                                               hash_operator=self.hash_operator,
                                               salt=self.salt)
        LOGGER.info("encrypted guest id for the 1st time")
        id_remote_first = self._exchange_id(self.id_local_first, keep_key)[0]

        # 2nd ID encrypt & receive doubly encrypted ID list: # (EEh, Eh)
        self.id_remote_second = self._sign_id(id_remote_first,
                                              self.curve_instance,
                                              reserve_original_key=keep_key)
        LOGGER.info("encrypted remote id for the 2nd time")

        # receive doubly encrypted ID list from all host:
        self.id_local_second = self._sync_doubly_encrypted_id()[0]  # get (EEg, Eg)

        # find intersection per host
        id_intersect_cipher_cipher = self.extract_intersect_ids(self.id_remote_second,
                                                                self.id_local_second,
                                                                keep_both=keep_key)  # (EEi, [Eh, Eg])
        LOGGER.info("encrypted intersection ids found")

        return id_intersect_cipher_cipher

    def decrypt_intersect_doubly_encrypted_id(self, id_intersect_cipher_cipher):
        # EEi -> (Eg, Eh)
        id_intersect_cipher = id_intersect_cipher_cipher.map(lambda k, v: (v[1], v[0]))

        # find intersect ids: (Eg, [Eh, original id])
        encrypt_intersect_ids = self.extract_intersect_ids(id_intersect_cipher,
                                                           self.id_local_first,
                                                           keep_both=True)
        # (original_id, Eh)
        intersect_ids = encrypt_intersect_ids.map(lambda k, v: (v[1], v[0]))
        LOGGER.info(f"intersection found")

        if self.sync_intersect_ids:
            self.send_intersect_ids(intersect_ids)
        else:
            LOGGER.info("Skip sync intersect ids with Host(s).")
        intersect_ids = intersect_ids.mapValues(lambda v: None)
        return intersect_ids

    def get_intersect_key(self, party_id):
        intersect_key = {"curve_key": self.curve_instance.get_curve_key().decode("latin1")}
        return intersect_key

    def load_intersect_key(self, cache_meta):
        host_party = self.host_party_id_list[0]
        intersect_key = cache_meta[str(host_party)]["intersect_key"]
        curve_key = intersect_key["curve_key"].encode("latin1")
        self.init_curve(curve_key)

    def generate_cache(self, data_instances):
        self.init_curve()
        LOGGER.info(f"curve instance obtained")

        cache_id = self.cache_transfer_variable.get(idx=-1)
        LOGGER.info(f"got cache_id from host")

        id_remote_first = self.transfer_variable.id_ciphertext_exchange_h2g.get(idx=-1)[0]
        LOGGER.info("Get id ciphertext list from host")

        # 2nd ID encrypt & receive doubly encrypted ID list: # (EEh, Eh)
        id_remote_second = [self._sign_id(id_remote_first,
                                          self.curve_instance,
                                          reserve_original_key=True)]
        LOGGER.info("encrypted remote id for the 2nd time")

        cache_data, cache_meta = {}, {}
        intersect_meta = self.get_intersect_method_meta()
        for i, party_id in enumerate(self.host_party_id_list):
            meta = {"cache_id": cache_id[i],
                    "intersect_meta": intersect_meta,
                    "intersect_key": self.get_intersect_key(party_id)}
            cache_meta[party_id] = meta
            cache_data[party_id] = id_remote_second[i]

        return cache_data, cache_meta

    def get_intersect_doubly_encrypted_id_from_cache(self, data_instances, cache_data):
        self.id_local_first = self._encrypt_id(data_instances,
                                               self.curve_instance,
                                               reserve_original_key=True,
                                               hash_operator=self.hash_operator,
                                               salt=self.salt)
        LOGGER.info("encrypted guest id for the 1st time")

        id_only = self.id_local_first.mapValues(lambda v: None)
        self.transfer_variable.id_ciphertext_exchange_g2h.remote(id_only,
                                                                 role=consts.HOST,
                                                                 idx=0)
        LOGGER.info(f"sent id 1st ciphertext to host")

        # receive doubly encrypted ID from host:
        self.id_local_second = self._sync_doubly_encrypted_id()[0]  # get (EEg, Eg)

        # find intersection
        cache_host = self.extract_cache_list(cache_data, self.host_party_id_list)[0]
        id_intersect_cipher_cipher = self.extract_intersect_ids(cache_host,
                                                                self.id_local_second,
                                                                keep_both=True)  # (EEi, [Eh, Eg])
        LOGGER.info("encrypted intersection ids found")
        self.id_remote_second = cache_host

        return id_intersect_cipher_cipher

    def run_cardinality(self, data_instances):
        LOGGER.info(f"run cardinality_only with ECDH")
        if len(self.host_party_id_list) > 1:
            raise ValueError(f"cardinality_only with ECDH only supports single-host task.")
        id_intersect_cipher_cipher = self.get_intersect_doubly_encrypted_id(data_instances, keep_key=False)
        self.intersect_num = id_intersect_cipher_cipher.count()
        if self.sync_cardinality:
            self.transfer_variable.cardinality.remote(self.intersect_num, role=consts.HOST, idx=-1)
            LOGGER.info("Sent intersect cardinality to host.")
        else:
            LOGGER.info("Skip sync intersect cardinality with host")
