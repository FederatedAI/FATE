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

from federatedml.statistic.intersect.ecdh_intersect.ecdh_intersect_base import EcdhIntersect
from federatedml.util import consts, LOGGER


class EcdhIntersectionHost(EcdhIntersect):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST
        self.id_local_first = None

    def _exchange_id(self, id, replace_val=True):
        if replace_val:
            id_only = id.mapValues(lambda v: None)
        else:
            id_only = id
        self.transfer_variable.id_ciphertext_exchange_h2g.remote(id_only,
                                                                 role=consts.GUEST,
                                                                 idx=0)
        LOGGER.info("sent id 1st ciphertext list to guest")

        id_guest = self.transfer_variable.id_ciphertext_exchange_g2h.get(idx=0)
        LOGGER.info("got id 1st ciphertext list from guest")

        return id_guest

    def _sync_doubly_encrypted_id(self, id):
        self.transfer_variable.doubly_encrypted_id.remote(id,
                                                          role=consts.GUEST,
                                                          idx=0)
        LOGGER.info("sent doubly encrypted id list to guest")

    def get_intersect_ids(self):
        first_cipher_intersect_ids = self.transfer_variable.intersect_ids.get(idx=0)
        LOGGER.info(f"obtained cipher intersect ids from guest")
        intersect_ids = self.map_encrypt_id_to_raw_id(first_cipher_intersect_ids,
                                                      self.id_local_first,
                                                      keep_encrypt_id=False)
        return intersect_ids

    def get_intersect_doubly_encrypted_id(self, data_instances, keep_key=True):
        self.init_curve()
        LOGGER.info(f"curve instance obtained")

        # 1st ID encrypt: (Eh, (h, Instance))
        self.id_local_first = self._encrypt_id(data_instances,
                                               self.curve_instance,
                                               reserve_original_key=keep_key,
                                               hash_operator=self.hash_operator,
                                               salt=self.salt,
                                               reserve_original_value=keep_key)
        LOGGER.info("encrypted local id for the 1st time")
        # send (Eh, -1), get (Eg, -1)
        id_remote_first = self._exchange_id(self.id_local_first, keep_key)

        # 2nd ID encrypt & send doubly encrypted guest ID list to guest
        id_remote_second = self._sign_id(id_remote_first,
                                         self.curve_instance,
                                         reserve_original_key=keep_key)  # (EEg, Eg)
        LOGGER.info("encrypted guest id for the 2nd time")
        self._sync_doubly_encrypted_id(id_remote_second)

    def decrypt_intersect_doubly_encrypted_id(self, id_intersect_cipher_cipher=None):
        intersect_ids = None
        if self.sync_intersect_ids:
            intersect_ids = self.get_intersect_ids()

        return intersect_ids

    def get_intersect_key(self, party_id=None):
        intersect_key = {"curve_key": self.curve_instance.get_curve_key().decode("latin1")}
        return intersect_key

    def load_intersect_key(self, cache_meta):
        intersect_key = cache_meta[str(self.guest_party_id)]["intersect_key"]
        curve_key = intersect_key["curve_key"].encode("latin1")
        self.init_curve(curve_key)

    def generate_cache(self, data_instances):
        self.init_curve()
        LOGGER.info(f"curve instance obtained")

        cache_id = str(uuid.uuid4())
        self.cache_id = {self.guest_party_id: cache_id}
        self.cache_transfer_variable.remote(cache_id, role=consts.GUEST, idx=0)
        LOGGER.info(f"remote cache_id to guest")

        # 1st ID encrypt: (Eh, (h, Instance))
        id_local_first = self._encrypt_id(data_instances,
                                          self.curve_instance,
                                          reserve_original_key=True,
                                          hash_operator=self.hash_operator,
                                          salt=self.salt,
                                          reserve_original_value=True)
        LOGGER.info("encrypted local id for the 1st time")

        id_only = id_local_first.mapValues(lambda v: None)
        self.transfer_variable.id_ciphertext_exchange_h2g.remote(id_only,
                                                                 role=consts.GUEST,
                                                                 idx=0)
        LOGGER.info("sent id 1st ciphertext list to guest")

        cache_data = {self.guest_party_id: id_local_first}
        cache_meta = {self.guest_party_id: {"cache_id": cache_id,
                                            "intersect_meta": self.get_intersect_method_meta(),
                                            "intersect_key": self.get_intersect_key()}}
        return cache_data, cache_meta

    def get_intersect_doubly_encrypted_id_from_cache(self, data_instances, cache_data):
        id_remote_first = self.transfer_variable.id_ciphertext_exchange_g2h.get(idx=0)
        LOGGER.info("got id 1st ciphertext from guest")

        # 2nd ID encrypt & send doubly encrypted guest ID to guest
        id_remote_second = self._sign_id(id_remote_first,
                                         self.curve_instance,
                                         reserve_original_key=True)  # (EEg, Eg)
        LOGGER.info("encrypted guest id for the 2nd time")
        self.id_local_first = self.extract_cache_list(cache_data, self.guest_party_id)[0]
        self._sync_doubly_encrypted_id(id_remote_second)

    def run_cardinality(self, data_instances):
        LOGGER.info(f"run exact_cardinality with DH")
        self.get_intersect_doubly_encrypted_id(data_instances, keep_key=False)
        if self.sync_cardinality:
            self.intersect_num = self.transfer_variable.cardinality.get(idx=0)
            LOGGER.info("Got intersect cardinality from guest.")
