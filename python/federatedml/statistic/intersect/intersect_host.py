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

import hashlib

# from fate_arch.session import computing_session as session
from federatedml.secureprotol import gmpy_math
from federatedml.secureprotol.hash.hash_factory import Hash
from federatedml.statistic.intersect import RawIntersect
from federatedml.statistic.intersect import RsaIntersect
from federatedml.util import consts
from federatedml.util import LOGGER


class RsaIntersectionHost(RsaIntersect):
    def __init__(self):
        super().__init__()
        # self.transfer_variable = RsaIntersectTransferVariable()
        # parameter for intersection cache
        # self.is_version_match = False
        # self.has_cache_version = True

    def cal_host_ids_process_pair(self, data_instances):
        hash_operator = Hash(self.rsa_params.hash_method, self.rsa_params.base64)
        final_hash_operator = Hash(self.rsa_params.final_hash_method, self.rsa_params.base64)
        return data_instances.map(
            lambda k, v: (
                RsaIntersectionHost.hash(gmpy_math.powmod(int(RsaIntersectionHost.hash(k, hash_operator, self.rsa_params.salt), 16), self.d, self.n),
                                         final_hash_operator,
                                         self.rsa_params.salt), k)
        )

    def host_ids_process(self, data_instances):
        # (host_id_process, 1)
        LOGGER.info("Not using cache, calculate Za using raw id")
        host_ids_process_pair = self.cal_host_ids_process_pair(data_instances)

        return host_ids_process_pair

    def split_calculation_process(self, data_instances):
        pass

    def unified_calculation_process(self, data_instances):
        pass

    def fit(self, data_instances):
        LOGGER.info("Start rsa intersection")
        # generate rsa keys
        self.e, self.d, self.n = self.get_rsa_key()
        LOGGER.info("Generated host rsa key!")
        host_public_key = {"e": self.e, "n": self.n}
        if self.split_calculation and self.random_base_fraction:
            count = data_instances.count()
            self.r  = self.generate_r_base(self.random_bit, count, self.random_base_fraction)

        # sends public key e & n to guest
        self.transfer_variable.host_rsa_pubkey.remote(host_public_key,
                                                 role=consts.GUEST,
                                                 idx=0)
        LOGGER.info("Remote public key to Guest.")
        # hash host ids
        host_ids_process_pair = self.host_ids_process(data_instances)

        host_ids_process = host_ids_process_pair.mapValues(lambda v: 1)
        self.transfer_variable.intersect_host_ids_process.remote(host_ids_process,
                                                                 role=consts.GUEST,
                                                                 idx=0)
        LOGGER.info("Remote host_ids_process to Guest.")

        # Recv guest ids
        guest_ids = self.transfer_variable.intersect_guest_ids.get(idx=0)
        LOGGER.info("Get guest_ids from guest")

        # Process(signs) guest ids and return to guest
        guest_ids_process = guest_ids.map(lambda k, v: (k, gmpy_math.powmod(int(k), self.d, self.n)))
        self.transfer_variable.intersect_guest_ids_process.remote(guest_ids_process,
                                                                  role=consts.GUEST,
                                                                  idx=0)
        LOGGER.info("Remote guest_ids_process to Guest.")

        # recv intersect ids
        intersect_ids = None
        if self.sync_intersect_ids:
            encrypt_intersect_ids = self.transfer_variable.intersect_ids.get(idx=0)
            intersect_ids_pair = encrypt_intersect_ids.join(host_ids_process_pair, lambda e, h: h)
            intersect_ids = intersect_ids_pair.map(lambda k, v: (v, "id"))
            LOGGER.info("Get intersect ids from Guest")

            if not self.only_output_key:
                intersect_ids = self._get_value_from_data(intersect_ids, data_instances)

        return intersect_ids


class RawIntersectionHost(RawIntersect):
    def __init__(self, intersect_params):
        super().__init__(intersect_params)
        self.join_role = intersect_params.join_role
        self.role = consts.HOST

    def run(self, data_instances):
        LOGGER.info("Start raw intersection")

        if self.join_role == consts.GUEST:
            intersect_ids = self.intersect_send_id(data_instances)
        elif self.join_role == consts.HOST:
            intersect_ids = self.intersect_join_id(data_instances)
        else:
            raise ValueError("Unknown intersect join role, please check the configure of host")

        return intersect_ids
