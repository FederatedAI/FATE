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
from collections import Iterable

import gmpy2
import random

from arch.api import session
from arch.api.utils import log_utils
from federatedml.secureprotol import gmpy_math
from federatedml.statistic.intersect import RawIntersect
from federatedml.statistic.intersect import RsaIntersect
from federatedml.statistic.intersect.rsa_cache import cache_utils
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.rsa_intersect_transfer_variable import RsaIntersectTransferVariable

LOGGER = log_utils.getLogger()


class RsaIntersectionGuest(RsaIntersect):
    def __init__(self, intersect_params):
        super().__init__(intersect_params)

        self.random_bit = intersect_params.random_bit

        self.e = None
        self.n = None
        self.transfer_variable = RsaIntersectTransferVariable()

        # parameter for intersection cache
        self.intersect_cache_param = intersect_params.intersect_cache_param

    def map_raw_id_to_encrypt_id(self, raw_id_data, encrypt_id_data):
        encrypt_id_data_exchange_kv = encrypt_id_data.map(lambda k, v: (v, k))
        encrypt_raw_id = raw_id_data.join(encrypt_id_data_exchange_kv, lambda r, e: e)
        encrypt_common_id = encrypt_raw_id.map(lambda k, v: (v, "id"))

        return encrypt_common_id

    def get_cache_version_match_info(self):
        if self.intersect_cache_param.use_cache:
            LOGGER.info("Use cache is true")
            # check local cache version for each host
            for i, host_party_id in enumerate(self.host_party_id_list):
                current_version = cache_utils.guest_get_current_version(host_party_id=host_party_id,
                                                                        guest_party_id=None,
                                                                        id_type=self.intersect_cache_param.id_type,
                                                                        encrypt_type=self.intersect_cache_param.encrypt_type,
                                                                        tag='Za'
                                                                        )
                LOGGER.info("host_id:{}, current_version:{}".format(host_party_id, current_version))
                if current_version is None:
                    current_version = {"table_name": None, "namespace": None}

                self.transfer_variable.cache_version_info.remote(current_version,
                                                                 role=consts.HOST,
                                                                 idx=i)
                LOGGER.info("Remote current version to host:{}".format(host_party_id))

            cache_version_match_info = self.transfer_variable.cache_version_match_info.get(idx=-1)
            LOGGER.info("Get cache version match info:{}".format(cache_version_match_info))
        else:
            cache_version_match_info = None
            LOGGER.info("Not using cache, cache_version_match_info is None")

        return cache_version_match_info

    def get_host_id_process(self, cache_version_match_info):
        if self.intersect_cache_param.use_cache:
            host_ids_process_list = [None for _ in self.host_party_id_list]

            if isinstance(cache_version_match_info, Iterable):
                for i, version_info in enumerate(cache_version_match_info):
                    if version_info.get('version_match'):
                        host_ids_process_list[i] = session.table(name=version_info.get('version'),
                                                                 namespace=version_info.get('namespace'),
                                                                 create_if_missing=True,
                                                                 error_if_exist=False)
                        LOGGER.info("Read host {} 's host_ids_process from cache".format(self.host_party_id_list[i]))
            else:
                LOGGER.info("cache_version_match_info is not iterable, not use cache_version_match_info")

            # check which host_id_process is not receive yet
            host_ids_process_rev_idx_list = [ i for i, e in enumerate(host_ids_process_list) if e is None ]

            if len(host_ids_process_rev_idx_list) > 0:
                # Recv host_ids_process
                # table(host_id_process, 1)
                host_ids_process = []
                for rev_idx in host_ids_process_rev_idx_list:
                    host_ids_process.append(self.transfer_variable.intersect_host_ids_process.get(idx=rev_idx))
                    LOGGER.info("Get host_ids_process from host {}".format(self.host_party_id_list[rev_idx]))

                for i, host_idx in enumerate(host_ids_process_rev_idx_list):
                    host_ids_process_list[host_idx] = host_ids_process[i]

                    version = cache_version_match_info[host_idx].get('version')
                    namespace = cache_version_match_info[host_idx].get('namespace')

                    cache_utils.store_cache(dtable=host_ids_process[i],
                                            guest_party_id=self.guest_party_id,
                                            host_party_id=self.host_party_id_list[i],
                                            version=version,
                                            id_type=self.intersect_cache_param.id_type,
                                            encrypt_type=self.intersect_cache_param.encrypt_type,
                                            tag=consts.INTERSECT_CACHE_TAG,
                                            namespace=namespace
                                            )
                    LOGGER.info("Store host {}'s host_ids_process to cache.".format(self.host_party_id_list[host_idx]))
        else:
            host_ids_process_list = self.transfer_variable.intersect_host_ids_process.get(idx=-1)
            LOGGER.info("Not using cache, get host_ids_process from all host")

        return host_ids_process_list

    @staticmethod
    def guest_id_process(sid, random_bit, rsa_e, rsa_n):
        r = random.SystemRandom().getrandbits(random_bit)
        re_hash = gmpy_math.powmod(r, rsa_e, rsa_n) * int(RsaIntersectionGuest.hash(sid), 16) % rsa_n
        return re_hash, (sid, r)

    def run(self, data_instances):
        LOGGER.info("Start rsa intersection")

        public_keys = self.transfer_variable.rsa_pubkey.get(-1)
        LOGGER.info("Get RSA public_key:{} from Host".format(public_keys))
        self.e = [int(public_key["e"]) for public_key in public_keys]
        self.n = [int(public_key["n"]) for public_key in public_keys]

        cache_version_match_info = self.get_cache_version_match_info()

        # table (r^e % n * hash(sid), sid, r)
        guest_id_process_list = [ data_instances.map(
                lambda k, v: self.guest_id_process(k, random_bit=self.random_bit, rsa_e=self.e[i], rsa_n=self.n[i])) for i in range(len(self.e)) ]

        # table(r^e % n *hash(sid), 1)
        for i, guest_id in enumerate(guest_id_process_list):
            mask_guest_id = guest_id.mapValues(lambda v: 1)
            self.transfer_variable.intersect_guest_ids.remote(mask_guest_id,
                                                              role=consts.HOST,
                                                              idx=i)
            LOGGER.info("Remote guest_id to Host {}".format(i))

        host_ids_process_list = self.get_host_id_process(cache_version_match_info)
        LOGGER.info("Get host_ids_process")

        # Recv process guest ids
        # table(r^e % n *hash(sid), guest_id_process)
        recv_guest_ids_process = self.transfer_variable.intersect_guest_ids_process.get(idx=-1)
        LOGGER.info("Get guest_ids_process from Host")

        # table(r^e % n *hash(sid), sid, hash(guest_ids_process/r))
        guest_ids_process_final = [v.join(recv_guest_ids_process[i], lambda g, r: (g[0], RsaIntersectionGuest.hash(gmpy2.divm(int(r), int(g[1]), self.n[i]))))
                                   for i, v in enumerate(guest_id_process_list)]

        # table(hash(guest_ids_process/r), sid))
        sid_guest_ids_process_final = [
            g.map(lambda k, v: (v[1], v[0]))
            for i, g in enumerate(guest_ids_process_final)]

        # intersect table(hash(guest_ids_process/r), sid)
        encrypt_intersect_ids = [v.join(host_ids_process_list[i], lambda sid, h: sid) for i, v in
                                 enumerate(sid_guest_ids_process_final)]

        if len(self.host_party_id_list) > 1:
            raw_intersect_ids = [e.map(lambda k, v: (v, 1)) for e in encrypt_intersect_ids]
            intersect_ids = self.get_common_intersection(raw_intersect_ids)

            # send intersect id
            if self.sync_intersect_ids:
                for i, host_party_id in enumerate(self.host_party_id_list):
                    remote_intersect_id = self.map_raw_id_to_encrypt_id(intersect_ids, encrypt_intersect_ids[i])
                    self.transfer_variable.intersect_ids.remote(remote_intersect_id,
                                                                role=consts.HOST,
                                                                idx=i)
                    LOGGER.info("Remote intersect ids to Host {}!".format(host_party_id))
            else:
                LOGGER.info("Not send intersect ids to Host!")
        else:
            intersect_ids = encrypt_intersect_ids[0]
            if self.sync_intersect_ids:
                remote_intersect_id = intersect_ids.mapValues(lambda v: 1)
                self.transfer_variable.intersect_ids.remote(remote_intersect_id,
                                                            role=consts.HOST,
                                                            idx=0)

            intersect_ids = intersect_ids.map(lambda k, v: (v, 1))

        LOGGER.info("Finish intersect_ids computing")

        if not self.only_output_key:
            intersect_ids = self._get_value_from_data(intersect_ids, data_instances)

        return intersect_ids


class RawIntersectionGuest(RawIntersect):
    def __init__(self, intersect_params):
        super().__init__(intersect_params)
        self.role = consts.GUEST
        self.join_role = intersect_params.join_role

    def run(self, data_instances):
        LOGGER.info("Start raw intersection")

        if self.join_role == consts.HOST:
            intersect_ids = self.intersect_send_id(data_instances)
        elif self.join_role == consts.GUEST:
            intersect_ids = self.intersect_join_id(data_instances)
        else:
            raise ValueError("Unknown intersect join role, please check the configure of guest")

        return intersect_ids
