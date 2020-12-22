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

import gmpy2

from federatedml.secureprotol import gmpy_math
from federatedml.statistic.intersect import RawIntersect
from federatedml.statistic.intersect import RsaIntersect
from federatedml.util import consts
from federatedml.util import LOGGER


class RsaIntersectionGuest(RsaIntersect):
    def __init__(self):
        super().__init__()

        #self.random_bit = intersect_params.random_bit
        # self.e = None
        # self.n = None
        # parameter for intersection cache
        # self.intersect_cache_param = intersect_params.intersect_cache_param

    def map_raw_id_to_encrypt_id(self, raw_id_data, encrypt_id_data):
        encrypt_id_data_exchange_kv = encrypt_id_data.map(lambda k, v: (v, k))
        encrypt_raw_id = raw_id_data.join(encrypt_id_data_exchange_kv, lambda r, e: e)
        encrypt_common_id = encrypt_raw_id.map(lambda k, v: (v, "id"))

        return encrypt_common_id

    def get_host_prvkey_ids(self):
        host_prvkey_ids_list = self.transfer_variable.host_prvkey_ids.get(idx=-1)
        LOGGER.info("Not using cache, get host_prvkey_ids from all host")

        return host_prvkey_ids_list

    def split_calculation_process(self, data_instances):
        # split data
        sid_hash_odd = data_instances.filter(lambda k, v: k & 1)
        sid_hash_even = data_instances.filter(lambda k, v: not k & 1)

        self.e, self.d, self.n = self.generate_protocol_key()
        LOGGER.info("Generated guest rsa key!")
        guest_public_key = {"e": self.e, "n": self.n}

        # sends public key e & n to host
        self.transfer_variable.guest_pubkey.remote(guest_public_key,
                                                      role=consts.HOST,
                                                      idx=0)
        LOGGER.info("Remote public key to Host.")

        # generate ri
        count = sid_hash_odd.count()
        self.r = self.generate_r_base(self.random_bit, count, self.random_base_fraction)

        host_public_keys = self.transfer_variable.host_pubkey.get(-1)
        LOGGER.info("Get RSA host_public_key:{} from Host".format(host_public_keys))
        self.pub_e = [int(public_key["e"]) for public_key in host_public_keys]
        self.pub_n = [int(public_key["n"]) for public_key in host_public_keys]

        prvkey_ids_process_pair = self.cal_prvkey_ids_process_pair(sid_hash_even)
        prvkey_ids_process = prvkey_ids_process_pair.mapValues(lambda v: 1)
        self.transfer_variable.guest_sign_ids.remote(prvkey_ids_process,
                                                     role=consts.HOST,
                                                     idx=-1)

        pubkey_id_process_list = [data_instances.map(
            lambda k, v: self.pubkey_id_process(k,
                                                v,
                                                random_bit=self.random_bit,
                                                rsa_e=self.pub_e[i],
                                                rsa_n=self.pub_n[i],
                                                rsa_r=self.r)) for i in range(len(self.pub_e))]
        for i, guest_id in enumerate(pubkey_id_process_list):
            mask_guest_id = guest_id.mapValues(lambda v: 1)
            self.transfer_variable.guest_pubkey_ids.remote(mask_guest_id,
                                                           role=consts.HOST,
                                                           idx=i)
            LOGGER.info("Remote guest_mask_ids to Host {}".format(i))

        host_prvkey_ids_list = self.get_host_prvkey_ids()
        LOGGER.info("Get host_prvkey_ids")
        # Recv signed guest ids
        # table(r^e % n *hash(sid), guest_id_process)
        recv_host_sign_guest_ids_list = self.transfer_variable.host_sign_guest_ids.get(idx=-1)
        LOGGER.info("Get host_sign_guest_ids from Host")

        # table(r^e % n *hash(sid), sid, hash(guest_ids_process/r))
        # g[0]=(r^e % n *hash(sid), sid), g[1]=random bits r
        guest_ids_process_final = [v.join(recv_host_sign_guest_ids_list[i], lambda g, r: (
        g[0], RsaIntersectionGuest.hash(gmpy2.divm(int(r), int(g[1]), self.n[i]),
                                        self.final_hash_operator, self.rsa_params.salt)))
                                   for i, v in enumerate(host_prvkey_ids_list)]

        # table(hash(guest_ids_process/r), sid))
        sid_guest_ids_process_final = [
            g.map(lambda k, v: (v[1], v[0]))
            for i, g in enumerate(guest_ids_process_final)]

        # intersect table(hash(guest_ids_process/r), sid)
        encrypt_intersect_ids = [v.join(host_prvkey_ids_list[i], lambda sid, h: sid) for i, v in
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
        #@todo: union odd & even intersect_ids
        return intersect_ids

    def unified_calculation_process(self, data_instances):
        # generate r
        count = data_instances.count()
        self.r = self.generate_r_base(self.random_bit, count, self.random_base_fraction)

        # receives public key e & n
        public_keys = self.transfer_variable.host_pubkey.get(-1)
        LOGGER.info(f"Get RSA host_public_key:{public_keys} from Host")
        self.pub_e = [int(public_key["e"]) for public_key in public_keys]
        self.pub_n = [int(public_key["n"]) for public_key in public_keys]

        pubkey_id_process_list = [data_instances.map(
            lambda k, v: self.pubkey_id_process(k, v,
                                                random_bit=self.random_bit,
                                                rsa_e=self.pub_e[i],
                                                rsa_n=self.pub_n[i],
                                                rsa_r=self.r)) for i in range(len(self.pub_e))]

        for i, guest_id in enumerate(pubkey_id_process_list):
            mask_guest_id = guest_id.mapValues(lambda v: 1)
            self.transfer_variable.guest_pubkey_ids.remote(mask_guest_id,
                                                         role=consts.HOST,
                                                         idx=i)
            LOGGER.info("Remote guest_mask_ids to Host {}".format(i))

        host_prvkey_ids_list = self.get_host_prvkey_ids()
        LOGGER.info("Get host_prvkey_ids")

        # Recv signed guest ids
        # table(r^e % n *hash(sid), guest_id_process)
        recv_host_sign_guest_ids_list = self.transfer_variable.host_sign_guest_ids.get(idx=-1)
        LOGGER.info("Get host_sign_guest_ids from Host")

        # table(r^e % n *hash(sid), sid, hash(guest_ids_process/r))
        # g[0]=(r^e % n *hash(sid), sid), g[1]=random bits r
        guest_final_sign_ids = [v.join(recv_host_sign_guest_ids_list[i],
                                          lambda g, r: (g[0], RsaIntersectionGuest.hash(gmpy2.divm(int(r),
                                                                                                   int(g[1]),
                                                                                                   self.n[i]),
                                                                                        self.final_hash_operator,
                                                                                        self.rsa_params.salt)))
                                   for i, v in enumerate(host_prvkey_ids_list)]

        # table(hash(guest_ids_process/r), sid))
        sid_guest_final_sign_ids = [
            g.map(lambda k, v: (v[1], v[0]))
            for i, g in enumerate(guest_final_sign_ids)]

        # intersect table(hash(guest_ids_process/r), sid)
        encrypt_intersect_ids = [v.join(host_prvkey_ids_list[i], lambda sid, h: sid) for i, v in
                                 enumerate(sid_guest_final_sign_ids)]

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

    def run_intersect(self, data_instances):
        LOGGER.info("Start raw intersection")

        if self.join_role == consts.HOST:
            intersect_ids = self.intersect_send_id(data_instances)
        elif self.join_role == consts.GUEST:
            intersect_ids = self.intersect_join_id(data_instances)
        else:
            raise ValueError("Unknown intersect join role, please check the configure of guest")

        return intersect_ids
