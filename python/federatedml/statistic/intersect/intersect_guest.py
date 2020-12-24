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
        self.role = consts.GUEST

    def get_host_prvkey_ids(self):
        host_prvkey_ids_list = self.transfer_variable.host_prvkey_ids.get(idx=-1)
        LOGGER.info("Not using cache, get host_prvkey_ids from all host")

        return host_prvkey_ids_list

    def get_host_pubkey_ids(self):
        host_pubkey_ids_list = self.transfer_variable.host_pubkey_ids.get(idx=-1)
        LOGGER.info("Not using cache, get host_pubkey_ids from all host")

        return host_pubkey_ids_list

    def sign_host_ids(self, host_pubkey_ids_list):
        LOGGER.info("Get host_pubkey_ids from host")
        # Process(signs) host ids to host
        guest_sign_host_ids_list = [host_pubkey_ids.map(lambda k, v:
                                                   (k, self.sign_id(k, self.d[i], self.n[i])))
                               for i, host_pubkey_ids in enumerate(host_pubkey_ids_list)]
        return guest_sign_host_ids_list

    def filter_intersect_ids(self, encrypt_intersect_ids):
        if len(encrypt_intersect_ids) > 1:
            raw_intersect_ids = [e.map(lambda k, v: (v, 1)) for e in encrypt_intersect_ids]
            intersect_ids = self.get_common_intersection(raw_intersect_ids)
        else:
            intersect_ids = encrypt_intersect_ids[0]
            intersect_ids = intersect_ids.map(lambda k, v: (v, 1))
        return intersect_ids

    def send_intersect_ids(self, encrypt_intersect_ids, intersect_ids):
        if len(self.host_party_id_list) > 1:
            for i, host_party_id in enumerate(self.host_party_id_list):
                remote_intersect_id = self.map_raw_id_to_encrypt_id(intersect_ids, encrypt_intersect_ids[i])
                self.transfer_variable.intersect_ids.remote(remote_intersect_id,
                                                            role=consts.HOST,
                                                            idx=i)
                LOGGER.info(f"Remote intersect ids to Host {host_party_id}!")
        else:
            remote_intersect_id = intersect_ids.mapValues(lambda v: 1)
            self.transfer_variable.intersect_ids.remote(remote_intersect_id,
                                                        role=consts.HOST,
                                                        idx=0)
            LOGGER.info(f"Remote intersect ids to Host!")

    def get_host_intersect_ids(self, guest_sign_host_ids_list):
        encrypt_intersect_ids_list = self.transfer_variable.intersect_ids.get(idx=-1)
        LOGGER.info("Get intersect ids from Host")
        intersect_ids_pair_list = [ids.join(guest_sign_host_ids_list[i],
                                            lambda e, h: h) for i, ids in enumerate(encrypt_intersect_ids_list)]
        intersect_ids = self.get_common_intersection(intersect_ids_pair_list)
        return intersect_ids

    def split_calculation_process(self, data_instances):
        # split data
        sid_hash_odd = data_instances.filter(lambda k, v: k & 1)
        sid_hash_even = data_instances.filter(lambda k, v: not k & 1)

        # generate pub keys for even ids
        self.e, self.d, self.n = self.generate_protocol_key()
        LOGGER.info("Generated guest rsa key!")
        # send public key e & n to all host
        for i, host_party_id in enumerate(self.host_party_id_list):
            guest_public_key = {"e": self.e[i], "n": self.n[i]}
            self.transfer_variable.guest_pubkey.remote(guest_public_key,
                                                       role=consts.HOST,
                                                       idx=i)
            LOGGER.info(f"Remote public key to Host {host_party_id}.")

        # generate ri for odd ids
        count = sid_hash_odd.count()
        self.r = self.generate_r_base(self.random_bit, count, self.random_base_fraction)

        # receive host pub keys for odds ids
        host_public_keys = self.transfer_variable.host_pubkey.get(-1)
        LOGGER.info("Get RSA host_public_key:{} from Host".format(host_public_keys))
        self.pub_e = [int(public_key["e"]) for public_key in host_public_keys]
        self.pub_n = [int(public_key["n"]) for public_key in host_public_keys]

        # send prvkey encrypted even ids to host
        prvkey_ids_process_pair = self.cal_prvkey_ids_process_pair(sid_hash_even)
        prvkey_ids_process = prvkey_ids_process_pair.mapValues(lambda v: 1)
        self.transfer_variable.guest_prvkey_ids.remote(prvkey_ids_process,
                                                       role=consts.HOST,
                                                       idx=-1)
        # get prvkey encrypted odds ids from host
        host_prvkey_ids_list = self.get_host_prvkey_ids()
        LOGGER.info("Get host_prvkey_ids")

        # encrypt own ids with pub keys from host
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

        # get & sign host pub key encrypted even ids
        host_pubkey_ids_list = self.get_host_pubkey_ids()
        guest_sign_host_ids_list = self.sign_host_ids(host_pubkey_ids_list)
        # send signed host even ids
        for i, host_party_id in enumerate(self.host_party_id_list):
            self.transfer_variable.guest_sign_host_ids.remote(guest_sign_host_ids_list[i],
                                                              role=consts.HOST,
                                                              idx=i)
            LOGGER.info(f"Remote host_sign_guest_ids_process to Host {host_party_id}.")

        # Recv host signed odd ids
        # table(guest_pubkey_id, host signed odd ids)
        recv_host_sign_guest_ids_list = self.transfer_variable.host_sign_guest_ids.get(idx=-1)
        LOGGER.info("Get host_sign_guest_ids from Host")

        # table(r^e % n *hash(sid), sid, hash(guest_ids_process/r))
        # g[0]=(r^e % n *hash(sid), sid), g[1]=random bits r
        host_sign_guest_ids_list = [v.join(recv_host_sign_guest_ids_list[i],
                                           lambda g, r: (g[0], RsaIntersectionGuest.hash(gmpy2.divm(int(r),
                                                                                                    int(g[1]),
                                                                                                    self.pub_n[i]),
                                                                                         self.final_hash_operator,
                                                                                         self.rsa_params.salt)))
                                    for i, v in enumerate(pubkey_id_process_list)]

        # table(hash(guest_ids_process/r), sid))
        sid_host_sign_guest_ids_list = [g.map(lambda k, v: (v[1], v[0])) for i, g in enumerate(host_sign_guest_ids_list)]

        # get intersect odd ids
        # intersect table(hash(guest_ids_process/r), sid)
        encrypt_intersect_odd_ids = [v.join(host_prvkey_ids_list[i], lambda sid, h: sid) for i, v in
                                     enumerate(sid_host_sign_guest_ids_list)]
        intersect_odd_ids = self.filter_intersect_ids(encrypt_intersect_odd_ids)
        intersect_even_ids = self.get_host_intersect_ids(guest_sign_host_ids_list)
        intersect_ids = intersect_odd_ids.union(intersect_even_ids)

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
        host_sign_guest_ids_list = [v.join(recv_host_sign_guest_ids_list[i],
                                          lambda g, r: (g[0], RsaIntersectionGuest.hash(gmpy2.divm(int(r),
                                                                                                   int(g[1]),
                                                                                                   self.pub_n[i]),
                                                                                        self.final_hash_operator,
                                                                                        self.rsa_params.salt)))
                                   for i, v in enumerate(pubkey_id_process_list)]

        # table(hash(guest_ids_process/r), sid))
        sid_host_sign_guest_ids_list = [
            g.map(lambda k, v: (v[1], v[0]))
            for i, g in enumerate(host_sign_guest_ids_list)]

        # intersect table(hash(guest_ids_process/r), sid)
        encrypt_intersect_ids = [v.join(host_prvkey_ids_list[i], lambda sid, h: sid) for i, v in
                                 enumerate(sid_host_sign_guest_ids_list)]

        intersect_ids = self.filter_intersect_ids(encrypt_intersect_ids)
        if self.sync_intersect_ids:
            self.send_intersect_ids(encrypt_intersect_ids, intersect_ids)
        else:
            LOGGER.info("Skip sync intersect ids with Host(s).")

        return intersect_ids


class RawIntersectionGuest(RawIntersect):
    def __init__(self):
        super().__init__()
        self.role = consts.GUEST

    def run_intersect(self, data_instances):
        LOGGER.info("Start raw intersection")

        if self.join_role == consts.HOST:
            intersect_ids = self.intersect_send_id(data_instances)
        elif self.join_role == consts.GUEST:
            intersect_ids = self.intersect_join_id(data_instances)
        else:
            raise ValueError("Unknown intersect join role, please check the configure of guest")

        return intersect_ids
