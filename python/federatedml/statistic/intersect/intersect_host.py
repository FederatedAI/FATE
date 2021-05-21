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

from federatedml.statistic.intersect import RawIntersect, RsaIntersect
from federatedml.util import consts, LOGGER


class RsaIntersectionHost(RsaIntersect):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST

    def split_calculation_process(self, data_instances):
        LOGGER.info("RSA intersect using split calculation.")
        # split data
        sid_hash_odd = data_instances.filter(lambda k, v: k & 1)
        sid_hash_even = data_instances.filter(lambda k, v: not k & 1)
        # LOGGER.debug(f"sid_hash_odd count: {sid_hash_odd.count()},"
        #              f"odd fraction: {sid_hash_odd.count()/data_instances.count()}")

        # generate rsa keys
        self.e, self.d, self.n = self.generate_protocol_key()
        LOGGER.info("Generate host protocol key!")
        public_key = {"e": self.e, "n": self.n}

        # sends public key e & n to guest
        self.transfer_variable.host_pubkey.remote(public_key,
                                                  role=consts.GUEST,
                                                  idx=0)
        LOGGER.info("Remote public key to Guest.")

        # generate ri for even ids
        # count = sid_hash_even.count()
        # self.r = self.generate_r_base(self.random_bit, count, self.random_base_fraction)
        # LOGGER.info(f"Generate {len(self.r)} r values.")

        # receive guest key for even ids
        guest_public_key = self.transfer_variable.guest_pubkey.get(0)
        # LOGGER.debug("Get guest_public_key:{} from Guest".format(guest_public_key))
        LOGGER.info(f"Get guest_public_key from Guest")

        self.rcv_e = int(guest_public_key["e"])
        self.rcv_n = int(guest_public_key["n"])

        # encrypt & send guest pubkey-encrypted odd ids
        pubkey_ids_process = self.pubkey_id_process(sid_hash_even,
                                                    fraction=self.random_base_fraction,
                                                    random_bit=self.random_bit,
                                                    rsa_e=self.rcv_e,
                                                    rsa_n=self.rcv_n)
        LOGGER.info(f"Finish pubkey_ids_process")
        mask_host_id = pubkey_ids_process.mapValues(lambda v: 1)
        self.transfer_variable.host_pubkey_ids.remote(mask_host_id,
                                                      role=consts.GUEST,
                                                      idx=0)
        LOGGER.info("Remote host_pubkey_ids to Guest")

        # encrypt & send prvkey-encrypted host odd ids to guest
        prvkey_ids_process_pair = self.cal_prvkey_ids_process_pair(sid_hash_odd, self.d, self.n)
        prvkey_ids_process = prvkey_ids_process_pair.mapValues(lambda v: 1)

        self.transfer_variable.host_prvkey_ids.remote(prvkey_ids_process,
                                                      role=consts.GUEST,
                                                      idx=0)
        LOGGER.info("Remote host_prvkey_ids to Guest.")

        # get & sign guest pubkey-encrypted odd ids
        guest_pubkey_ids = self.transfer_variable.guest_pubkey_ids.get(idx=0)
        LOGGER.info(f"Get guest_pubkey_ids from guest")
        host_sign_guest_ids = guest_pubkey_ids.map(lambda k, v: (k, self.sign_id(k, self.d, self.n)))
        LOGGER.debug(f"host sign guest_pubkey_ids")
        # send signed guest odd ids
        self.transfer_variable.host_sign_guest_ids.remote(host_sign_guest_ids,
                                                          role=consts.GUEST,
                                                          idx=0)
        LOGGER.info("Remote host_sign_guest_ids_process to Guest.")

        # recv guest privkey-encrypted even ids
        guest_prvkey_ids = self.transfer_variable.guest_prvkey_ids.get(idx=0)
        LOGGER.info("Get guest_prvkey_ids")

        # receive guest-signed host even ids
        recv_guest_sign_host_ids = self.transfer_variable.guest_sign_host_ids.get(idx=0)
        LOGGER.info(f"Get guest_sign_host_ids from Guest.")
        guest_sign_host_ids = pubkey_ids_process.join(recv_guest_sign_host_ids,
                                                      lambda g, r: (g[0],
                                                                    RsaIntersectionHost.hash(gmpy2.divm(int(r),
                                                                                                        int(g[1]),
                                                                                                        self.rcv_n),
                                                                                             self.final_hash_operator,
                                                                                             self.rsa_params.salt)))
        sid_guest_sign_host_ids = guest_sign_host_ids.map(lambda k, v: (v[1], v[0]))

        encrypt_intersect_even_ids = sid_guest_sign_host_ids.join(guest_prvkey_ids, lambda sid, h: sid)

        # filter & send intersect even ids
        intersect_even_ids = self.filter_intersect_ids([encrypt_intersect_even_ids])

        remote_intersect_even_ids = encrypt_intersect_even_ids.mapValues(lambda v: 1)
        self.transfer_variable.host_intersect_ids.remote(remote_intersect_even_ids, role=consts.GUEST, idx=0)
        LOGGER.info(f"Remote host intersect ids to Guest")

        # recv intersect ids
        intersect_ids = None
        if self.sync_intersect_ids:
            encrypt_intersect_odd_ids = self.transfer_variable.intersect_ids.get(idx=0)
            intersect_odd_ids_pair = encrypt_intersect_odd_ids.join(prvkey_ids_process_pair, lambda e, h: h)
            intersect_odd_ids = intersect_odd_ids_pair.map(lambda k, v: (v, 1))
            intersect_ids = intersect_odd_ids.union(intersect_even_ids)
            LOGGER.info("Get intersect ids from Guest")
        return intersect_ids

    def unified_calculation_process(self, data_instances):
        LOGGER.info("RSA intersect using unified calculation.")
        # generate rsa keys
        self.e, self.d, self.n = self.generate_protocol_key()
        LOGGER.info("Generate protocol key!")
        public_key = {"e": self.e, "n": self.n}

        # sends public key e & n to guest
        self.transfer_variable.host_pubkey.remote(public_key,
                                                  role=consts.GUEST,
                                                  idx=0)
        LOGGER.info("Remote public key to Guest.")
        # hash host ids
        prvkey_ids_process_pair = self.cal_prvkey_ids_process_pair(data_instances,
                                                                   self.d,
                                                                   self.n,
                                                                   self.first_hash_operator)

        prvkey_ids_process = prvkey_ids_process_pair.mapValues(lambda v: 1)
        self.transfer_variable.host_prvkey_ids.remote(prvkey_ids_process,
                                                      role=consts.GUEST,
                                                      idx=0)
        LOGGER.info("Remote host_ids_process to Guest.")

        # Recv guest ids
        guest_pubkey_ids = self.transfer_variable.guest_pubkey_ids.get(idx=0)
        LOGGER.info("Get guest_pubkey_ids from guest")

        # Process(signs) guest ids and return to guest
        host_sign_guest_ids = guest_pubkey_ids.map(lambda k, v: (k, self.sign_id(k, self.d, self.n)))
        self.transfer_variable.host_sign_guest_ids.remote(host_sign_guest_ids,
                                                          role=consts.GUEST,
                                                          idx=0)
        LOGGER.info("Remote host_sign_guest_ids_process to Guest.")

        # recv intersect ids
        intersect_ids = None
        if self.sync_intersect_ids:
            encrypt_intersect_ids = self.transfer_variable.intersect_ids.get(idx=0)
            intersect_ids_pair = encrypt_intersect_ids.join(prvkey_ids_process_pair, lambda e, h: h)
            intersect_ids = intersect_ids_pair.map(lambda k, v: (v, "id"))
            LOGGER.info("Get intersect ids from Guest")

        return intersect_ids


class RawIntersectionHost(RawIntersect):
    def __init__(self):
        super().__init__()
        self.role = consts.HOST

    def run_intersect(self, data_instances):
        LOGGER.info("Start raw intersection")

        if self.join_role == consts.GUEST:
            intersect_ids = self.intersect_send_id(data_instances)
        elif self.join_role == consts.HOST:
            intersect_ids = self.intersect_join_id(data_instances)
        else:
            raise ValueError("Unknown intersect join role, please check job configuration")

        return intersect_ids
