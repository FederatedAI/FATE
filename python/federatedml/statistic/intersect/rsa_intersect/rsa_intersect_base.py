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

import random

from federatedml.param.intersect_param import DEFAULT_RANDOM_BIT
from federatedml.secureprotol import gmpy_math
from federatedml.secureprotol.encrypt import RsaEncrypt
from federatedml.secureprotol.hash.hash_factory import Hash
from federatedml.statistic.intersect import Intersect
from federatedml.transfer_variable.transfer_class.rsa_intersect_transfer_variable import RsaIntersectTransferVariable
from federatedml.util import consts, LOGGER


class RsaIntersect(Intersect):
    def __init__(self):
        super().__init__()
        # self.intersect_cache_param = intersect_params.intersect_cache_param
        self.rcv_e = None
        self.rcv_n = None
        self.e = None
        self.d = None
        self.n = None
        self.p = None
        self.q = None
        self.cp = None
        self.cq = None
        # self.r = None
        self.transfer_variable = RsaIntersectTransferVariable()
        self.role = None

    def load_params(self, param):
        # self.only_output_key = param.only_output_key
        # self.sync_intersect_ids = param.sync_intersect_ids
        super().load_params(param=param)
        self.rsa_params = param.rsa_params
        self.random_bit = self.rsa_params.random_bit
        """
        if param.random_bit is not None and self.random_bit == DEFAULT_RANDOM_BIT:
            self.random_bit = param.random_bit
            LOGGER.warning(f"param 'random_bit' of IntersectParam will be deprecated in future version, "
                           f"please use 'random_bit' in RSAParams.")
        """
        self.split_calculation = self.rsa_params.split_calculation
        self.random_base_fraction = self.rsa_params.random_base_fraction
        self.first_hash_operator = Hash(self.rsa_params.hash_method, False)
        self.final_hash_operator = Hash(self.rsa_params.final_hash_method, False)
        self.salt = self.rsa_params.salt

    def get_intersect_method_meta(self):
        rsa_meta = {"intersect_method": self.intersect_method,
                    "hash_method": self.rsa_params.hash_method,
                    "final_hash_method": self.rsa_params.final_hash_method,
                    "salt": self.salt,
                    "random_bit": self.random_bit}
        return rsa_meta

    @staticmethod
    def extend_pair(v1, v2):
        return v1 + v2

    @staticmethod
    def pubkey_id_process(data, fraction, random_bit, rsa_e, rsa_n, hash_operator=None, salt=''):
        if fraction and fraction <= consts.MAX_BASE_FRACTION:
            LOGGER.debug(f"fraction value: {fraction} provided, use fraction in pubkey id process")
            count = max(round(data.count() * max(fraction, consts.MIN_BASE_FRACTION)), 1)

            def group_kv(kv_iterator):
                res = []
                for k, v in kv_iterator:
                    if hash_operator is not None:
                        v = (k, v)
                        k = int(Intersect.hash(k, hash_operator, salt), 16)
                    res.append((k % count, [(k, v)]))
                return res

            reduced_pair_group = data.mapReducePartitions(group_kv, RsaIntersect.extend_pair)

            def pubkey_id_generate(k, pair):
                r = random.SystemRandom().getrandbits(random_bit)
                r_e = gmpy_math.powmod(r, rsa_e, rsa_n)
                for hash_sid, v in pair:
                    processed_id = r_e * hash_sid % rsa_n
                    yield processed_id, (v[0], r)

            return reduced_pair_group.flatMap(pubkey_id_generate)
        else:
            LOGGER.debug(f"fraction not provided or invalid, fraction value: {fraction}.")
            return data.map(lambda k, v: RsaIntersect.pubkey_id_process_per(k, v, random_bit, rsa_e, rsa_n,
                                                                            hash_operator, salt))

    @staticmethod
    def generate_rsa_key(rsa_bit=1024):
        LOGGER.info(f"Generate {rsa_bit}-bit RSA key.")
        encrypt_operator = RsaEncrypt()
        encrypt_operator.generate_key(rsa_bit)
        return encrypt_operator.get_key_pair()

    def generate_protocol_key(self):
        if self.role == consts.HOST:
            self.e, self.d, self.n, self.p, self.q = self.generate_rsa_key(self.rsa_params.key_length)
            self.cp, self.cq = gmpy_math.crt_coefficient(self.p, self.q)
        else:
            e, d, n, p, q, cp, cq = [], [], [], [], [], [], []
            for i in range(len(self.host_party_id_list)):
                e_i, d_i, n_i, p_i, q_i = self.generate_rsa_key(self.rsa_params.key_length)
                cp_i, cq_i = gmpy_math.crt_coefficient(p_i, q_i)
                e.append(e_i)
                d.append(d_i)
                n.append(n_i)
                p.append(p_i)
                q.append(q_i)
                cp.append(cp_i)
                cq.append(cq_i)
            self.e = e
            self.d = d
            self.n = n
            self.p = p
            self.q = q
            self.cp = cp
            self.cq = cq

    @staticmethod
    def pubkey_id_process_per(hash_sid, v, random_bit, rsa_e, rsa_n, hash_operator=None, salt=''):
        r = random.SystemRandom().getrandbits(random_bit)
        if hash_operator:
            processed_id = gmpy_math.powmod(r, rsa_e, rsa_n) * \
                int(Intersect.hash(hash_sid, hash_operator, salt), 16) % rsa_n
            return processed_id, (hash_sid, r)
        else:
            processed_id = gmpy_math.powmod(r, rsa_e, rsa_n) * hash_sid % rsa_n
            return processed_id, (v[0], r)

    @staticmethod
    def prvkey_id_process(
            hash_sid,
            v,
            rsa_d,
            rsa_n,
            rsa_p,
            rsa_q,
            cp,
            cq,
            final_hash_operator,
            salt,
            first_hash_operator=None):
        if first_hash_operator:
            processed_id = Intersect.hash(gmpy_math.powmod_crt(int(Intersect.hash(
                hash_sid, first_hash_operator, salt), 16), rsa_d, rsa_n, rsa_p, rsa_q, cp, cq), final_hash_operator, salt)
            return processed_id, hash_sid
        else:
            processed_id = Intersect.hash(gmpy_math.powmod_crt(hash_sid, rsa_d, rsa_n, rsa_p, rsa_q, cp, cq),
                                          final_hash_operator,
                                          salt)
            return processed_id, v[0]

    def cal_prvkey_ids_process_pair(self, data_instances, d, n, p, q, cp, cq, first_hash_operator=None):
        return data_instances.map(
            lambda k, v: self.prvkey_id_process(k, v, d, n, p, q, cp, cq,
                                                self.final_hash_operator,
                                                self.rsa_params.salt,
                                                first_hash_operator)
        )

    @staticmethod
    def sign_id(hash_sid, rsa_d, rsa_n, rsa_p, rsa_q, cp, cq):
        return gmpy_math.powmod_crt(hash_sid, rsa_d, rsa_n, rsa_p, rsa_q, cp, cq)

    def split_calculation_process(self, data_instances):
        raise NotImplementedError("This method should not be called here")

    def unified_calculation_process(self, data_instances):
        raise NotImplementedError("This method should not be called here")

    def cache_unified_calculation_process(self, data_instances, cache_set):
        raise NotImplementedError("This method should not be called here")

    def run_intersect(self, data_instances):
        LOGGER.info("Start RSA Intersection")
        if self.split_calculation:
            # H(k), (k, v)
            hash_data_instances = data_instances.map(
                lambda k, v: (int(Intersect.hash(k, self.first_hash_operator, self.salt), 16), (k, v)))
            intersect_ids = self.split_calculation_process(hash_data_instances)
        else:
            intersect_ids = self.unified_calculation_process(data_instances)
        return intersect_ids

    def run_cache_intersect(self, data_instances, cache_data):
        LOGGER.info("Start RSA Intersection with cache")
        if self.split_calculation:
            LOGGER.warning(f"split_calculation not applicable to cache-enabled RSA intersection.")
        intersect_ids = self.cache_unified_calculation_process(data_instances, cache_data)
        return intersect_ids
