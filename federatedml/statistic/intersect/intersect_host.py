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

from arch.api import session
from arch.api.utils import log_utils
from federatedml.secureprotol import gmpy_math
from federatedml.secureprotol.encrypt import RsaEncrypt
from federatedml.statistic.intersect.rsa_cache import cache_utils
from federatedml.statistic.intersect import RawIntersect
from federatedml.statistic.intersect import RsaIntersect
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.rsa_intersect_transfer_variable import RsaIntersectTransferVariable

LOGGER = log_utils.getLogger()


class RsaIntersectionHost(RsaIntersect):
    def __init__(self, intersect_params):
        super().__init__(intersect_params)
        self.transfer_variable = RsaIntersectTransferVariable()

        self.e = None
        self.d = None
        self.n = None

        # parameter for intersection cache
        self.is_version_match = False
        self.has_cache_version = True

    def cal_host_ids_process_pair(self, data_instances: session.table) -> session.table:
        return data_instances.map(
            lambda k, v: (
                RsaIntersectionHost.hash(gmpy_math.powmod(int(RsaIntersectionHost.hash(k), 16), self.d, self.n)), k)
        )

    def generate_rsa_key(self, rsa_bit=1024):
        encrypt_operator = RsaEncrypt()
        encrypt_operator.generate_key(rsa_bit)
        return encrypt_operator.get_key_pair()

    def get_rsa_key(self):
        if self.intersect_cache_param.use_cache:
            LOGGER.info("Using intersection cache scheme, start to getting rsa key from cache.")
            rsa_key = cache_utils.get_rsa_of_current_version(host_party_id=self.host_party_id,
                                                             id_type=self.intersect_cache_param.id_type,
                                                             encrypt_type=self.intersect_cache_param.encrypt_type,
                                                             tag='Za')
            if rsa_key is not None:
                e = int(rsa_key.get('rsa_e'))
                d = int(rsa_key.get('rsa_d'))
                n = int(rsa_key.get('rsa_n'))
            else:
                self.has_cache_version = False
                LOGGER.info("Use cache but can not find any version in cache, set has_cache_version to false")
                LOGGER.info("Start to generate rsa key")
                e, d, n = self.generate_rsa_key()
        else:
            LOGGER.info("Not use cache, generate rsa keys.")
            e, d, n = self.generate_rsa_key()
        return e, d, n

    def store_cache(self, host_id, rsa_key: dict, assign_version=None, assign_namespace=None):
        store_cache_ret = cache_utils.store_cache(dtable=host_id,
                                                  guest_party_id=None,
                                                  host_party_id=self.host_party_id,
                                                  version=assign_version,
                                                  id_type=self.intersect_cache_param.id_type,
                                                  encrypt_type=self.intersect_cache_param.encrypt_type,
                                                  tag='Za',
                                                  namespace=assign_namespace)
        LOGGER.info("Finish store host_ids_process to cache")

        version = store_cache_ret.get('table_name')
        namespace = store_cache_ret.get('namespace')
        cache_utils.store_rsa(host_party_id=self.host_party_id,
                              id_type=self.intersect_cache_param.id_type,
                              encrypt_type=self.intersect_cache_param.encrypt_type,
                              tag='Za',
                              namespace=namespace,
                              version=version,
                              rsa=rsa_key
                              )
        LOGGER.info("Finish store rsa key to cache")

        return version, namespace

    def host_ids_process(self, data_instances):
        # (host_id_process, 1)
        if self.intersect_cache_param.use_cache:
            LOGGER.info("Use intersect cache.")
            if self.has_cache_version:
                current_version = cache_utils.host_get_current_verison(host_party_id=self.host_party_id,
                                                                       id_type=self.intersect_cache_param.id_type,
                                                                       encrypt_type=self.intersect_cache_param.encrypt_type,
                                                                       tag='Za')
                version = current_version.get('table_name')
                namespace = current_version.get('namespace')
                guest_current_version = self.transfer_variable.cache_version_info.get(0)
                LOGGER.info("current_version:{}".format(current_version))
                LOGGER.info("guest_current_version:{}".format(guest_current_version))

                if guest_current_version.get('table_name') == version \
                        and guest_current_version.get('namespace') == namespace and \
                        current_version is not None:
                    self.is_version_match = True
                else:
                    self.is_version_match = False

                version_match_info = {'version_match': self.is_version_match,
                                      'version': version,
                                      'namespace': namespace}
                self.transfer_variable.cache_version_match_info.remote(version_match_info,
                                                                       role=consts.GUEST,
                                                                       idx=0)

                host_ids_process_pair = None
                if not self.is_version_match or self.sync_intersect_ids:
                    # if self.sync_intersect_ids is true, host will get the encrypted intersect id from guest,
                    # which need the Za to decrypt them
                    LOGGER.info("read Za from cache")
                    host_ids_process_pair = session.table(name=version,
                                                          namespace=namespace,
                                                          create_if_missing=True,
                                                          error_if_exist=False)
                    if host_ids_process_pair.count() == 0:
                        host_ids_process_pair = self.cal_host_ids_process_pair(data_instances)
                        rsa_key = {'rsa_e': self.e, 'rsa_d': self.d, 'rsa_n': self.n}
                        self.store_cache(host_ids_process_pair, rsa_key=rsa_key)
            else:
                self.is_version_match = False
                LOGGER.info("is version_match:{}".format(self.is_version_match))
                namespace = cache_utils.gen_cache_namespace(id_type=self.intersect_cache_param.id_type,
                                                            encrypt_type=self.intersect_cache_param.encrypt_type,
                                                            tag='Za',
                                                            host_party_id=self.host_party_id)
                version = cache_utils.gen_cache_version(namespace=namespace,
                                                        create=True)
                version_match_info = {'version_match': self.is_version_match,
                                      'version': version,
                                      'namespace': namespace}
                self.transfer_variable.cache_version_match_info.remote(version_match_info,
                                                                       role=consts.GUEST,
                                                                       idx=0)

                host_ids_process_pair = self.cal_host_ids_process_pair(data_instances)
                rsa_key = {'rsa_e': self.e, 'rsa_d': self.d, 'rsa_n': self.n}
                self.store_cache(host_ids_process_pair, rsa_key=rsa_key, assign_version=version, assign_namespace=namespace)

            LOGGER.info("remote version match info to guest")
        else:
            LOGGER.info("Not using cache, calculate Za using raw id")
            host_ids_process_pair = self.cal_host_ids_process_pair(data_instances)

        return host_ids_process_pair

    def run(self, data_instances):
        LOGGER.info("Start rsa intersection")
        self.e, self.d, self.n = self.get_rsa_key()
        LOGGER.info("Get rsa key!")
        public_key = {"e": self.e, "n": self.n}

        self.transfer_variable.rsa_pubkey.remote(public_key,
                                                 role=consts.GUEST,
                                                 idx=0)
        LOGGER.info("Remote public key to Guest.")
        host_ids_process_pair = self.host_ids_process(data_instances)

        if self.intersect_cache_param.use_cache and not self.is_version_match or not self.intersect_cache_param.use_cache:
            host_ids_process = host_ids_process_pair.mapValues(lambda v: 1)
            self.transfer_variable.intersect_host_ids_process.remote(host_ids_process,
                                                                     role=consts.GUEST,
                                                                     idx=0)
            LOGGER.info("Remote host_ids_process to Guest.")

        # Recv guest ids
        guest_ids = self.transfer_variable.intersect_guest_ids.get(idx=0)
        LOGGER.info("Get guest_ids from guest")

        # Process guest ids and return to guest
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
