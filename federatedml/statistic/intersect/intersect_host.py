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
from arch.api.federation import remote, get
from arch.api.utils import log_utils
from federatedml.secureprotol import gmpy_math
from federatedml.secureprotol.encrypt import RsaEncrypt
from federatedml.statistic.intersect import Intersect
from federatedml.util import consts
from federatedml.util.transfer_variable import RawIntersectTransferVariable
from federatedml.util.transfer_variable import RsaIntersectTransferVariable

LOGGER = log_utils.getLogger()


class RsaIntersectionHost(Intersect):
    def __init__(self, intersect_params):
        self.get_intersect_ids_flag = intersect_params.is_get_intersect_ids
        self.transfer_variable = RsaIntersectTransferVariable()

        self.e = None
        self.d = None
        self.n = None

    @staticmethod
    def hash(value):
        return hashlib.sha256(bytes(str(value), encoding='utf-8')).hexdigest()

    def run(self, data_instances):
        LOGGER.info("Start ras intersection")

        encrypt_operator = RsaEncrypt()
        encrypt_operator.generate_key(rsa_bit=1028)
        self.e, self.d, self.n = encrypt_operator.get_key_pair()
        LOGGER.info("Generate rsa keys.")
        public_key = {"e": self.e, "n": self.n}
        remote(public_key,
               name=self.transfer_variable.rsa_pubkey.name,
               tag=self.transfer_variable.generate_transferid(self.transfer_variable.rsa_pubkey),
               role=consts.GUEST,
               idx=0)
        LOGGER.info("Remote public key to Guest.")

        # (host_id_process, 1)
        table_host_ids_process = data_instances.map(
            lambda k, v: (
                RsaIntersectionHost.hash(gmpy_math.powmod(int(RsaIntersectionHost.hash(k), 16), self.d, self.n)), 1)
        )
        remote(table_host_ids_process,
               name=self.transfer_variable.intersect_host_ids_process.name,
               tag=self.transfer_variable.generate_transferid(self.transfer_variable.intersect_host_ids_process),
               role=consts.GUEST,
               idx=0)
        LOGGER.info("Remote host_ids_process to Guest.")

        # Recv guest ids
        guest_ids = get(name=self.transfer_variable.intersect_guest_ids.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.intersect_guest_ids),
                        idx=0)
        LOGGER.info("Get guest_ids from guest")

        # Process guest ids and return to guest
        guest_ids_process = guest_ids.map(lambda k, v: (k, gmpy_math.powmod(int(k), self.d, self.n)))
        remote(guest_ids_process,
               name=self.transfer_variable.intersect_guest_ids_process.name,
               tag=self.transfer_variable.generate_transferid(self.transfer_variable.intersect_guest_ids_process),
               role=consts.GUEST,
               idx=0)
        LOGGER.info("Remote guest_ids_process to Guest.")

        # recv intersect ids
        intersect_ids = None
        if self.get_intersect_ids_flag:
            intersect_ids = get(name=self.transfer_variable.intersect_ids.name,
                                tag=self.transfer_variable.generate_transferid(self.transfer_variable.intersect_ids),
                                idx=0)
            LOGGER.info("Get intersect ids from Guest")
        return intersect_ids


class RawIntersectionHost(Intersect):
    def __init__(self, intersect_params):
        self.get_intersect_id_flag = intersect_params.is_get_intersect_ids
        self.transfer_variable = RawIntersectTransferVariable()

    def run(self, data_instances):
        LOGGER.info("Start raw intersection")
        data_sid = data_instances.mapValues(lambda v: 1)

        remote(data_sid,
               name=self.transfer_variable.intersect_host_ids.name,
               tag=self.transfer_variable.generate_transferid(self.transfer_variable.intersect_host_ids),
               role=consts.GUEST,
               idx=0)

        LOGGER.info("Remote data_sid to Guest")
        intersect_ids = None
        if self.get_intersect_id_flag:
            intersect_ids = get(name=self.transfer_variable.intersect_ids.name,
                                tag=self.transfer_variable.generate_transferid(self.transfer_variable.intersect_ids),
                                idx=0)
            LOGGER.info("Get intersect ids from Guest")
        else:
            LOGGER.info("Not Get intersect ids from Guest")
            
        return intersect_ids
