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

from federatedml.secureprotol.fate_paillier import PaillierKeypair
from federatedml.secureprotol.spdz.communicator import Communicator
from federatedml.secureprotol.spdz.utils import NamingService
from federatedml.secureprotol.spdz.utils import naming


class SPDZ(object):
    __instance = None

    @classmethod
    def get_instance(cls) -> 'SPDZ':
        return cls.__instance

    @classmethod
    def set_instance(cls, instance):
        prev = cls.__instance
        cls.__instance = instance
        return prev

    @classmethod
    def has_instance(cls):
        return cls.__instance is not None

    def __init__(self, name="ss", q_field=2 << 60, local_party=None, all_parties=None):
        self.name_service = naming.NamingService(name)
        self._prev_name_service = None
        self._pre_instance = None

        self.communicator = Communicator(local_party, all_parties)

        self.party_idx = self.communicator.party_idx
        self.other_parties = self.communicator.other_parties
        if len(self.other_parties) > 1:
            raise EnvironmentError("support 2-party secret share only")
        self.public_key, self.private_key = PaillierKeypair.generate_keypair(1024)
        self.q_field = q_field

    def __enter__(self):
        self._prev_name_service = NamingService.set_instance(self.name_service)
        self._pre_instance = self.set_instance(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        NamingService.set_instance(self._pre_instance)
        # self.communicator.clean()

    def __reduce__(self):
        raise PermissionError("it's unsafe to transfer this")

    def partial_rescontruct(self):
        # todo: partial parties gets rescontructed tensor
        pass

    @classmethod
    def dot(cls, left, right, target_name=None):
        return left.dot(right, target_name)
