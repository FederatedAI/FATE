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
import abc

from federatedml.secureprotol.spdz.utils import NamingService


class TensorBase(object):
    __array_ufunc__ = None

    def __init__(self, q_field, tensor_name: str = None):
        self.q_field = q_field
        self.tensor_name = NamingService.get_instance().next() if tensor_name is None else tensor_name

    @classmethod
    def get_spdz(cls):
        from federatedml.secureprotol.spdz import SPDZ
        return SPDZ.get_instance()

    @abc.abstractmethod
    def dot(self, other, target_name=None):
        pass
