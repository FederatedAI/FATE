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

from federatedml.homo.utils.scatter import scatter
from federatedml.util.transfer_variable.base_transfer_variable import Variable
from federatedml.util import consts


class _Arbiter(object):

    def __init__(self, guest_pw_trv, host_pw_trv):
        self._guest_pw_trv = guest_pw_trv
        self._host_pw_trv = host_pw_trv

    def get(self):
        weights = list(scatter(self._host_pw_trv, self._guest_pw_trv))
        total = sum(weights)
        return [x / total for x in weights]


class _Client(object):
    def __init__(self, trv: Variable):
        self._trv = trv

    def send(self, obj):
        self._trv.remote(obj=obj, role=consts.ARBITER, idx=0)


def arbiter(guest_pw_trv, host_pw_trv):
    return _Arbiter(guest_pw_trv, host_pw_trv)


def host(host_pw_trv):
    return _Client(host_pw_trv)


def guest(guest_pw_trv):
    return _Client(guest_pw_trv)
