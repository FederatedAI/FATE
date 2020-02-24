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

from federatedml.util import consts


class Arbiter(object):

    def register_average_rate_transfer(self, transfer_variable):
        pass


class Client(object):

    def __init__(self):
        self.host_average_rate_sync = None
        self.guest_average_rate_sync = None

    def register_average_rate_transfer(self, transfer_variable):
        self.host_average_rate_sync = transfer_variable.host_average_rate
        self.guest_average_rate_sync = transfer_variable.guest_average_rate
        return self


class Host(Client):

    def send_host_average_rate(self, average_rate, suffix=tuple()):
        self.host_average_rate_sync.remote(obj=average_rate, role=consts.GUEST, idx=0, suffix=suffix)

    def get_guest_average_rate(self, suffix=tuple()):
        return self.guest_average_rate_sync.get(idx=0, suffix=suffix)


class Guest(Client):

    def send_guest_average_rate(self, average_rate, suffix=tuple()):
        self.guest_average_rate_sync.remote(obj=average_rate, role=consts.HOST, idx=0, suffix=suffix)

    def get_host_average_rate(self, suffix=tuple()):
        host_average_rate = self.host_average_rate_sync.get(idx=0, suffix=suffix)
        return host_average_rate
        # assert (len(host_average_rate) == 1, "Current MF only support single host party")
        # return host_average_rate[0]


