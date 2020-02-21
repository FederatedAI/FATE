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

    def register_user_num_transfer(self, transfer_variable):
        pass


class Client(object):

    def __init__(self):
        self.host_user_num_sync = None
        self.guest_user_num_sync = None

    def register_user_num_transfer(self, transfer_variable):
        self.host_user_num_sync = transfer_variable.host_user_num
        self.guest_user_num_sync = transfer_variable.guest_user_num
        return self


class Host(Client):

    def send_host_user_num(self, user_num, suffix=tuple()):
        self.host_user_num_sync.remote(obj=user_num, role=consts.GUEST, idx=0, suffix=suffix)

    def get_guest_user_num(self, suffix=tuple()):
        return self.guest_user_num_sync.get(idx=0, suffix=suffix)


class Guest(Client):

    def send_guest_user_num(self, user_num, suffix=tuple()):
        self.guest_user_num_sync.remote(obj=user_num, role=consts.HOST, idx=0, suffix=suffix)

    def get_host_user_num(self, suffix=tuple()):
        host_user_num = self.host_user_num_sync.get(idx=0, suffix=suffix)
        return host_user_num


