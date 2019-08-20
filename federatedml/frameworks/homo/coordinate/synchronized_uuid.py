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

import uuid

from federatedml.frameworks.homo.coordinate.base import Coordinate
from federatedml.frameworks.homo.coordinate.transfer import arbiter_scatter, arbiter_broadcast
from federatedml.util.transfer_variable.homo_transfer_variable import HomeModelTransferVariable


class SynchronizedUUID(Coordinate):

    @staticmethod
    def from_transfer_variable(transfer_variable: HomeModelTransferVariable):
        host_name = transfer_variable.host_uuid.name
        host_tag = transfer_variable.generate_transferid(transfer_variable.host_uuid)
        guest_name = transfer_variable.guest_uuid.name
        guest_tag = transfer_variable.generate_transferid(transfer_variable.guest_uuid)
        conflict_flag_name = transfer_variable.converge_flag.name
        conflict_flag_tag = transfer_variable.generate_transferid(transfer_variable.converge_flag)
        return SynchronizedUUID(host_name=host_name, host_tag=host_tag,
                                guest_name=guest_name, guest_tag=guest_tag,
                                conflict_flag_name=conflict_flag_name, conflict_flag_tag=conflict_flag_tag)

    def __init__(self, host_name, host_tag, guest_name, guest_tag, conflict_flag_name, conflict_flag_tag):
        self._uuid_scatter = arbiter_scatter(host_name=host_name,
                                             host_tag=host_tag,
                                             guest_name=guest_name,
                                             guest_tag=guest_tag)

        self._conflict_flag_broadcast = arbiter_broadcast(name=conflict_flag_name, tag=conflict_flag_tag)

    def guest_call(self):
        _uid = uuid.uuid1()
        ind = 0
        flag = True
        while flag:
            self._uuid_scatter.remote_guest(_uid, suffix=f"try.{ind}")
            flag = self._conflict_flag_broadcast.get(suffix=f"try.{ind}")
            ind = ind + 1
            _uid = uuid.uuid1()
        return _uid

    def host_call(self):
        _uid = uuid.uuid1()
        ind = 0
        flag = True
        while flag:
            self._uuid_scatter.remote_host(_uid, suffix=f"try.{ind}")
            flag = self._conflict_flag_broadcast.get(suffix=f"try.{ind}")
            ind = ind + 1
            _uid = uuid.uuid1()
        return _uid

    def arbiter_call(self):
        ind = 0
        flag = True
        while flag:
            guest_uid = self._uuid_scatter.get_guest(suffix=f"try.{ind}")
            hosts_uid = self._uuid_scatter.get_hosts(suffix=f"try.{ind}")
            hosts_uid_set = set(hosts_uid)
            if len(hosts_uid_set) == len(hosts_uid) and guest_uid not in hosts_uid_set:
                flag = False
            self._conflict_flag_broadcast.remote(flag, suffix=f"try.{ind}")
            ind = ind + 1
        return True
