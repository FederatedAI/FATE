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

from federatedml.homo.transfer import arbiter_scatter, arbiter_broadcast, Scatter, Transfer


def _tag_suffix(index):
    return f"try_{index}"


class _Arbiter(object):
    def __init__(self, uuid_scatter: Scatter, conflict_flag_broadcast: Transfer):
        self._uuid_scatter = uuid_scatter
        self._conflict_flag_broadcast = conflict_flag_broadcast

    def validate_uuid(self):
        ind = 0
        while True:
            guest_uid = self._uuid_scatter.get_guest(suffix=_tag_suffix(ind))
            hosts_uid = self._uuid_scatter.get_hosts(suffix=_tag_suffix(ind))
            hosts_uid_set = set(hosts_uid)
            if len(hosts_uid_set) == len(hosts_uid) and guest_uid not in hosts_uid_set:
                self._conflict_flag_broadcast.remote(True, suffix=_tag_suffix(ind))
                break
            self._conflict_flag_broadcast.remote(False, suffix=_tag_suffix(ind))
            ind = ind + 1


class _Client(object):
    def __init__(self, uuid_scatter: Scatter, conflict_flag_broadcast: Transfer):
        self._uuid_scatter = uuid_scatter
        self._conflict_flag_broadcast = conflict_flag_broadcast

    def generate_uuid(self):
        ind = -1
        while True:
            ind = ind + 1
            _uid = uuid.uuid1()
            self._uuid_scatter.remote_guest(_uid, suffix=_tag_suffix(ind))
            no_conflict = self._conflict_flag_broadcast.get(suffix=_tag_suffix(ind))
            if no_conflict:
                break
        return _uid


def _parse_transfer_variable(transfer_variable):

    host_name = transfer_variable.host_uuid.name
    host_tag = transfer_variable.generate_transferid(transfer_variable.host_uuid)
    guest_name = transfer_variable.guest_uuid.name
    guest_tag = transfer_variable.generate_transferid(transfer_variable.guest_uuid)
    uuid_scatter = arbiter_scatter(host_name=host_name,
                                   host_tag=host_tag,
                                   guest_name=guest_name,
                                   guest_tag=guest_tag)

    conflict_flag_name = transfer_variable.converge_flag.name
    conflict_flag_tag = transfer_variable.generate_transferid(transfer_variable.converge_flag)
    conflict_flag_broadcast = arbiter_broadcast(name=conflict_flag_name, tag=conflict_flag_tag)

    return uuid_scatter, conflict_flag_broadcast


class SynchronizedUUIDProcedure(object):

    @staticmethod
    def arbiter(transfer_variable) -> _Arbiter:
        return _Arbiter(*_parse_transfer_variable(transfer_variable))

    @staticmethod
    def guest(transfer_variable) -> _Client:
        return _Client(*_parse_transfer_variable(transfer_variable))

    @staticmethod
    def host(transfer_variable) -> _Client:
        return _Client(*_parse_transfer_variable(transfer_variable))
