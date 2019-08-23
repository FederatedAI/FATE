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

from arch.api import federation
from federatedml.util import consts


class Transfer(object):
    def __init__(self, name, tag, default_receive_idx, send_role, default_send_idx):
        self._name = name
        self._tag = tag
        self._default_receive_idx = default_receive_idx
        self._send_role = send_role
        self._default_send_idx = default_send_idx

    def get(self, idx=None, suffix=None):
        tag = f"{self._tag}.{suffix}" if suffix else self._tag
        idx = idx if idx else self._default_receive_idx
        return federation.get(name=self._name, tag=tag, idx=idx)

    def remote(self, value, idx=None, suffix=None):
        tag = f"{self._tag}.{suffix}" if suffix else self._tag
        idx = idx if idx else self._default_send_idx
        return federation.remote(value, name=self._name, tag=tag, role=self._send_role, idx=idx)


class Scatter(object):
    def __init__(self, guest_name, guest_tag, host_name, host_tag):
        self._guest_name = guest_name
        self._guest_tag = guest_tag
        self._host_name = host_name
        self._host_tag = host_tag

    def get_guest(self, suffix=None):
        tag = f"{self._guest_tag}.{suffix}" if suffix else self._guest_tag
        return federation.get(name=self._guest_name, tag=tag, idx=0)

    def get_hosts(self, suffix=None):
        tag = f"{self._host_tag}.{suffix}" if suffix else self._host_tag
        return federation.get(name=self._host_name, tag=tag, idx=-1)

    def remote_guest(self, value, suffix=None):
        tag = f"{self._guest_tag}.{suffix}" if suffix else self._guest_tag
        return federation.remote(obj=value, name=self._guest_name, tag=tag, role=consts.ARBITER, idx=0)

    def remote_host(self, value, suffix=None):
        tag = f"{self._host_tag}.{suffix}" if suffix else self._host_tag
        return federation.remote(obj=value, name=self._host_name, tag=tag, role=consts.ARBITER, idx=0)


"""1 to 1"""


def arbiter2guest(name, tag):
    return Transfer(name=name, tag=tag, default_receive_idx=0, send_role=consts.GUEST, default_send_idx=0)


def arbiter2host(name, tag, host_idx=-1):
    return Transfer(name=name, tag=tag, default_receive_idx=0, send_role=consts.HOST, default_send_idx=host_idx)


def guest2arbiter(name, tag):
    return Transfer(name=name, tag=tag, default_receive_idx=0, send_role=consts.ARBITER, default_send_idx=0)


def host2arbiter(name, tag, host_idx=-1):
    return Transfer(name=name, tag=tag, default_receive_idx=host_idx, send_role=consts.ARBITER, default_send_idx=0)


def guest2host(name, tag, host_idx=-1):
    return Transfer(name=name, tag=tag, default_receive_idx=0, send_role=consts.HOST, default_send_idx=host_idx)


def host2guest(name, tag, host_idx=-1):
    return Transfer(name=name, tag=tag, default_receive_idx=host_idx, send_role=consts.GUEST, default_send_idx=0)


"""1 to 2"""


def arbiter_broadcast(name, tag):
    return Transfer(name=name, tag=tag, default_receive_idx=0, send_role=None, default_send_idx=-1)


def arbiter_scatter(host_name, host_tag, guest_name, guest_tag):
    return Scatter(guest_name=guest_name, guest_tag=guest_tag, host_name=host_name, host_tag=host_tag)
