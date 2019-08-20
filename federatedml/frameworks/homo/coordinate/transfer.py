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
    def __init__(self, name, tag, receive_idx, send_role, send_idx):
        self._name = name
        self._tag = tag
        self._receive_idx = receive_idx
        self._send_role = send_role
        self._send_idx = send_idx

    def get(self, suffix=None):
        tag = f"{self._tag}.{suffix}" if suffix else self._tag
        return federation.get(name=self._name, tag=tag, idx=self._receive_idx)

    def remote(self, value, suffix=None):
        tag = f"{self._tag}.{suffix}" if suffix else self._tag
        return federation.remote(value, name=self._name, tag=tag, role=self._send_role, idx=self._send_idx)


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
    return Transfer(name=name, tag=tag, receive_idx=0, send_role=consts.GUEST, send_idx=0)


def arbiter2host(name, tag, host_idx=-1):
    return Transfer(name=name, tag=tag, receive_idx=0, send_role=consts.HOST, send_idx=host_idx)


def guest2arbiter(name, tag):
    return Transfer(name=name, tag=tag, receive_idx=0, send_role=consts.ARBITER, send_idx=0)


def host2arbiter(name, tag, host_idx=-1):
    return Transfer(name=name, tag=tag, receive_idx=host_idx, send_role=consts.ARBITER, send_idx=0)


def guest2host(name, tag, host_idx=-1):
    return Transfer(name=name, tag=tag, receive_idx=0, send_role=consts.HOST, send_idx=host_idx)


def host2guest(name, tag, host_idx=-1):
    return Transfer(name=name, tag=tag, receive_idx=host_idx, send_role=consts.GUEST, send_idx=0)


"""1 to 2"""


def arbiter_broadcast(name, tag):
    return Transfer(name=name, tag=tag, receive_idx=0, send_role=None, send_idx=-1)


def arbiter_scatter(host_name, host_tag, guest_name, guest_tag):
    return Scatter(guest_name=guest_name, guest_tag=guest_tag, host_name=host_name, host_tag=host_tag)
    #
    # @staticmethod
    # def _hosts_to_arbiter(name, tag):
    #     return Transfer(name=name, tag=tag, receive_idx=-1, send_role=consts.ARBITER, send_idx=0)
    #
    # @staticmethod
    # def _guest_to_arbiter(name, tag):
    #     return Transfer(name=name, tag=tag, receive_idx=0, send_role=consts.ARBITER, send_idx=0)
    #
    # @staticmethod
    # def _arbiter_broadcast_hosts(name, tag, idx=-1):
    #     return Transfer(name=name, tag=tag, receive_idx=0, send_role=consts.HOST, send_idx=idx)
    #
    # @staticmethod
    # def _arbiter_broadcast_guest(name, tag):
    #     return Transfer(name=name, tag=tag, receive_idx=0, send_role=consts.GUEST, send_idx=-1)
    #
    # @staticmethod
    # def _arbiter_broadcast(name, tag):
    #     return Transfer(name=name, tag=tag, receive_idx=0, send_role=None, send_idx=-1)
    #
    # @staticmethod
    # def _client_to_server(host_name, host_tag, guest_name, guest_tag):
    #     def _fn(role):
    #         if role == consts.HOST:
    #             return Transfer._hosts_to_arbiter(host_name, host_tag)
    #         if role == consts.GUEST:
    #             return Transfer._guest_to_arbiter(guest_name, guest_tag)
    #         else:
    #             raise ValueError("role should be {0} or {1}".format(consts.HOST, consts.GUEST))
    #
    #     return _fn
    #
    # @staticmethod
    # def _generate_transfer(transfer_variable, func, variable_name, tag_suffix=None, **kwargs):
    #     variable = getattr(transfer_variable, variable_name)
    #     name = variable.name
    #     tag = transfer_variable.generate_transferid(variable)
    #     if isinstance(tag_suffix, types.FunctionType):
    #         return lambda x: func(name=name, tag=f"{tag}.{tag_suffix(x)}", **kwargs)
    #     elif isinstance(tag_suffix, str):
    #         return func(name=name, tag=f"{tag}.{tag_suffix}")
    #     else:
    #         return func(name=name, tag=tag, **kwargs)
    #
    # @staticmethod
    # def scatter_hosts(transfer_variable_name, transfer_tag):
    #     return Transfer._hosts_to_arbiter(transfer_variable_name, transfer_tag)
    #
    # @staticmethod
    # def scatter_guest(transfer_variable, variable_name, tag_suffix=None):
    #     return Transfer._generate_transfer(
    #         transfer_variable=transfer_variable,
    #         func=Transfer._guest_to_arbiter,
    #         variable_name=variable_name,
    #         tag_suffix=tag_suffix)
    #
    # @staticmethod
    # def scatter(transfer_variable, host_variable_name, guest_variable_name, tag_suffix=None):
    #     def _fn(role):
    #         if role == consts.HOST:
    #             return Transfer.scatter_hosts(
    #                 transfer_variable=transfer_variable,
    #                 variable_name=host_variable_name,
    #                 tag_suffix=tag_suffix)
    #         elif role == consts.GUEST:
    #             return Transfer.scatter_guest(
    #                 transfer_variable=transfer_variable,
    #                 variable_name=guest_variable_name,
    #                 tag_suffix=tag_suffix)
    #         else:
    #             raise ValueError("role should be {0} or {1}".format(consts.HOST, consts.GUEST))
    #
    #     return _fn
    #
    # @staticmethod
    # def broadcast(transfer_name, transfer_tag):
    #     return Transfer._arbiter_broadcast(transfer_name, transfer_tag)
    #
    # @staticmethod
    # def broadcast_hosts(transfer_variable_name, transfer_tag, tag_suffix=None, idx=-1):
    #     return Transfer._arbiter_broadcast_hosts()
    #
    # @staticmethod
    # def broadcast_guest(transfer_variable, variable_name, tag_suffix=None):
    #     return Transfer._generate_transfer(
    #         transfer_variable=transfer_variable,
    #         func=Transfer._arbiter_broadcast_guest,
    #         variable_name=variable_name,
    #         tag_suffix=tag_suffix)
