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

from federatedml.homo.transfer import host2arbiter, guest2arbiter


class _Arbiter(object):
    def __init__(self, transfer_variable, host_name, guest_name):
        host_variable = getattr(transfer_variable, host_name)
        self._host2arbiter = host2arbiter(name=host_variable.name,
                                          tag=transfer_variable.generate_transferid(host_variable))
        guest_variable = getattr(transfer_variable, guest_name)
        self._guest2arbiter = guest2arbiter(name=guest_variable.name,
                                            tag=transfer_variable.generate_transferid(guest_variable))

    def get_guest(self, suffix=None):
        return self._guest2arbiter.get(idx=0, suffix=suffix)

    def get_hosts(self, suffix=None):
        return self._host2arbiter.get(suffix=suffix)


class _Guest(object):
    def __init__(self, transfer_variable, guest_name):
        guest_variable = getattr(transfer_variable, guest_name)
        self._guest2arbiter = guest2arbiter(name=guest_variable.name,
                                            tag=transfer_variable.generate_transferid(guest_variable))

    def send(self, value, suffix):
        return self._guest2arbiter.remote(value=value, suffix=suffix)


class _Host(object):
    def __init__(self, transfer_variable, host_name):
        host_variable = getattr(transfer_variable, host_name)
        self._host2arbiter = host2arbiter(name=host_variable.name,
                                          tag=transfer_variable.generate_transferid(host_variable))

    def send(self, value, suffix):
        return self._host2arbiter.remote(value=value, suffix=suffix)


class ScatterParameters(object):
    """@hosts, @guest -> @arbiter
    transfer models from hosts and guest to arbiter for model aggregation
    """
    _host_name = "host_parameters"
    _guest_name = "guest_parameters"

    @classmethod
    def arbiter(cls, transfer_variable):
        return _Arbiter(transfer_variable, cls._host_name, cls._guest_name)

    @classmethod
    def guest(cls, transfer_variable):
        return _Guest(transfer_variable, cls._guest_name)

    @classmethod
    def host(cls, transfer_variable):
        return _Host(transfer_variable, cls._host_name)
