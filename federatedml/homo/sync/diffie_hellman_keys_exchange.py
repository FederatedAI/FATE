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

from federatedml.frameworks.homo.utils.secret import DiffieHellman

from federatedml.homo.transfer import arbiter_broadcast, arbiter_scatter, Transfer, Scatter


class _Arbiter(object):
    def __init__(self, dh_key_pair_broadcast: Transfer, dh_pubkey_scatter: Scatter, dh_pubkey_broadcast: Transfer):
        self._dh_key_pair_broadcast = dh_key_pair_broadcast
        self._dh_pubkey_scatter = dh_pubkey_scatter
        self._dh_pubkey_broadcast = dh_pubkey_broadcast

    def key_exchange(self):
        p, g = DiffieHellman.key_pair()
        self._dh_key_pair_broadcast.remote((int(p), int(g)))

        guest_uuid, guest_pubkey = self._dh_pubkey_scatter.get_guest()
        pubkey = {guest_uuid: guest_pubkey}
        for host_uuid, host_pubkey in self._dh_pubkey_scatter.get_hosts():
            pubkey[host_uuid] = host_pubkey

        self._dh_pubkey_broadcast.remote(pubkey)


class _Host(object):
    def __init__(self, dh_key_pair_broadcast: Transfer, dh_pubkey_scatter: Scatter, dh_pubkey_broadcast: Transfer):
        self._dh_key_pair_broadcast = dh_key_pair_broadcast
        self._dh_pubkey_scatter = dh_pubkey_scatter
        self._dh_pubkey_broadcast = dh_pubkey_broadcast

    def key_exchange(self, uuid):
        p, g = self._dh_key_pair_broadcast.get()
        r = DiffieHellman.generate_secret(p)
        gr = DiffieHellman.encrypt(g, r, p)
        self._dh_pubkey_scatter.remote_host((uuid, gr))

        pubkey = self._dh_pubkey_broadcast.get()
        privates = {_uuid: DiffieHellman.decrypt(_gr, r, p) for _uuid, _gr in pubkey.items() if _uuid != uuid}
        return privates


class _Guest(object):
    def __init__(self, dh_key_pair_broadcast: Transfer, dh_pubkey_scatter: Scatter, dh_pubkey_broadcast: Transfer):
        self._dh_key_pair_broadcast = dh_key_pair_broadcast
        self._dh_pubkey_scatter = dh_pubkey_scatter
        self._dh_pubkey_broadcast = dh_pubkey_broadcast

    def key_exchange(self, uuid):
        p, g = self._dh_key_pair_broadcast.get()
        r = DiffieHellman.generate_secret(p)
        gr = DiffieHellman.encrypt(g, r, p)
        self._dh_pubkey_scatter.remote_guest((uuid, gr))

        pubkey = self._dh_pubkey_broadcast.get()
        privates = {_uuid: DiffieHellman.decrypt(_gr, r, p) for _uuid, _gr in pubkey.items() if _uuid != uuid}
        return privates


def _parse_transfer_variable(transfer_variable):
    dh_pubkey_transfer_name = transfer_variable.dh_public_key.name,
    dh_pubkey_transfer_tag = transfer_variable.generate_transferid(transfer_variable.dh_public_key),
    dh_host_pubkey_transfer_name = transfer_variable.dh_host_public_key.name,
    dh_host_pubkey_transfer_tag = transfer_variable.generate_transferid(transfer_variable.dh_host_public_key),
    dh_guest_pubkey_transfer_name = transfer_variable.dh_guest_public_key.name,
    dh_guest_pubkey_transfer_tag = transfer_variable.generate_transferid(transfer_variable.dh_guest_public_key),
    dh_all_pubkey_name = transfer_variable.host_and_guest_key.name,
    dh_all_pubkey_tag = transfer_variable.generate_transferid(transfer_variable.host_and_guest_key)
    dh_key_pair_broadcast = arbiter_broadcast(name=dh_pubkey_transfer_name,
                                              tag=dh_pubkey_transfer_tag)
    dh_pubkey_scatter = arbiter_scatter(guest_name=dh_guest_pubkey_transfer_name,
                                        guest_tag=dh_guest_pubkey_transfer_tag,
                                        host_name=dh_host_pubkey_transfer_name,
                                        host_tag=dh_host_pubkey_transfer_tag)
    dh_pubkey_broadcast = arbiter_broadcast(name=dh_all_pubkey_name, tag=dh_all_pubkey_tag)
    return dh_key_pair_broadcast, dh_pubkey_scatter, dh_pubkey_broadcast


class DHKeysExchange(object):

    @staticmethod
    def arbiter(transfer_variable):
        return _Arbiter(*_parse_transfer_variable(transfer_variable))

    @staticmethod
    def host(transfer_variable):
        return _Host(*_parse_transfer_variable(transfer_variable))

    @staticmethod
    def guest(transfer_variable):
        return _Guest(*_parse_transfer_variable(transfer_variable))
