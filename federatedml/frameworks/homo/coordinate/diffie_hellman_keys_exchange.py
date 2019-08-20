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

from federatedml.frameworks.homo.coordinate.base import Coordinate
from federatedml.frameworks.homo.coordinate.transfer import arbiter_broadcast, arbiter_scatter
from federatedml.frameworks.homo.utils.secret import DiffieHellman
from federatedml.util import consts
from federatedml.util.transfer_variable.homo_transfer_variable import HomeModelTransferVariable


class DHKeysExchange(Coordinate):

    @staticmethod
    def from_transfer_variable(transfer_variable: HomeModelTransferVariable):
        return DHKeysExchange(
            dh_pubkey_transfer_name=transfer_variable.dh_public_key.name,
            dh_pubkey_transfer_tag=transfer_variable.generate_transferid(transfer_variable.dh_public_key),
            dh_host_pubkey_transfer_name=transfer_variable.dh_host_public_key.name,
            dh_host_pubkey_transfer_tag=transfer_variable.generate_transferid(transfer_variable.dh_host_public_key),
            dh_guest_pubkey_transfer_name=transfer_variable.dh_guest_public_key.name,
            dh_guest_pubkey_transfer_tag=transfer_variable.generate_transferid(transfer_variable.dh_guest_public_key),
            dh_all_pubkey_name=transfer_variable.host_and_guest_key.name,
            dh_all_pubkey_tag=transfer_variable.generate_transferid(transfer_variable.host_and_guest_key)
        )

    def __init__(self,
                 dh_pubkey_transfer_name,
                 dh_pubkey_transfer_tag,
                 dh_host_pubkey_transfer_name,
                 dh_host_pubkey_transfer_tag,
                 dh_guest_pubkey_transfer_name,
                 dh_guest_pubkey_transfer_tag,
                 dh_all_pubkey_name,
                 dh_all_pubkey_tag):
        self._dh_key_pair_broadcast = arbiter_broadcast(name=dh_pubkey_transfer_name,
                                                        tag=dh_pubkey_transfer_tag)
        self._dh_pubkey_scatter = arbiter_scatter(guest_name=dh_guest_pubkey_transfer_name,
                                                  guest_tag=dh_guest_pubkey_transfer_tag,
                                                  host_name=dh_host_pubkey_transfer_name,
                                                  host_tag=dh_host_pubkey_transfer_tag)
        self._dh_pubkey_broadcast = arbiter_broadcast(name=dh_all_pubkey_name, tag=dh_all_pubkey_tag)

    def _client_call(self, role, uuid):
        p, g = self._dh_key_pair_broadcast.get()
        r = DiffieHellman.generate_secret(p)
        gr = DiffieHellman.encrypt(g, r, p)
        self._dh_pubkey_scatter(role).remote((uuid, gr))

        pubkey = self._dh_pubkey_broadcast.get()
        return {_uuid: DiffieHellman.decrypt(_gr, r, p) for _uuid, _gr in pubkey.items() if _uuid != uuid}

    def guest_call(self, uuid):
        return self._client_call(consts.GUEST, uuid)

    def host_call(self, uuid):
        return self._client_call(consts.HOST, uuid)

    def arbiter_call(self):
        p, g = DiffieHellman.key_pair()
        self._dh_key_pair_broadcast.remote((int(p), int(g)))

        guest_uuid, guest_pubkey = self._dh_pubkey_scatter(consts.GUEST).get()
        pubkey = {guest_uuid: guest_pubkey}
        for host_uuid, host_pubkey in self._dh_pubkey_scatter(consts.HOST).get():
            pubkey[host_uuid] = host_pubkey

        self._dh_pubkey_broadcast.remote(pubkey)
