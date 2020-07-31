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


import typing

from fate_arch.common import Party


class _FederationParties(object):
    def __init__(self, party, parties):
        self._party = party
        self._parties = parties

    @property
    def local_party(self) -> Party:
        return self._party

    @property
    def all_parties(self):
        return [party for parties in self._parties.values for party in parties]

    def roles_to_parties(self, roles: typing.Iterable) -> list:
        return [party for role in roles for party in self._parties[role]]

    def role_to_party(self, role, idx) -> Party:
        return self._parties[role][idx]


__all__ = ["_FederationParties"]
