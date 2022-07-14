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


class Role:
    def __init__(self, parties) -> None:
        self._parties = parties
        self._size = len(self._parties)

    def __getitem__(self, key):
        return self._parties[key]


class _PartiesMeta(type):
    @property
    def Guest(cls) -> Role:
        return cls._get_instance()._guest

    @property
    def Host(cls) -> Role:
        return cls._get_instance()._host

    @property
    def Arbiter(cls) -> Role:
        return cls._get_instance()._arbiter


class PartiesInfo(metaclass=_PartiesMeta):
    _instance = None

    @classmethod
    def _set_instance(cls, inst):
        cls._instance = inst

    @classmethod
    def _get_instance(cls) -> "PartiesInfo":
        if cls._instance is None:
            raise RuntimeError(f"parties not initialized")
        return cls._instance

    @classmethod
    def get_parties(cls, parties) -> typing.List[Party]:
        if isinstance(parties, Party):
            return [parties]

        elif isinstance(parties, Role):
            return parties[:]

        elif isinstance(parties, list):
            plain_parties = []
            for p in parties:
                plain_parties.extend(cls.get_parties(p))
            if len(set(plain_parties)) != len(plain_parties):
                raise ValueError(f"duplicated parties exsits: {plain_parties}")
            return plain_parties
        raise ValueError(f"unsupported type: {type(parties)}")

    @staticmethod
    def from_conf(conf: typing.MutableMapping[str, dict]):
        try:
            local = Party(
                role=conf["local"]["role"], party_id=conf["local"]["party_id"]
            )
            role_to_parties = {}
            for role, party_id_list in conf.get("role", {}).items():
                role_to_parties[role] = [
                    Party(role=role, party_id=party_id) for party_id in party_id_list
                ]
        except Exception as e:
            raise RuntimeError(
                "conf parse error, a correct configuration could be:\n"
                "{\n"
                "  'local': {'role': 'guest', 'party_id': 10000},\n"
                "  'role': {'guest': [10000], 'host': [9999, 9998]}, 'arbiter': [9997]}\n"
                "}"
            ) from e
        return PartiesInfo(local, role_to_parties)

    def __init__(
            self,
            local: Party,
            role_to_parties: typing.MutableMapping[str, typing.List[Party]],
    ):
        self._local = local
        self._role_to_parties = role_to_parties

        self._guest = Role(role_to_parties.get("guest", []))
        self._host = Role(role_to_parties.get("host", []))
        self._arbiter = Role(role_to_parties.get("arbiter", []))

        self._set_instance(self)

    @property
    def local_party(self) -> Party:
        return self._local

    @property
    def all_parties(self):
        return [
            party for parties in self._role_to_parties.values() for party in parties
        ]

    @property
    def role_set(self):
        return set(self._role_to_parties)

    def roles_to_parties(self, roles: typing.Iterable, strict=True) -> list:
        parties = []
        for role in roles:
            if role not in self._role_to_parties:
                if strict:
                    raise RuntimeError(
                        f"try to get role {role} "
                        f"which is not configured in `role` in runtime conf({self._role_to_parties})"
                    )
                else:
                    continue
            parties.extend(self._role_to_parties[role])

        return parties

    def role_to_party(self, role, idx) -> Party:
        return self._role_to_parties[role][idx]


__all__ = ["PartiesInfo", "Role"]
