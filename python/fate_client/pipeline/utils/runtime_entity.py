from types import SimpleNamespace


class Roles(object):
    def __init__(self):
        self._role_party_mappings = dict()
        self._role_party_index_mapping = dict()
        self._leader_role = None

    def set_role(self, role, party_id):
        if not isinstance(party_id, list):
            party_id = [party_id]

        if role not in self._role_party_mappings:
            self._role_party_mappings[role] = []
            self._role_party_index_mapping[role] = dict()

        for pid in party_id:
            if pid in self._role_party_index_mapping[role]:
                raise ValueError(f"role {role}, party {pid} has been added before")
            self._role_party_index_mapping[role][pid] = len(self._role_party_mappings[role])
            self._role_party_mappings[role].append(pid)

        self._role_party_mappings[role] = party_id

    def set_leader(self, role, party_id):
        self._leader_role = SimpleNamespace(role=role,
                                            party_id=party_id)

    @property
    def leader(self):
        return self._leader_role

    def get_party_by_role(self, role):
        return self._role_party_mappings[role]

    def get_party_by_role_index(self, role, index):
        return self._role_party_mappings[role][index]
