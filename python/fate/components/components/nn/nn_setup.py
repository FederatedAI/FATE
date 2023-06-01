from fate.components import Role


class NNSetup(object):

    def __init__(self) -> None:
        
        self._role = None
        self._party_id = None
        self._cpn_input_data = None
        self._cpn_input_model = None

    def set_role(self, role: Role):
        self._role = role

    def is_client(self):
        return self._role.is_guest or self._role.is_host
    
    def is_server(self):
        return self._role.is_arbiter
    
    def set_party_id(self, party_id):
        assert isinstance(self._party_id, int)
        self._party_id = party_id

    def set_cpn_input_data(self, cpn_input):
        self._cpn_input_data = cpn_input

    def set_cpn_input_model(self, cpn_input):
        self._cpn_input_model = cpn_input

    def get_cpn_input_data(self):
        return self._cpn_input_data

    def get_cpn_input_model(self):
        return self._cpn_input_model

    def setup(self):
        raise NotImplementedError("method setup must be implemented")
