from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables

class NormDataTransferVariable(BaseTransferVariables):
    def __init__(self,flowid=0):
        super(NormDataTransferVariable, self).__init__(flowid)
        self.guest_to_host = self._create_variable(name="guest_to_host",src=["guest"],dst=["host"])
        self.host_to_guest = self._create_variable(name="host_to_guest",src=["host"],dst=["guest"])
        
