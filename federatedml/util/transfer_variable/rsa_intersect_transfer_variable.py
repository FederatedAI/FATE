#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable


class RsaIntersectTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.rsa_pubkey = Variable(name="RsaIntersectTransferVariable.rsa_pubkey", auth={'src': "host", 'dst': ['guest']})
        self.intersect_guest_ids = Variable(name="RsaIntersectTransferVariable.intersect_guest_ids", auth={'src': "guest", 'dst': ['host']})
        self.intersect_host_ids_process = Variable(name="RsaIntersectTransferVariable.intersect_host_ids_process", auth={'src': "host", 'dst': ['guest']})
        self.intersect_guest_ids_process = Variable(name="RsaIntersectTransferVariable.intersect_guest_ids_process", auth={'src': "host", 'dst': ['guest']})
        self.intersect_ids = Variable(name="RsaIntersectTransferVariable.intersect_ids", auth={'src': "guest", 'dst': ['host']})
        pass
