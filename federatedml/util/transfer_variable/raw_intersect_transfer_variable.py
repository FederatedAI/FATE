#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from federatedml.util.transfer_variable.base_transfer_variable import BaseTransferVariable, Variable


class RawIntersectTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.send_ids_host = Variable(name="RawIntersectTransferVariable.send_ids_host", auth={'src': "host", 'dst': ['guest']})
        self.send_ids_guest = Variable(name="RawIntersectTransferVariable.send_ids_guest", auth={'src': "guest", 'dst': ['host']})
        self.intersect_ids_host = Variable(name="RawIntersectTransferVariable.intersect_ids_host", auth={'src': "host", 'dst': ['guest']})
        self.intersect_ids_guest = Variable(name="RawIntersectTransferVariable.intersect_ids_guest", auth={'src': "guest", 'dst': ['host']})
        pass
