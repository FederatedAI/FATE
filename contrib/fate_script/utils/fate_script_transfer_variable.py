#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2018, WeBank Inc. All Rights Reserved
#
################################################################################
# =============================================================================
# TransferVariable Class
# =============================================================================


class Variable(object):
    def __init__(self, name, auth):
        self.name = name
        self.auth = auth

class BaseTransferVariable(object):
    def __init__(self, flowid=0):
        self.flowid = flowid
        self.define_transfer_variable()

    def set_flowid(self, flowid):
        self.flowid = flowid

    def generate_transferid(self, transfer_var, *suffix):
        if transfer_var.name.split(".", -1)[-1] not in self.__dict__:
            raise ValueError("transfer variable not in class, please check if!!!")

        transferid = transfer_var.name + "." + str(self.flowid)
        if suffix:
            transferid += "." + ".".join(map(str, suffix))
        return transferid

    def define_transfer_variable(self):
        pass


class HeteroLRTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.paillier_pubkey = Variable(name="HeteroLRTransferVariable.paillier_pubkey", auth={'src': "A", 'dst': ['H', 'D', 'G', 'E']})
        self.ml_conf = Variable(name="HeteroLRTransferVariable.ml_conf", auth={'src': "A", 'dst': ['H', 'D', 'G', 'E', 'R']})
        self._enc_forward = Variable(name="HeteroLRTransferVariable._enc_forward", auth={'src': "H", 'dst': ['G']})
        self._enc_forward_square = Variable(name="HeteroLRTransferVariable._enc_forward_square", auth={'src': "H", 'dst': ['G']})
        self._enc_fore_gradient = Variable(name="HeteroLRTransferVariable._enc_fore_gradient", auth={'src': "G", 'dst': ['H']})
        self._enc_grad_G = Variable(name="HeteroLRTransferVariable._enc_grad_G", auth={'src': "G", 'dst': ['A']})
        self._enc_grad_H = Variable(name="HeteroLRTransferVariable._enc_grad_H", auth={'src': "H", 'dst': ['A']})
        self.shape_w = Variable(name="HeteroLRTransferVariable.shape_w", auth={'src': "G", 'dst': ['A']})
        self.optim_grad_g = Variable(name="HeteroLRTransferVariable.optim_grad_g", auth={'src': "A", 'dst': ['G']})
        self.optim_grad_h = Variable(name="HeteroLRTransferVariable.optim_grad_h", auth={'src': "A", 'dst': ['H']})
        self._enc_loss = Variable(name="HeteroLRTransferVariable._enc_loss", auth={'src': "G", 'dst': ['A']})
        self.is_stopped = Variable(name="HeteroLRTransferVariable.is_stopped", auth={'src': "A", 'dst': ['G', 'H']})
        self.Z = Variable(name="HeteroLRTransferVariable.Z", auth={'src': "H", 'dst': ['G']})
        pass
