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
        self.guest_gradient = Variable(name="HeteroLRTransferVariable.guest_gradient", auth={'src': "guest", 'dst': [u'arbiter']})
        self.host_optim_gradient = Variable(name="HeteroLRTransferVariable.host_optim_gradient", auth={'src': "arbiter", 'dst': [u'host']})
        self.host_forward_dict = Variable(name="HeteroLRTransferVariable.host_forward_dict", auth={'src': "host", 'dst': [u'guest']})
        self.loss = Variable(name="HeteroLRTransferVariable.loss", auth={'src': "guest", 'dst': [u'arbiter']})
        self.fore_gradient = Variable(name="HeteroLRTransferVariable.fore_gradient", auth={'src': "guest", 'dst': [u'host']})
        self.is_stopped = Variable(name="HeteroLRTransferVariable.is_stopped", auth={'src': "arbiter", 'dst': [u'host', u'guest']})
        self.paillier_pubkey = Variable(name="HeteroLRTransferVariable.paillier_pubkey", auth={'src': "arbiter", 'dst': [u'host', u'guest']})
        self.batch_data_index = Variable(name="HeteroLRTransferVariable.batch_data_index", auth={'src': "guest", 'dst': [u'host']})
        self.guest_optim_gradient = Variable(name="HeteroLRTransferVariable.guest_optim_gradient", auth={'src': "arbiter", 'dst': [u'guest']})
        self.host_gradient = Variable(name="HeteroLRTransferVariable.host_gradient", auth={'src': "host", 'dst': [u'arbiter']})
        self.host_loss_regular = Variable(name="HeteroLRTransferVariable.host_loss_regular", auth={'src': "host", 'dst': [u'guest']})
        self.batch_info = Variable(name="HeteroLRTransferVariable.batch_info", auth={'src': "guest", 'dst': [u'host', u'arbiter']})
        self.host_prob = Variable(name="HeteroLRTransferVariable.host_prob", auth={'src': "host", 'dst': [u'guest']})
        pass


class RsaIntersectTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.intersect_guest_ids_process = Variable(name="RsaIntersectTransferVariable.intersect_guest_ids_process", auth={'src': "host", 'dst': [u'guest']})
        self.intersect_ids = Variable(name="RsaIntersectTransferVariable.intersect_ids", auth={'src': "guest", 'dst': [u'host']})
        self.intersect_guest_ids = Variable(name="RsaIntersectTransferVariable.intersect_guest_ids", auth={'src': "guest", 'dst': [u'host']})
        self.rsa_pubkey = Variable(name="RsaIntersectTransferVariable.rsa_pubkey", auth={'src': "host", 'dst': [u'guest']})
        self.intersect_host_ids_process = Variable(name="RsaIntersectTransferVariable.intersect_host_ids_process", auth={'src': "host", 'dst': [u'guest']})
        pass


class RawIntersectTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.intersect_ids_guest = Variable(name="RawIntersectTransferVariable.intersect_ids_guest", auth={'src': "guest", 'dst': [u'host']})
        self.send_ids_host = Variable(name="RawIntersectTransferVariable.send_ids_host", auth={'src': "host", 'dst': [u'guest']})
        self.send_ids_guest = Variable(name="RawIntersectTransferVariable.send_ids_guest", auth={'src': "guest", 'dst': [u'host']})
        self.intersect_ids_host = Variable(name="RawIntersectTransferVariable.intersect_ids_host", auth={'src': "host", 'dst': [u'guest']})
        pass


class HeteroDecisionTreeTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.node_positions = Variable(name="HeteroDecisionTreeTransferVariable.node_positions", auth={'src': "guest", 'dst': [u'host']})
        self.dispatch_node_host = Variable(name="HeteroDecisionTreeTransferVariable.dispatch_node_host", auth={'src': "guest", 'dst': [u'host']})
        self.tree = Variable(name="HeteroDecisionTreeTransferVariable.tree", auth={'src': "guest", 'dst': [u'host']})
        self.encrypted_splitinfo_host = Variable(name="HeteroDecisionTreeTransferVariable.encrypted_splitinfo_host", auth={'src': "host", 'dst': [u'guest']})
        self.predict_data_by_host = Variable(name="HeteroDecisionTreeTransferVariable.predict_data_by_host", auth={'src': "host", 'dst': [u'guest']})
        self.tree_node_queue = Variable(name="HeteroDecisionTreeTransferVariable.tree_node_queue", auth={'src': "guest", 'dst': [u'host']})
        self.federated_best_splitinfo_host = Variable(name="HeteroDecisionTreeTransferVariable.federated_best_splitinfo_host", auth={'src': "guest", 'dst': [u'host']})
        self.predict_finish_tag = Variable(name="HeteroDecisionTreeTransferVariable.predict_finish_tag", auth={'src': "guest", 'dst': [u'host']})
        self.final_splitinfo_host = Variable(name="HeteroDecisionTreeTransferVariable.final_splitinfo_host", auth={'src': "host", 'dst': [u'guest']})
        self.dispatch_node_host_result = Variable(name="HeteroDecisionTreeTransferVariable.dispatch_node_host_result", auth={'src': "host", 'dst': [u'guest']})
        self.encrypted_grad_and_hess = Variable(name="HeteroDecisionTreeTransferVariable.encrypted_grad_and_hess", auth={'src': "guest", 'dst': [u'host']})
        self.predict_data = Variable(name="HeteroDecisionTreeTransferVariable.predict_data", auth={'src': "guest", 'dst': [u'host']})
        pass


class SecureAddExampleTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.host_share = Variable(name="SecureAddExampleTransferVariable.host_share", auth={'src': "host", 'dst': [u'guest']})
        self.guest_share = Variable(name="SecureAddExampleTransferVariable.guest_share", auth={'src': "guest", 'dst': [u'host']})
        self.host_sum = Variable(name="SecureAddExampleTransferVariable.host_sum", auth={'src': "host", 'dst': [u'guest']})
        pass


class HeteroSecureBoostingTreeTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.tree_dim = Variable(name="HeteroSecureBoostingTreeTransferVariable.tree_dim", auth={'src': "guest", 'dst': [u'host']})
        self.stop_flag = Variable(name="HeteroSecureBoostingTreeTransferVariable.stop_flag", auth={'src': "guest", 'dst': [u'host']})
        pass


class HeteroFeatureSelectionTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.result_left_cols = Variable(name="HeteroFeatureSelectionTransferVariable.result_left_cols", auth={'src': "guest", 'dst': [u'host']})
        self.host_iv_threshold = Variable(name="HeteroFeatureSelectionTransferVariable.host_iv_threshold", auth={'src': "host", 'dst': [u'guest']})
        pass


class HeteroFeatureBinningTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.encrypted_label = Variable(name="HeteroFeatureBinningTransferVariable.encrypted_label", auth={'src': "guest", 'dst': [u'host']})
        self.encrypted_bin_sum = Variable(name="HeteroFeatureBinningTransferVariable.encrypted_bin_sum", auth={'src': "host", 'dst': [u'guest']})
        self.paillier_pubkey = Variable(name="HeteroFeatureBinningTransferVariable.paillier_pubkey", auth={'src': "guest", 'dst': [u'host']})
        pass


class HomoLRTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.host_loss = Variable(name="HomoLRTransferVariable.host_loss", auth={'src': "host", 'dst': [u'arbiter']})
        self.guest_party_weight = Variable(name="HomoLRTransferVariable.guest_party_weight", auth={'src': "guest", 'dst': [u'arbiter']})
        self.use_encrypt = Variable(name="HomoLRTransferVariable.use_encrypt", auth={'src': "host", 'dst': [u'arbiter']})
        self.host_model = Variable(name="HomoLRTransferVariable.host_model", auth={'src': "host", 'dst': [u'arbiter']})
        self.guest_loss = Variable(name="HomoLRTransferVariable.guest_loss", auth={'src': "guest", 'dst': [u'arbiter']})
        self.guest_model = Variable(name="HomoLRTransferVariable.guest_model", auth={'src': "guest", 'dst': [u'arbiter']})
        self.re_encrypt_times = Variable(name="HomoLRTransferVariable.re_encrypt_times", auth={'src': "host", 'dst': [u'arbiter']})
        self.converge_flag = Variable(name="HomoLRTransferVariable.converge_flag", auth={'src': "arbiter", 'dst': [u'guest', u'host']})
        self.predict_wx = Variable(name="HomoLRTransferVariable.predict_wx", auth={'src': "host", 'dst': [u'arbiter']})
        self.to_encrypt_model = Variable(name="HomoLRTransferVariable.to_encrypt_model", auth={'src': "host", 'dst': [u'arbiter']})
        self.paillier_pubkey = Variable(name="HomoLRTransferVariable.paillier_pubkey", auth={'src': "arbiter", 'dst': [u'host']})
        self.final_model = Variable(name="HomoLRTransferVariable.final_model", auth={'src': "arbiter", 'dst': [u'guest', u'host']})
        self.re_encrypted_model = Variable(name="HomoLRTransferVariable.re_encrypted_model", auth={'src': "arbiter", 'dst': [u'host']})
        self.predict_result = Variable(name="HomoLRTransferVariable.predict_result", auth={'src': "arbiter", 'dst': [u'host']})
        self.host_party_weight = Variable(name="HomoLRTransferVariable.host_party_weight", auth={'src': "host", 'dst': [u'arbiter']})
        pass


class HeteroWorkFlowTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.test_data = Variable(name="HeteroWorkFlowTransferVariable.test_data", auth={'src': "guest", 'dst': [u'host']})
        self.train_data = Variable(name="HeteroWorkFlowTransferVariable.train_data", auth={'src': "guest", 'dst': [u'host']})
        pass


class HeteroFTLTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.is_encrypted_ftl_stopped = Variable(name="HeteroFTLTransferVariable.is_encrypted_ftl_stopped", auth={'src': "arbiter", 'dst': [u'host', u'guest']})
        self.decrypt_guest_gradient = Variable(name="HeteroFTLTransferVariable.decrypt_guest_gradient", auth={'src': "arbiter", 'dst': [u'guest']})
        self.batch_data_index = Variable(name="HeteroFTLTransferVariable.batch_data_index", auth={'src': "guest", 'dst': [u'host']})
        self.host_sample_indexes = Variable(name="HeteroFTLTransferVariable.host_sample_indexes", auth={'src': "host", 'dst': [u'guest']})
        self.encrypt_prob = Variable(name="HeteroFTLTransferVariable.encrypt_prob", auth={'src': "guest", 'dst': [u'arbiter']})
        self.batch_num = Variable(name="HeteroFTLTransferVariable.batch_num", auth={'src': "guest", 'dst': [u'arbiter', u'host']})
        self.encrypt_guest_gradient = Variable(name="HeteroFTLTransferVariable.encrypt_guest_gradient", auth={'src': "guest", 'dst': [u'arbiter']})
        self.encrypt_loss = Variable(name="HeteroFTLTransferVariable.encrypt_loss", auth={'src': "guest", 'dst': [u'arbiter']})
        self.encrypt_host_gradient = Variable(name="HeteroFTLTransferVariable.encrypt_host_gradient", auth={'src': "host", 'dst': [u'arbiter']})
        self.guest_component_list = Variable(name="HeteroFTLTransferVariable.guest_component_list", auth={'src': "guest", 'dst': [u'host']})
        self.is_stopped = Variable(name="HeteroFTLTransferVariable.is_stopped", auth={'src': "guest", 'dst': [u'host']})
        self.pred_prob = Variable(name="HeteroFTLTransferVariable.pred_prob", auth={'src': "guest", 'dst': [u'host']})
        self.decrypt_host_gradient = Variable(name="HeteroFTLTransferVariable.decrypt_host_gradient", auth={'src': "arbiter", 'dst': [u'host']})
        self.paillier_pubkey = Variable(name="HeteroFTLTransferVariable.paillier_pubkey", auth={'src': "arbiter", 'dst': [u'host', u'guest']})
        self.batch_size = Variable(name="HeteroFTLTransferVariable.batch_size", auth={'src': "guest", 'dst': [u'host']})
        self.host_component_list = Variable(name="HeteroFTLTransferVariable.host_component_list", auth={'src': "host", 'dst': [u'guest']})
        self.decrypt_prob = Variable(name="HeteroFTLTransferVariable.decrypt_prob", auth={'src': "arbiter", 'dst': [u'guest']})
        self.guest_sample_indexes = Variable(name="HeteroFTLTransferVariable.guest_sample_indexes", auth={'src': "guest", 'dst': [u'host']})
        self.host_prob = Variable(name="HeteroFTLTransferVariable.host_prob", auth={'src': "host", 'dst': [u'guest']})
        pass
