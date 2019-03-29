import numpy as np

from arch.api import eggroll
from arch.api.utils import log_utils
from federatedml.ftl.data_util.common_data_util import add_random_mask, remove_random_mask
from federatedml.logistic_regression.hetero_dnn_logistic_regression.federation_client import FATEFederationClient
from federatedml.util import consts
from federatedml.util.transfer_variable import HeteroDNNLRTransferVariable

LOGGER = log_utils.getLogger()


class LocalModelProxy(object):

    def __init__(self, model):
        self.model = model
        self.transfer_variable = HeteroDNNLRTransferVariable()
        self.federation_client = FATEFederationClient()

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_federation_client(self, federation_client):
        self.federation_client = federation_client

    def transform(self, instance_table):
        """
        transform instances into features.

        Parameters
        ----------
        :param instance_table: dtable with a collection of (index, instance) pairs
        :return:
        """

        LOGGER.debug("@ extract representative features from raw input")

        index_tracking_list = []

        indexed_instances = instance_table.collect()
        features_list = []
        instances_list = []
        for idx, inst in indexed_instances:
            index_tracking_list.append(idx)
            features_list.append(inst.features)
            instances_list.append(inst)
        raw_features = np.array(features_list)
        trans_features = self.model.transform(raw_features)

        indexed_instances = []
        for idx, inst, feat in zip(index_tracking_list, instances_list, trans_features):
            inst.set_feature(feat)
            indexed_instances.append((idx, inst))

        dtable = eggroll.parallelize(indexed_instances, include_key=True, partition=instance_table._partitions)
        return dtable, index_tracking_list

    def update_local_model(self, fore_gradient_table, instance_table, coef, **training_info):
        """
        Update local model

        Parameters
        ----------
        :param fore_gradient_table: dtable with a collection of (index, gradient) pairs
        :param instance_table: dtable with a collection of (index, instance) pairs
        :param coef: coefficients or parameters of logistic regression model
        """

        LOGGER.debug("@ update local model")

        n_iter = training_info["iteration"]
        batch_index = training_info["batch_index"]
        index_tracking_list = training_info["index_tracking_list"]
        is_host = training_info["is_host"]

        indexed_fore_gradients = fore_gradient_table.collect()
        indexed_instances = instance_table.collect()
        fore_gradients_dict = dict(indexed_fore_gradients)
        instances_dict = dict(indexed_instances)

        grad_list = []
        feat_list = []
        for idx in index_tracking_list:
            grad = fore_gradients_dict[idx]
            inst = instances_dict[idx]
            grad_list.append(grad)
            feat_list.append(inst.features)

        grads = np.array(grad_list)
        feats = np.array(feat_list)

        grads = grads.reshape(len(grads), 1)
        coef = coef.reshape(1, len(coef))

        grads = np.tile(grads, (1, coef.shape[1]))
        coef = np.tile(coef, (grads.shape[0], 1))

        enc_back_grad = grads * coef

        dec_back_grad = self.__decrypt_gradients(enc_back_grad, is_host, n_iter, batch_index)
        self.model.backpropogate(feats, None, dec_back_grad)

    def __decrypt_gradients(self, enc_grads, is_host, n_iter, batch_index):

        if is_host:
            remote_name = self.transfer_variable.host_enc_gradient.name
            get_name = self.transfer_variable.host_dec_gradient.name
            remote_tag = self.transfer_variable.generate_transferid(self.transfer_variable.host_enc_gradient, n_iter,
                                                                    batch_index)
            get_tag = self.transfer_variable.generate_transferid(self.transfer_variable.host_dec_gradient, n_iter,
                                                                 batch_index)
        else:
            remote_name = self.transfer_variable.guest_enc_gradient.name
            get_name = self.transfer_variable.guest_dec_gradient.name
            remote_tag = self.transfer_variable.generate_transferid(self.transfer_variable.guest_enc_gradient, n_iter,
                                                                    batch_index)
            get_tag = self.transfer_variable.generate_transferid(self.transfer_variable.guest_dec_gradient, n_iter,
                                                                 batch_index)

        masked_enc_grads, grads_mask = add_random_mask(enc_grads)
        self.federation_client.remote(masked_enc_grads, name=remote_name, tag=remote_tag, role=consts.ARBITER, idx=0)

        masked_dec_grads = self.federation_client.get(name=get_name, tag=get_tag, idx=0)
        cleared_dec_grads = remove_random_mask(masked_dec_grads, grads_mask)

        return cleared_dec_grads
