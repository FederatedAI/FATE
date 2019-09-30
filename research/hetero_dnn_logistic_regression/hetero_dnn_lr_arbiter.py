import time

from federatedml.util.transfer import HeteroDNNLRTransferVariable

from arch.api.utils import log_utils
from federatedml.ftl.eggroll_computation.helper import distribute_decrypt_matrix
from federatedml.linear_model.logistic_regression import HeteroLRArbiter
from federatedml.util import consts
from research.hetero_dnn_logistic_regression.federation_client import FATEFederationClient

LOGGER = log_utils.getLogger()


class HeteroDNNLRArbiter(HeteroLRArbiter):

    def __init__(self, logistic_params):
        super(HeteroDNNLRArbiter, self).__init__(logistic_params)
        self.dnn_lr_transfer_variable = HeteroDNNLRTransferVariable()
        self.federation_client = FATEFederationClient()

    def _decrypt(self, enc_item):
        return distribute_decrypt_matrix(self.cipher_operator.get_privacy_key(), enc_item)

    def _decrypt_grads(self, enc_grads):
        if type(enc_grads) is list:
            dec_grad_list = []
            for enc_grad in enc_grads:
                dec_grad = self._decrypt(enc_grad)
                dec_grad_list.append(dec_grad)
            return dec_grad_list
        else:
            return self._decrypt(enc_grads)

    def perform_subtasks(self, **training_info):
        n_iter = training_info["iteration"]
        batch_index = training_info["batch_index"]

        # decrypt host encrypted gradient
        host_get_enc_gradient_name = self.dnn_lr_transfer_variable.host_enc_gradient.name
        host_get_enc_gradient_tag = self.dnn_lr_transfer_variable.generate_transferid(
            self.dnn_lr_transfer_variable.host_enc_gradient, n_iter, batch_index)

        host_remote_dec_gradient_name = self.dnn_lr_transfer_variable.host_dec_gradient.name
        host_remote_dec_gradient_tag = self.dnn_lr_transfer_variable.generate_transferid(
            self.dnn_lr_transfer_variable.host_dec_gradient, n_iter, batch_index)

        LOGGER.debug("get host_enc_grads from arbiter")
        host_enc_grads = self.federation_client.get(name=host_get_enc_gradient_name, tag=host_get_enc_gradient_tag,
                                                    idx=0)

        start = time.time()
        host_dec_grads = self._decrypt_grads(host_enc_grads)
        end = time.time()
        LOGGER.debug("@ host_dec_grads shape:" + str(len(host_dec_grads)))
        LOGGER.debug("@ decrypt: host_dec_grads time:" + str(end - start))

        LOGGER.debug("Remote host_dec_grads to host")
        self.federation_client.remote(host_dec_grads, name=host_remote_dec_gradient_name,
                                      tag=host_remote_dec_gradient_tag, role=consts.HOST, idx=0)

        # decrypt guest encrypted gradient
        guest_get_enc_gradient_name = self.dnn_lr_transfer_variable.guest_enc_gradient.name
        guest_get_enc_gradient_tag = self.dnn_lr_transfer_variable.generate_transferid(
            self.dnn_lr_transfer_variable.guest_enc_gradient, n_iter, batch_index)
        guest_remote_dec_gradient_name = self.dnn_lr_transfer_variable.guest_dec_gradient.name
        guest_remote_dec_gradient_tag = self.dnn_lr_transfer_variable.generate_transferid(
            self.dnn_lr_transfer_variable.guest_dec_gradient, n_iter, batch_index)

        LOGGER.debug("get guest_enc_grads from arbiter")
        guest_enc_grads = self.federation_client.get(name=guest_get_enc_gradient_name, tag=guest_get_enc_gradient_tag,
                                                     idx=0)

        start = time.time()
        guest_dec_grads = self._decrypt_grads(guest_enc_grads)
        end = time.time()
        LOGGER.debug("@ guest_dec_grads shape:" + str(len(guest_dec_grads)))
        LOGGER.debug("@ decrypt: guest_dec_grads time:" + str(end - start))

        LOGGER.debug("Remote guest_dec_grads to guest")
        self.federation_client.remote(guest_dec_grads, name=guest_remote_dec_gradient_name,
                                      tag=guest_remote_dec_gradient_tag, role=consts.GUEST, idx=0)
