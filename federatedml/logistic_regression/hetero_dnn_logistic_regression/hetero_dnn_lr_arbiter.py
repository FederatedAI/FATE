from arch.api.utils import log_utils
from federatedml.logistic_regression.hetero_dnn_logistic_regression.federation_client import FATEFederationClient
from federatedml.logistic_regression.hetero_logistic_regression import HeteroLRArbiter
from federatedml.util import consts
from federatedml.util.transfer_variable import HeteroDNNLRTransferVariable

LOGGER = log_utils.getLogger()


class HeteroDNNLRArbiter(HeteroLRArbiter):

    def __init__(self, logistic_params):
        super(HeteroDNNLRArbiter, self).__init__(logistic_params)
        self.dnn_lr_transfer_variable = HeteroDNNLRTransferVariable()
        self.federation_client = FATEFederationClient()

    def __decrypt_grads(self, enc_grads):
        for i in range(enc_grads.shape[0]):
            for j in range(enc_grads.shape[1]):
                enc_grads[i][j] = self.encrypt_operator.decrypt(enc_grads[i][j])
        return enc_grads

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
        host_dec_grads = self.__decrypt_grads(host_enc_grads)

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
        guest_dec_grads = self.__decrypt_grads(guest_enc_grads)

        LOGGER.debug("Remote guest_dec_grads to guest")
        self.federation_client.remote(guest_dec_grads, name=guest_remote_dec_gradient_name,
                                      tag=guest_remote_dec_gradient_tag, role=consts.GUEST, idx=0)
