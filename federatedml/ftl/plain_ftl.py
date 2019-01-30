#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()

from federatedml.optim.activation import sigmoid


class PartyModelInterface(object):

    def send_components(self):
        pass

    def receive_components(self, components):
        pass

    def send_gradients(self):
        pass

    def receive_gradients(self, gradients):
        pass

    def predict(self, X):
        pass


class PlainFTLGuestModel(PartyModelInterface):

    def __init__(self, local_model, model_param, is_trace=False):
        super(PlainFTLGuestModel, self).__init__()
        self.localModel = local_model
        self.feature_dim = local_model.get_encode_dim()
        self.alpha = model_param.alpha
        self.is_trace = is_trace
        self.logger = LOGGER

    def set_batch(self, X, y, non_overlap_indexes=None, overlap_indexes=None):
        self.X = X
        self.y = y
        self.non_overlap_indexes = non_overlap_indexes
        self.overlap_indexes = overlap_indexes
        self.phi = None

    def __compute_phi(self, U_A, y):
        length_y = len(y)
        return np.expand_dims(np.sum(y * U_A, axis=0) / length_y, axis=0)

    def _compute_components(self):
        self.U_A = self.localModel.transform(self.X)
        # y_A_u_A has shape (1, feature_dim)
        # y_A_u_A_2 has shape (feature_dim, feature_dim)
        self.phi = self.__compute_phi(self.U_A, self.y)
        self.phi_2 = np.matmul(self.phi.transpose(), self.phi)

        # y_overlap and y_overlap2 have shape (len(overlap_indexes), 1)
        self.y_overlap = self.y[self.overlap_indexes]
        self.y_overlap_2 = self.y_overlap * self.y_overlap

        if self.is_trace:
            self.logger.debug("phi shape" + str(self.phi.shape))
            self.logger.debug("phi_2 shape" + str(self.phi_2.shape))
            self.logger.debug("y_overlap shape" + str(self.y_overlap.shape))
            self.logger.debug("y_overlap_2 shape" + str(self.y_overlap_2.shape))

        # following two parameters will be sent to host
        # y_overlap_2_phi_2 has shape (len(overlap_indexes), feature_dim, feature_dim)
        # y_overlap_phi has shape (len(overlap_indexes), feature_dim)
        self.y_overlap_2_phi_2 = 0.25 * np.expand_dims(self.y_overlap_2, axis=2) * self.phi_2
        self.y_overlap_phi = -0.5 * self.y_overlap * self.phi

        self.U_A_overlap = self.U_A[self.overlap_indexes]
        # mapping_comp_A has shape (len(overlap_indexes), feature_dim)
        self.mapping_comp_A = - self.U_A_overlap / self.feature_dim

        if self.is_trace:
            self.logger.debug("y_overlap_2_phi_2 shape" + str(self.y_overlap_2_phi_2.shape))
            self.logger.debug("y_overlap_phi shape" + str(self.y_overlap_phi.shape))
            self.logger.debug("mapping_comp_A shape" + str(self.mapping_comp_A.shape))

    def send_components(self):
        self._compute_components()
        return [self.y_overlap_2_phi_2, self.y_overlap_phi, self.mapping_comp_A]

    def receive_components(self, components):
        self.U_B_overlap = components[0]
        self.U_B_overlap_2 = components[1]
        self.mapping_comp_B = components[2]
        self._update_gradients()
        self._update_loss()

    def _update_gradients(self):

        # y_overlap2 have shape (len(overlap_indexes), 1),
        # y_A_u_A has shape (1, feature_dim),
        # y_overlap2_y_A_u_A has shape (len(overlap_indexes), 1, feature_dim)
        y_overlap2_y_A_u_A = np.expand_dims(self.y_overlap_2 * self.phi, axis=1)

        # U_B_overlap_2 has shape (len(overlap_indexes), feature_dim, feature_dim)
        # tmp has shape (len(overlap_indexes), feature_dim)
        tmp = 0.25 * np.squeeze(np.matmul(y_overlap2_y_A_u_A, self.U_B_overlap_2), axis=1)

        if self.is_trace:
            self.logger.debug("tmp shape" + str(tmp.shape))
            self.logger.debug("y_overlap shape" + str(self.y_overlap.shape))
            self.logger.debug("U_B_overlap shape" + str(self.U_B_overlap.shape))

        const = np.sum(tmp, axis=0) - 0.5 * np.sum(self.y_overlap * self.U_B_overlap, axis=0)
        # grad_A_nonoverlap has shape (len(non_overlap_indexes), feature_dim)
        # grad_A_overlap has shape (len(overlap_indexes), feature_dim)
        grad_A_nonoverlap = self.alpha * const * self.y[self.non_overlap_indexes] / len(self.y)
        grad_A_overlap = self.alpha * const * self.y_overlap / len(self.y) + self.mapping_comp_B

        grad_loss_A = np.zeros((len(self.y), self.U_B_overlap.shape[1]))
        grad_loss_A[self.non_overlap_indexes, :] = grad_A_nonoverlap
        grad_loss_A[self.overlap_indexes, :] = grad_A_overlap
        self.loss_grads = grad_loss_A
        self.localModel.backpropogate(self.X, self.y, grad_loss_A)

    def send_loss(self):
        return self.loss

    def receive_loss(self, loss):
        self.loss = loss

    def _update_loss(self):
        U_A_overlap_prime = - self.U_A_overlap / self.feature_dim
        loss_overlap = np.sum(U_A_overlap_prime * self.U_B_overlap)
        loss_Y = self.__compute_loss_y(self.U_B_overlap, self.y_overlap, self.phi)
        self.loss = self.alpha * loss_Y + loss_overlap

    def __compute_loss_y(self, U_B_overlap, y_overlap, y_A_u_A):
        UB_yAuA = np.matmul(U_B_overlap, y_A_u_A.transpose())
        loss_Y = (-0.5 * np.sum(y_overlap * UB_yAuA) + 1.0 / 8 * np.sum(UB_yAuA * UB_yAuA)) + len(y_overlap) * np.log(2)
        return loss_Y

    def get_loss_grads(self):
        return self.loss_grads

    def predict(self, U_B):
        if self.phi is None:
            self.U_A = self.localModel.transform(self.X)
            self.phi = self.__compute_phi(self.U_A, self.y)
        return sigmoid(np.matmul(U_B, self.phi.transpose()))

    def restore_model(self, model_parameters):
        self.localModel.restore_model(model_parameters)

    def get_model_parameters(self):
        return self.localModel.get_model_parameters()


class PlainFTLHostModel(PartyModelInterface):

    def __init__(self, local_model, model_param, is_trace=False):
        super(PlainFTLHostModel, self).__init__()
        self.localModel = local_model
        self.feature_dim = local_model.get_encode_dim()
        self.alpha = model_param.alpha
        self.is_trace = is_trace
        self.logger = LOGGER

    def set_batch(self, X, overlap_indexes):
        self.X = X
        self.overlap_indexes = overlap_indexes

    def _compute_components(self):
        self.U_B = self.localModel.transform(self.X)

        # following three parameters will be sent to guest
        # U_B_overlap has shape (len(overlap_indexes), feature_dim)
        # U_B_overlap_2 has shape (len(overlap_indexes), feature_dim, feature_dim)
        # mapping_comp_B has shape (len(overlap_indexes), feature_dim)
        self.U_B_overlap = self.U_B[self.overlap_indexes]
        self.U_B_overlap_2 = np.matmul(np.expand_dims(self.U_B_overlap, axis=2), np.expand_dims(self.U_B_overlap, axis=1))
        self.mapping_comp_B = - self.U_B_overlap / self.feature_dim

        if self.is_trace:
            self.logger.debug("U_B_overlap shape" + str(self.U_B_overlap.shape))
            self.logger.debug("U_B_overlap_2 shape" + str(self.U_B_overlap_2.shape))
            self.logger.debug("mapping_comp_B shape" + str(self.mapping_comp_B.shape))

    def send_components(self):
        self._compute_components()
        return [self.U_B_overlap, self.U_B_overlap_2, self.mapping_comp_B]

    def receive_components(self, components):
        self.comp_A_beta1 = components[0]
        self.comp_A_beta2 = components[1]
        self.mapping_comp_A = components[2]
        self._update_gradients()

    def _update_gradients(self):
        U_B_overlap_ex = np.expand_dims(self.U_B_overlap, axis=1)
        U_B_comp_A_beta1 = np.matmul(U_B_overlap_ex, self.comp_A_beta1)
        grad_l1_B = np.squeeze(U_B_comp_A_beta1, axis=1) + self.comp_A_beta2
        grad_loss_B = self.alpha * grad_l1_B + self.mapping_comp_A
        self.loss_grads = grad_loss_B
        self.localModel.backpropogate(self.X[self.overlap_indexes], None, grad_loss_B)

    def get_loss_grads(self):
        return self.loss_grads

    def predict(self, X):
        return self.localModel.transform(X)

    def restore_model(self, model_parameters):
        self.localModel.restore_model(model_parameters)

    def get_model_parameters(self):
        return self.localModel.get_model_parameters()


class LocalPlainFederatedTransferLearning(object):

    def __init__(self, guest: PlainFTLGuestModel, host: PlainFTLHostModel, private_key=None):
        super(LocalPlainFederatedTransferLearning, self).__init__()
        self.guest = guest
        self.host = host
        self.private_key = private_key

    def fit(self, X_A, X_B, y, overlap_indexes, non_overlap_indexes):
        self.guest.set_batch(X_A, y, non_overlap_indexes, overlap_indexes)
        self.host.set_batch(X_B, overlap_indexes)
        comp_B = self.host.send_components()
        comp_A = self.guest.send_components()
        self.guest.receive_components(comp_B)
        self.host.receive_components(comp_A)
        loss = self.guest.send_loss()
        return loss

    def predict(self, X_B):
        msg = self.host.predict(X_B)
        return self.guest.predict(msg)
