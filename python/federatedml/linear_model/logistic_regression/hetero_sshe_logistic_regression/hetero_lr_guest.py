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

import functools
import operator

import numpy as np

from federatedml.linear_model.logistic_regression.hetero_sshe_logistic_regression.hetero_lr_base import HeteroLRBase
from federatedml.optim import activation
from federatedml.protobuf.generated import lr_model_param_pb2
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fate_paillier import PaillierPublicKey, PaillierPrivateKey
from federatedml.secureprotol.spdz.secure_matrix.secure_matrix import SecureMatrix
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.util import LOGGER, consts
from federatedml.util import fate_operator
from federatedml.util.io_check import assert_io_num_rows_equal


class HeteroLRGuest(HeteroLRBase):

    def __init__(self):
        super().__init__()
        self.encrypted_error = None
        self.encrypted_wx = None
        self.z_square = None
        self.wx_self = None
        self.wx_remote = None

    def _cal_z_in_share(self, w_self, w_remote, features, suffix, cipher):
        z1 = features.dot_local(w_self)

        za_suffix = ("za",) + suffix

        za_share = self.secure_matrix_obj.secure_matrix_mul(w_remote,
                                                            tensor_name=".".join(za_suffix),
                                                            cipher=cipher,
                                                            suffix=za_suffix)
        zb_suffix = ("zb",) + suffix
        zb_share = self.secure_matrix_obj.secure_matrix_mul(features,
                                                            tensor_name=".".join(zb_suffix),
                                                            cipher=None,
                                                            suffix=zb_suffix)

        z = z1 + za_share + zb_share
        return z

    def _compute_sigmoid(self, z, remote_z):
        complete_z = z + remote_z

        sigmoid_z = complete_z * 0.25 + 0.5

        return sigmoid_z

    def forward(self, weights, features, suffix, cipher):
        if not self.reveal_every_iter:
            LOGGER.info(f"[forward]: Calculate z in share...")
            w_self, w_remote = weights
            z = self._cal_z_in_share(w_self, w_remote, features, suffix, cipher)
        else:
            LOGGER.info(f"[forward]: Calculate z directly...")
            w = weights.unboxed
            z = features.dot_local(w)

        remote_z = self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                                 is_remote=False,
                                                                 cipher=None,
                                                                 z=None)[0]

        self.wx_self = z
        self.wx_remote = remote_z

        sigmoid_z = self._compute_sigmoid(z, remote_z)

        self.encrypted_wx = self.wx_self + self.wx_remote

        self.encrypted_error = sigmoid_z - self.labels

        tensor_name = ".".join(("sigmoid_z",) + suffix)
        shared_sigmoid_z = SecureMatrix.from_source(tensor_name,
                                                    sigmoid_z,
                                                    cipher,
                                                    self.fixedpoint_encoder.n,
                                                    self.fixedpoint_encoder)
        return shared_sigmoid_z

    def backward(self, error, features, suffix, cipher):
        LOGGER.info(f"[backward]: Calculate gradient...")
        batch_num = self.batch_num[int(suffix[1])]
        error_1_n = error * (1 / batch_num)

        ga2_suffix = ("ga2",) + suffix
        ga2_2 = self.secure_matrix_obj.secure_matrix_mul(error_1_n,
                                                         tensor_name=".".join(ga2_suffix),
                                                         cipher=cipher,
                                                         suffix=ga2_suffix,
                                                         is_fixedpoint_table=False)

        # LOGGER.debug(f"ga2_2: {ga2_2}")

        encrypt_g = self.encrypted_error.dot(features) * (1 / batch_num)

        # LOGGER.debug(f"encrypt_g: {encrypt_g}")

        tensor_name = ".".join(("encrypt_g",) + suffix)
        gb2 = SecureMatrix.from_source(tensor_name,
                                       encrypt_g,
                                       self.cipher,
                                       self.fixedpoint_encoder.n,
                                       self.fixedpoint_encoder)

        # LOGGER.debug(f"gb2: {gb2}")

        return gb2, ga2_2

    def compute_loss(self, weights, suffix, cipher=None):
        """
          Use Taylor series expand log loss:
          Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
          Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx -1/8(wx)^2)
        """
        LOGGER.info(f"[compute_loss]: Calculate loss ...")
        wx = (-0.5 * self.encrypted_wx).reduce(operator.add)
        ywx = (self.encrypted_wx * self.labels).reduce(operator.add)

        wx_square = (2 * self.wx_remote * self.wx_self).reduce(operator.add) + \
                    (self.wx_self * self.wx_self).reduce(operator.add)

        wx_remote_square = self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                                         is_remote=False,
                                                                         cipher=None,
                                                                         wx_self_square=None)[0]

        wx_square = (wx_remote_square + wx_square) * -0.125

        batch_num = self.batch_num[int(suffix[2])]
        loss = (wx + ywx + wx_square) * (-1 / batch_num) - np.log(0.5)

        tensor_name = ".".join(("shared_loss",) + suffix)
        share_loss = SecureMatrix.from_source(tensor_name=tensor_name,
                                              source=loss,
                                              cipher=None,
                                              q_field=self.fixedpoint_encoder.n,
                                              encoder=self.fixedpoint_encoder)

        tensor_name = ".".join(("loss",) + suffix)
        loss = share_loss.get(tensor_name=tensor_name,
                              broadcast=False)[0]

        if self.reveal_every_iter:
            loss_norm = self.optimizer.loss_norm(weights)
            if loss_norm:
                loss += loss_norm
        else:
            if self.optimizer.penalty == consts.L2_PENALTY:
                w_self, w_remote = weights

                w_encode = np.hstack((w_remote.value, w_self.value))

                w_encode = np.array([w_encode])

                w_tensor_name = ".".join(("loss_norm_w",) + suffix)
                w_tensor = fixedpoint_numpy.FixedPointTensor(value=w_encode,
                                                             q_field=self.fixedpoint_encoder.n,
                                                             endec=self.fixedpoint_encoder,
                                                             tensor_name=w_tensor_name)

                w_tensor_transpose_name = ".".join(("loss_norm_w_transpose",) + suffix)
                w_tensor_transpose = fixedpoint_numpy.FixedPointTensor(value=w_encode.T,
                                                                       q_field=self.fixedpoint_encoder.n,
                                                                       endec=self.fixedpoint_encoder,
                                                                       tensor_name=w_tensor_transpose_name)

                loss_norm_tensor_name = ".".join(("loss_norm",) + suffix)

                loss_norm = w_tensor.dot(w_tensor_transpose, target_name=loss_norm_tensor_name).get(broadcast=False)
                loss_norm = 0.5 * self.optimizer.alpha * loss_norm[0][0]
                loss = loss + loss_norm

        LOGGER.info(f"[compute_loss]: loss={loss}, reveal_every_iter={self.reveal_every_iter}")

        return loss

    def _reveal_every_iter_weights_check(self, last_w, new_w, suffix):
        square_sum = np.sum((last_w - new_w) ** 2)
        host_sums = self.converge_transfer_variable.square_sum.get(suffix=suffix)
        for hs in host_sums:
            square_sum += hs
        weight_diff = np.sqrt(square_sum)
        is_converge = False
        if weight_diff < self.model_param.tol:
            is_converge = True
        LOGGER.info(f"n_iter: {self.n_iter_}, weight_diff: {weight_diff}")
        self.converge_transfer_variable.converge_info.remote(is_converge, role=consts.HOST, suffix=suffix)
        return is_converge

    @assert_io_num_rows_equal
    def predict(self, data_instances):
        """
        Prediction of lr
        Parameters
        ----------
        data_instances: Table of Instance, input data

        Returns
        ----------
        Table
            include input data label, predict probably, label
        """
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        if self.need_one_vs_rest:
            predict_result = self.one_vs_rest_obj.predict(data_instances)
            return predict_result
        LOGGER.debug(
            f"Before_predict_reveal_strategy: {self.model_param.reveal_strategy}, {self.is_respectively_reveal}")

        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)

        pred_prob = data_instances.mapValues(f)
        host_probs = self.transfer_variable.host_prob.get(idx=-1)

        LOGGER.info("Get probability from Host")

        # guest probability
        for host_prob in host_probs:
            if not self.is_respectively_reveal:
                host_prob = self.cipher.distribute_decrypt(host_prob)
            pred_prob = pred_prob.join(host_prob, lambda g, h: g + h)
        pred_prob = pred_prob.mapValues(lambda p: activation.sigmoid(p))
        threshold = self.model_param.predict_param.threshold
        predict_result = self.predict_score_to_output(data_instances, pred_prob, classes=[0, 1], threshold=threshold)

        return predict_result

    def _get_param(self):
        if self.need_cv:
            param_protobuf_obj = lr_model_param_pb2.LRModelParam()
            return param_protobuf_obj

        if self.need_one_vs_rest:
            one_vs_rest_result = self.one_vs_rest_obj.save(lr_model_param_pb2.SingleModel)
            single_result = {'header': self.header, 'need_one_vs_rest': True, "best_iteration": -1}
        else:
            one_vs_rest_result = None
            single_result = self.get_single_model_param()

            single_result['need_one_vs_rest'] = False
        single_result['one_vs_rest_result'] = one_vs_rest_result
        LOGGER.debug(f"saved_model: {single_result}")
        param_protobuf_obj = lr_model_param_pb2.LRModelParam(**single_result)
        return param_protobuf_obj

    def get_single_model_param(self, model_weights=None, header=None):
        result = super().get_single_model_param(model_weights, header)
        if not self.is_respectively_reveal:
            result["cipher"] = dict(public_key=dict(n=str(self.cipher.public_key.n)),
                                    private_key=dict(p=str(self.cipher.privacy_key.p),
                                                     q=str(self.cipher.privacy_key.q)))
        return result

    def load_single_model(self, single_model_obj):
        super(HeteroLRGuest, self).load_single_model(single_model_obj)
        if not self.is_respectively_reveal:
            cipher_info = single_model_obj.cipher
            self.cipher = PaillierEncrypt()
            public_key = PaillierPublicKey(int(cipher_info.public_key.n))
            privacy_key = PaillierPrivateKey(public_key, int(cipher_info.private_key.p), int(cipher_info.private_key.q))
            self.cipher.set_public_key(public_key=public_key)
            self.cipher.set_privacy_key(privacy_key=privacy_key)

    def get_model_summary(self):
        summary = super(HeteroLRGuest, self).get_model_summary()
        return summary
