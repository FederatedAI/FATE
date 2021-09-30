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

from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.linear_model.logistic_regression.hetero_sshe_logistic_regression.hetero_lr_base import HeteroLRBase
from federatedml.optim import activation
from federatedml.protobuf.generated import lr_model_param_pb2
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

    def _init_model(self, params):
        super()._init_model(params)

    def transfer_pubkey(self):
        public_key = self.cipher.public_key
        self.transfer_variable.pubkey.remote(public_key, role=consts.HOST, suffix=("guest_pubkey",))
        remote_pubkey = self.transfer_variable.pubkey.get(role=consts.HOST, idx=0,
                                                          suffix=("host_pubkey",))
        return remote_pubkey

    def _cal_z_in_share(self, w_self, w_remote, features, suffix):
        z1 = features.dot_local(w_self.value, fit_intercept=self.fit_intercept)

        za_suffix = ("za",) + suffix

        za_share = self.secure_matrix_obj.secure_matrix_mul(w_remote,
                                                            tensor_name=".".join(za_suffix),
                                                            cipher=self.cipher,
                                                            suffix=za_suffix)
        zb_suffix = ("zb",) + suffix
        zb_share = self.secure_matrix_obj.secure_matrix_mul(features,
                                                            tensor_name=".".join(zb_suffix),
                                                            cipher=None,
                                                            suffix=zb_suffix)
        # self.secure_matrix_mul_active(w_remote, cipher=self.cipher, suffix=za_suffix)
        # za_share = self.received_share_matrix(self.cipher, q_field=self.fixpoint_encoder.n,
        #                                       encoder=self.fixpoint_encoder, suffix=za_suffix)
        # zb_share = self.secure_matrix_mul_passive(features,
        #                                           suffix=("zb",) + suffix)

        z = z1 + za_share + zb_share
        return z

    def _compute_sigmoid(self, z, remote_z):
        # z_square = z * z

        complete_z = remote_z + z
        # self.z_square = z_square + remote_z_square
        # self.z_square = self.z_square + 2 * remote_z * z
        sigmoid_z = complete_z * 0.25 + 0.5

        # complete_z_cube = remote_z_cube + remote_z_square * z * 3 + remote_z * z_square * 3 + z_cube
        # sigmoid_z = complete_z * 0.197 - complete_z_cube * 0.004 + 0.5
        return sigmoid_z

    def cal_prediction(self, w_self, w_remote, features, spdz, suffix):
        if not self.review_every_iter:
            z = self._cal_z_in_share(w_self, w_remote, features, suffix)
        else:
            LOGGER.debug(f"Calculate z directly.")
            z = features.dot_local(self.model_weights.coef_)
            if self.model_weights.fit_intercept:
                z.value = z.value.mapValues(lambda x: x + self.model_weights.intercept_)

        remote_z = self.share_encrypted_value(suffix=suffix, is_remote=False, z=None)[0]
        sigmoid_z = self._compute_sigmoid(z, remote_z)

        # sigmoid_z = complete_z * 0.2 + 0.5
        self.encrypted_wx = sigmoid_z
        encrypted_error = sigmoid_z.value.join(self.labels, operator.sub)
        self.encrypted_error = fixedpoint_table.PaillierFixedPointTensor(
            encrypted_error
        )
        # shared_sigmoid_z = self.share_matrix(sigmoid_z, suffix=("sigmoid_z",) + suffix)
        tensor_name = ".".join(("sigmoid_z",) + suffix)
        shared_sigmoid_z = SecureMatrix.from_source(tensor_name,
                                                    sigmoid_z,
                                                    self.cipher,
                                                    self.fixpoint_encoder.n,
                                                    self.fixpoint_encoder)
        return shared_sigmoid_z

    def compute_gradient(self, wa, wb, error, features, suffix):
        encoded_1_n = self.encoded_batch_num[int(suffix[1])]
        error_1_n = error * encoded_1_n

        encrypt_g = self.encrypted_error.value.join(features.value, operator.mul).reduce(operator.add) * encoded_1_n
        if self.fit_intercept:
            bias = self.encrypted_error.value.reduce(operator.add) * encoded_1_n
            encrypt_g = np.array(list(encrypt_g) + list(bias))
        encrypt_g = fixedpoint_numpy.PaillierFixedPointTensor(encrypt_g)
        # gb2 = self.share_matrix(encrypt_g, suffix=("encrypt_g",) + suffix)
        tensor_name = ".".join(("encrypt_g",) + suffix)
        gb2 = SecureMatrix.from_source(tensor_name,
                                       encrypt_g,
                                       self.cipher,
                                       self.fixpoint_encoder.n,
                                       self.fixpoint_encoder)

        ga2_suffix = ("ga2",) + suffix
        ga2_2 = self.secure_matrix_obj.secure_matrix_mul(error_1_n,
                                                         tensor_name=".".join(ga2_suffix),
                                                         cipher=self.cipher,
                                                         suffix=ga2_suffix)

        # self.secure_matrix_mul_active(error_1_n, cipher=self.cipher,
        #                               suffix=ga2_suffix)
        # ga2_2 = self.received_share_matrix(self.cipher, q_field=self.fixpoint_encoder.n,
        #                                    encoder=self.fixpoint_encoder, suffix=ga2_suffix)

        # wb = wb - gb2 * self.model_param.learning_rate
        ga2_2 = ga2_2.reshape(ga2_2.shape[0])
        # LOGGER.debug(f"wa shape: {wa.shape}, ga_shape: {ga2_2.shape}")
        # wa = wa - ga2_2 * self.model_param.learning_rate
        # wa = wa.reshape(wa.shape[-1])

        return ga2_2, gb2

    def compute_loss(self, spdz, suffix):
        """
        Use Taylor series expand log loss:
        Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
        Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + wxy + 1/8(wx)^2)
        """

        # shared_wx = self.share_matrix(self.encrypted_wx, suffix=suffix)

        tensor_name = ".".join(("shared_wx",) + suffix)
        shared_wx = SecureMatrix.from_source(tensor_name,
                                             self.encrypted_wx,
                                             self.cipher,
                                             self.fixpoint_encoder.n,
                                             self.fixpoint_encoder)

        wxy = spdz.dot(shared_wx, self.shared_y, ("wxy",) + suffix).get()
        LOGGER.debug(f"wxy_value: {wxy}")
        # wxy_sum = wxy_tensor.value[0]
        # self.transfer_variable.wxy_sum.remote(wxy_tensor, suffix=suffix)
        wx_square = shared_wx * shared_wx
        # LOGGER.debug(f"shared_wx: {shared_wx}, wx_square: {wx_square}, wxy_sum: {wxy_sum}")
        self.share_encrypted_value(suffix=suffix, is_remote=True, wx=shared_wx,
                                   wx_square=wx_square)

        loss = self.transfer_variable.loss.get(idx=0, suffix=suffix)
        loss = self.cipher.decrypt(loss)
        loss_norm = self.optimizer.loss_norm(self.model_weights)
        LOGGER.debug(f"loss: {loss}, loss_norm: {loss_norm}")
        if loss_norm:
            loss += loss_norm
        return loss

    def check_converge_by_weights(self, last_w, new_w, suffix):
        if self.is_respectively_reveal:
            return self._respectively_check(last_w, new_w, suffix)
        else:
            new_w = np.append(new_w, self.host_model_weights.unboxed)
            return self._unbalanced_check(new_w, suffix)

    def _respectively_check(self, last_w, new_w, suffix):
        square_sum = np.sum((last_w - new_w) ** 2)
        host_sums = self.converge_transfer_variable.square_sum.get(suffix=suffix)
        for hs in host_sums:
            square_sum += hs
        norm_diff = np.sqrt(square_sum)
        is_converge = False
        if norm_diff < self.model_param.tol:
            is_converge = True
        LOGGER.debug(f"n_iter: {self.n_iter_}, diff: {norm_diff}")
        self.converge_transfer_variable.converge_info.remote(is_converge, role=consts.HOST, suffix=suffix)
        return is_converge

    def _unbalanced_check(self, new_weight, suffix):
        is_converge = self.converge_func.is_converge(new_weight)
        self.converge_transfer_variable.converge_info.remote(is_converge, role=consts.HOST, suffix=suffix)
        return is_converge

    @assert_io_num_rows_equal
    def predict(self, data_instances):
        """
        Prediction of lr
        Parameters
        ----------
        data_instances: DTable of Instance, input data

        Returns
        ----------
        DTable
            include input data label, predict probably, label
        """
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        if self.need_one_vs_rest:
            predict_result = self.one_vs_rest_obj.predict(data_instances)
            return predict_result
        LOGGER.debug(
            f"Before_predict_review_strategy: {self.model_param.reveal_strategy}, {self.is_respectively_reveal}")

        if self.is_respectively_reveal:
            return self._respectively_predict(data_instances)
        else:
            return self._unbalanced_predict(data_instances)

    def _respectively_predict(self, data_instances):
        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)
        # pred_prob = data_instances.mapValues(lambda v: fate_operator.vec_dot(v.features, self.model_weights.coef_)
        #                                                + self.model_weights.intercept_)
        pred_prob = data_instances.mapValues(f)
        host_probs = self.transfer_variable.host_prob.get(idx=-1)

        LOGGER.info("Get probability from Host")

        # guest probability
        for host_prob in host_probs:
            pred_prob = pred_prob.join(host_prob, lambda g, h: g + h)
        pred_prob = pred_prob.mapValues(lambda p: activation.sigmoid(p))
        threshold = self.model_param.predict_param.threshold
        predict_result = self.predict_score_to_output(data_instances, pred_prob, classes=[0, 1], threshold=threshold)

        return predict_result

    def _unbalanced_predict(self, data_instances):
        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)
        pred_prob = data_instances.mapValues(f)
        for idx, host_weights in enumerate([self.host_model_weights]):
            encrypted_host_weight = self.cipher.recursive_encrypt(host_weights.coef_)
            self.transfer_variable.encrypted_host_weights.remote(encrypted_host_weight,
                                                                 role=consts.HOST,
                                                                 idx=idx)
        host_probs = self.transfer_variable.host_prob.get(idx=-1)
        for host_prob in host_probs:
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
            host_models = []
            for idx, hw in enumerate([self.host_model_weights]):
                host_weights = lr_model_param_pb2.HostWeights(
                    host_weights=list(hw.unboxed),
                    party_id=str(self.component_properties.host_party_idlist[idx]))
                host_models.append(host_weights)
            result["host_models"] = host_models
        return result

    def load_single_model(self, single_model_obj):
        super(HeteroLRGuest, self).load_single_model(single_model_obj)
        if not self.is_respectively_reveal:
            hw = list(single_model_obj.host_models)[0]
            weights = np.array(hw.host_weights)
            self.host_model_weights = LinearModelWeights(weights, fit_intercept=False)

    def get_model_summary(self):
        summary = super(HeteroLRGuest, self).get_model_summary()
        if self.host_model_weights is not None:
            host_weights = {}
            for idx, hw in enumerate([self.host_model_weights]):
                host_weights[f"host_{idx}"] = list(hw.unboxed)
            summary["host_weights"] = host_weights
        return summary
