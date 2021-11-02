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
from federatedml.protobuf.generated import lr_model_param_pb2
from federatedml.secureprotol.spdz.secure_matrix.secure_matrix import SecureMatrix
from federatedml.secureprotol.spdz.tensor import fixedpoint_table, fixedpoint_numpy
from federatedml.util import consts, LOGGER
from federatedml.util import fate_operator


class HeteroLRHost(HeteroLRBase):
    def __init__(self):
        super().__init__()
        self.data_batch_count = []
        self.wx_self = None

    def _cal_z_in_share(self, w_self, w_remote, features, suffix):
        z1 = features.dot_local(w_self)

        za_suffix = ("za",) + suffix
        za_share = self.secure_matrix_obj.secure_matrix_mul(features,
                                                            tensor_name=".".join(za_suffix),
                                                            cipher=None,
                                                            suffix=za_suffix)

        zb_suffix = ("zb",) + suffix
        zb_share = self.secure_matrix_obj.secure_matrix_mul(w_remote,
                                                            tensor_name=".".join(zb_suffix),
                                                            cipher=self.cipher,
                                                            suffix=zb_suffix)

        z = z1 + za_share + zb_share
        return z

    def forward(self, weights, features, suffix):
        if not self.review_every_iter:
            w_self, w_remote = weights
            z = self._cal_z_in_share(w_self, w_remote, features, suffix)
        else:
            w = weights.unboxed
            z = features.dot_local(w)

        self.wx_self = z

        # # DEBUG;
        # de_wx_self = self.fixedpoint_encoder.decode(self.wx_self.value.reduce(operator.add))
        # LOGGER.info(f"forward: de_wx_self: {de_wx_self}")

        self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                      is_remote=True,
                                                      cipher=self.cipher,
                                                      z=z)

        tensor_name = ".".join(("sigmoid_z",) + suffix)
        shared_sigmoid_z = SecureMatrix.from_source(tensor_name,
                                                    self.other_party,
                                                    self.cipher,
                                                    self.fixedpoint_encoder.n,
                                                    self.fixedpoint_encoder)

        return shared_sigmoid_z

    def backward(self, error: fixedpoint_table.FixedPointTensor, features, suffix):
        batch_num = self.batch_num[int(suffix[1])]

        ga = error.dot_local(features)
        LOGGER.debug(f"ga: {ga}, batch_num: {batch_num}")
        ga = ga * (1 / batch_num)

        zb_suffix = ("ga2",) + suffix
        ga2_1 = self.secure_matrix_obj.secure_matrix_mul(features,
                                                         tensor_name=".".join(zb_suffix),
                                                         cipher=None,
                                                         suffix=zb_suffix)

        LOGGER.debug(f"ga2_1: {ga2_1}")

        ga_new = ga + ga2_1

        tensor_name = ".".join(("encrypt_g",) + suffix)
        gb1 = SecureMatrix.from_source(tensor_name,
                                       self.other_party,
                                       self.cipher,
                                       self.fixedpoint_encoder.n,
                                       self.fixedpoint_encoder,
                                       is_fixedpoint_table=False)

        LOGGER.debug(f"gb1: {gb1}")

        return ga_new, gb1

    def compute_loss(self, weights=None, suffix=None):
        """
          Use Taylor series expand log loss:
          Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
          Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx - 1/8(wx)^2)
        """

        wx_self_square = (self.wx_self * self.wx_self).reduce(operator.add)
        LOGGER.debug(f"wx_self_square: {wx_self_square}")

        self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                      is_remote=True,
                                                      cipher=self.cipher,
                                                      wx_self_square=wx_self_square)

        tensor_name = ".".join(("shared_loss",) + suffix)
        share_loss = SecureMatrix.from_source(tensor_name=tensor_name,
                                              source=self.other_party,
                                              cipher=self.cipher,
                                              q_field=self.fixedpoint_encoder.n,
                                              encoder=self.fixedpoint_encoder,
                                              is_fixedpoint_table=False)

        LOGGER.debug(f"share_loss: {share_loss}")

        if self.review_every_iter:
            loss_norm = self.optimizer.loss_norm(weights)
            if loss_norm:
                share_loss += loss_norm
            LOGGER.debug(f"share_loss+loss_norm: {share_loss}")
            tensor_name = ".".join(("loss",) + suffix)
            share_loss.broadcast_reconstruct_share(tensor_name=tensor_name)
        else:
            tensor_name = ".".join(("loss",) + suffix)
            share_loss.broadcast_reconstruct_share(tensor_name=tensor_name)
            if self.optimizer.penalty == consts.L2_PENALTY:
                w_self, w_remote = weights

                w_encode = np.hstack((w_self.value, w_remote.value))

                w_encode = np.array([w_encode])

                LOGGER.debug(f"w_encode: {w_encode}")
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

                loss_norm = w_tensor.dot(w_tensor_transpose, target_name=loss_norm_tensor_name)
                loss_norm.broadcast_reconstruct_share()

    def _review_every_iter_weights_check(self, last_w, new_w, suffix):
        square_sum = np.sum((last_w - new_w) ** 2)
        self.converge_transfer_variable.square_sum.remote(square_sum, role=consts.GUEST, idx=0, suffix=suffix)
        return self.converge_transfer_variable.converge_info.get(idx=0, suffix=suffix)

    def predict(self, data_instances):
        LOGGER.info("Start predict ...")
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        if self.need_one_vs_rest:
            self.one_vs_rest_obj.predict(data_instances)
            return

        LOGGER.debug(f"Before_predict_review_strategy: {self.model_param.reveal_strategy},"
                     f" {self.is_respectively_reveal}")

        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)
        prob_host = data_instances.mapValues(f)
        self.transfer_variable.host_prob.remote(prob_host, role=consts.GUEST, idx=0)
        LOGGER.info("Remote probability to Guest")

    def _get_param(self):
        if self.need_cv:
            param_protobuf_obj = lr_model_param_pb2.LRModelParam()
            return param_protobuf_obj

        self.header = self.header if self.header else []
        LOGGER.debug("In get_param, self.need_one_vs_rest: {}".format(self.need_one_vs_rest))

        if self.need_one_vs_rest:
            # one_vs_rest_class = list(map(str, self.one_vs_rest_obj.classes))
            one_vs_rest_result = self.one_vs_rest_obj.save(lr_model_param_pb2.SingleModel)
            single_result = {'header': self.header, 'need_one_vs_rest': True, "best_iteration": -1}
        else:
            one_vs_rest_result = None
            single_result = self.get_single_model_param()
            single_result['need_one_vs_rest'] = False
        single_result['one_vs_rest_result'] = one_vs_rest_result

        param_protobuf_obj = lr_model_param_pb2.LRModelParam(**single_result)

        return param_protobuf_obj

