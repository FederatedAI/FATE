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

    def transfer_pubkey(self):
        public_key = self.cipher.public_key
        self.transfer_variable.pubkey.remote(public_key, role=consts.GUEST, suffix=("host_pubkey",))
        remote_pubkey = self.transfer_variable.pubkey.get(role=consts.GUEST, idx=0,
                                                          suffix=("guest_pubkey",))
        return remote_pubkey

    def _init_weights(self, model_shape):
        # init_param_obj = copy.deepcopy(self.init_param_obj)
        # init_param_obj.fit_intercept = False
        self.init_param_obj.fit_intercept = False
        return self.initializer.init_model(model_shape, init_params=self.init_param_obj)

    def _cal_z_in_share(self, w_self, w_remote, features, suffix):
        z1 = features.dot_local(w_self)

        za_suffix = ("za",) + suffix
        za_share = self.secure_matrix_obj.secure_matrix_mul(features,
                                                            tensor_name=".".join(za_suffix),
                                                            cipher=None,
                                                            suffix=za_suffix)
        # za_share = self.secure_matrix_mul_passive(features, suffix=("za",) + suffix)

        zb_suffix = ("zb",) + suffix
        zb_share = self.secure_matrix_obj.secure_matrix_mul(w_remote,
                                                            tensor_name=".".join(zb_suffix),
                                                            cipher=self.cipher,
                                                            suffix=zb_suffix)
        # self.secure_matrix_mul_active(w_remote, cipher=self.cipher, suffix=zb_suffix)
        # zb_share = self.received_share_matrix(self.cipher, q_field=self.fixpoint_encoder.n,
        #                                       encoder=self.fixpoint_encoder, suffix=zb_suffix)
        z = z1 + za_share + zb_share
        return z

    def cal_prediction(self, w_self, w_remote, features, spdz, suffix):
        if not self.review_every_iter:
            z = self._cal_z_in_share(w_self, w_remote, features, suffix)
        else:
            z = features.dot_local(self.model_weights.coef_)

        # z_square = z * z
        # z_cube = z_square * z

        self.share_encrypted_value(suffix=suffix, is_remote=True, z=z)

        # shared_sigmoid_z = self.received_share_matrix(self.cipher,
        #                                               q_field=z.q_field,
        #                                               encoder=z.endec,
        #                                               suffix=("sigmoid_z",) + suffix)
        tensor_name = ".".join(("sigmoid_z",) + suffix)
        shared_sigmoid_z = SecureMatrix.from_source(tensor_name,
                                                    self.other_party,
                                                    self.cipher,
                                                    self.fixedpoint_encoder.n,
                                                    self.fixedpoint_encoder)

        return shared_sigmoid_z

    def compute_gradient(self, wa, wb, error: fixedpoint_table.FixedPointTensor, features, suffix):
        encoded_1_n = self.encoded_batch_num[int(suffix[1])]

        ga = error.dot_local(features)
        LOGGER.debug(f"ga: {ga}, encoded_1_n: {encoded_1_n}")
        ga = ga * encoded_1_n

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

    def compute_loss(self, spdz, suffix):

        # shared_wx = self.received_share_matrix(self.cipher, q_field=self.random_field,
        #                                        encoder=self.fixpoint_encoder, suffix=suffix)
        tensor_name = ".".join(("shared_wx",) + suffix)
        shared_wx = SecureMatrix.from_source(tensor_name,
                                             self.other_party,
                                             self.cipher,
                                             self.fixedpoint_encoder.n,
                                             self.fixedpoint_encoder)

        LOGGER.debug(f"share_wx: {type(shared_wx)}, shared_y: {type(self.shared_y)}")
        wxy = spdz.dot(shared_wx, self.shared_y, ("wxy",) + suffix).get()
        # wxy_sum = wxy_tensor.value[0]
        # wxy_sum_guest = self.transfer_variable.wxy_sum.get(idx=0, suffix=suffix)
        # wxy = wxy_tensor + wxy_sum_guest
        LOGGER.debug(f"wxy_value: {wxy}")
        wx_guest, wx_square_guest = self.share_encrypted_value(suffix=suffix, is_remote=False,
                                                               wx=None, wx_square=None)

        encrypted_wx = shared_wx + wx_guest
        encrypted_wx_sqaure = shared_wx * shared_wx + wx_square_guest + 2 * shared_wx * wx_guest
        # encrypted_wx_sqaure = wx_square_guest
        LOGGER.debug(f"encoded_batch_num: {self.encoded_batch_num}, suffix: {suffix}")
        encoded_1_n = self.encoded_batch_num[int(suffix[2])]
        LOGGER.debug(f"tmc, type: {encrypted_wx_sqaure}, {encrypted_wx}, {encoded_1_n}, {wxy}")
        loss = ((0.125 * encrypted_wx_sqaure - 0.5 * encrypted_wx).value.reduce(operator.add) +
                wxy) * encoded_1_n * -1 - np.log(0.5)
        # loss = ((0.125 * encrypted_wx_sqaure - 0.5 * encrypted_wx).value.reduce(operator.add)) * encoded_1_n * -1 - np.log(0.5)
        loss_norm = self.optimizer.loss_norm(self.model_weights)
        if loss_norm is not None:
            loss += loss_norm
        LOGGER.debug(f"loss: {loss}")
        self.transfer_variable.loss.remote(loss[0][0], suffix=suffix)

    def check_converge_by_weights(self, last_w, new_w, suffix):
        if self.is_respectively_reveal:
            return self._respectively_check(last_w[0], new_w, suffix)
        else:
            return self._unbalanced_check(suffix)

    def _respectively_check(self, last_w, new_w, suffix):
        square_sum = np.sum((last_w - new_w) ** 2)
        self.converge_transfer_variable.square_sum.remote(square_sum, role=consts.GUEST, idx=0, suffix=suffix)
        return self.converge_transfer_variable.converge_info.get(idx=0, suffix=suffix)

    def _unbalanced_check(self, suffix):
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

        if self.is_respectively_reveal:
            return self._respectively_predict(data_instances)
        else:
            return self._unbalanced_predict(data_instances)

    def _respectively_predict(self, data_instances):
        self.transfer_variable.host_prob.disable_auto_clean()

        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)
        prob_host = data_instances.mapValues(f)
        self.transfer_variable.host_prob.remote(prob_host, role=consts.GUEST, idx=0)
        LOGGER.info("Remote probability to Guest")

    def _unbalanced_predict(self, data_instances):
        encrypted_host_weights = self.transfer_variable.encrypted_host_weights.get(idx=-1)[0]
        prob_host = data_instances.mapValues(lambda v: fate_operator.vec_dot(v.features, encrypted_host_weights))
        self.transfer_variable.host_prob.remote(prob_host, role=consts.GUEST, idx=0)
        LOGGER.info("Remote probability to Guest")

    # def _get_param(self):
    #     single_result = self.get_single_model_param()
    #     single_result['need_one_vs_rest'] = False
    #     param_protobuf_obj = lr_model_param_pb2.LRModelParam(**single_result)
    #     return param_protobuf_obj

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
