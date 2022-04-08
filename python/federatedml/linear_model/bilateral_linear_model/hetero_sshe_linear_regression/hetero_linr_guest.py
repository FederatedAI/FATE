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

from federatedml.linear_model.bilateral_linear_model.hetero_sshe_linear_model import HeteroSSHEGuestBase
from federatedml.param.evaluation_param import EvaluateParam
from federatedml.param.hetero_sshe_linr_param import HeteroSSHELinRParam
from federatedml.protobuf.generated import linr_model_param_pb2, linr_model_meta_pb2
from federatedml.secureprotol.spdz.secure_matrix.secure_matrix import SecureMatrix
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy
from federatedml.util import consts, fate_operator, LOGGER
from federatedml.util.io_check import assert_io_num_rows_equal


class HeteroLinRGuest(HeteroSSHEGuestBase):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroLinearRegression'
        self.model_param_name = 'HeteroLinearRegressionParam'
        self.model_meta_name = 'HeteroLinearRegressionMeta'
        self.model_param = HeteroSSHELinRParam()
        # self.labels = None
        self.label_type = float

    def forward(self, weights, features, labels, suffix, cipher, batch_weight):
        self._cal_z(weights, features, suffix, cipher)
        complete_z = self.wx_self + self.wx_remote

        self.encrypted_wx = complete_z

        self.encrypted_error = complete_z - labels
        if batch_weight:
            complete_z = complete_z * batch_weight
            self.encrypted_error = self.encrypted_error * batch_weight

        tensor_name = ".".join(("complete_z",) + suffix)
        shared_z = SecureMatrix.from_source(tensor_name,
                                            complete_z,
                                            cipher,
                                            self.fixedpoint_encoder.n,
                                            self.fixedpoint_encoder)
        return shared_z

    def compute_loss(self, weights, labels, suffix, cipher=None):
        """
         Compute hetero linr loss:
            loss = (1/N)*\\sum(wx-y)^2 where y is label, w is model weight and x is features
            (wx - y)^2 = (wx_h)^2 + (wx_g - y)^2 + 2 * (wx_h * (wx_g - y))
        """
        LOGGER.info(f"[compute_loss]: Calculate loss ...")
        wxy_self = self.wx_self - labels
        wxy_self_square = (wxy_self * wxy_self).reduce(operator.add)

        wxy = (self.wx_remote * wxy_self).reduce(operator.add)
        wx_remote_square = self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                                         is_remote=False,
                                                                         cipher=None,
                                                                         wx_self_square=None)[0]
        loss = (wx_remote_square + wxy_self_square) + wxy * 2

        batch_num = self.batch_num[int(suffix[2])]
        loss = loss * (1 / (batch_num * 2))
        # loss = (wx_remote_square + wxy_self_square + 2 * wxy) / (2 * batch_num)

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

    @assert_io_num_rows_equal
    def predict(self, data_instances):
        """
        Prediction of linr
        Parameters
        ----------
        data_instances: Table of Instance, input data

        Returns
        ----------
        Table
            include input data label, predict result, predicted label
        """
        LOGGER.info("Start predict ...")
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)

        LOGGER.debug(
            f"Before_predict_reveal_strategy: {self.model_param.reveal_strategy}, {self.is_respectively_reveal}")

        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)

        pred_res = data_instances.mapValues(f)
        host_preds = self.transfer_variable.host_prob.get(idx=-1)

        LOGGER.info("Get probability from Host")

        for host_pred in host_preds:
            if not self.is_respectively_reveal:
                host_pred = self.cipher.distribute_decrypt(host_pred)
            pred_res = pred_res.join(host_pred, lambda g, h: g + h)
        predict_result = self.predict_score_to_output(data_instances=data_instances,
                                                      predict_score=pred_res,
                                                      classes=None)

        return predict_result

    def _get_param(self):
        if self.need_cv:
            param_protobuf_obj = linr_model_param_pb2.LinRModelParam()
            return param_protobuf_obj

        single_result = self.get_single_model_param()
        param_protobuf_obj = linr_model_param_pb2.LinRModelParam(**single_result)
        return param_protobuf_obj

    def _get_meta(self):
        meta_protobuf_obj = linr_model_meta_pb2.LinRModelMeta(penalty=self.model_param.penalty,
                                                              tol=self.model_param.tol,
                                                              alpha=self.alpha,
                                                              optimizer=self.model_param.optimizer,
                                                              batch_size=self.batch_size,
                                                              learning_rate=self.model_param.learning_rate,
                                                              max_iter=self.max_iter,
                                                              early_stop=self.model_param.early_stop,
                                                              fit_intercept=self.fit_intercept,
                                                              reveal_strategy=self.model_param.reveal_strategy)
        return meta_protobuf_obj

    def load_model(self, model_dict):
        result_obj, _ = super().load_model(model_dict)
        self.load_single_model(result_obj)

    def fit(self, data_instances, validate_data=None):
        LOGGER.info("Starting to fit hetero_sshe_linear_regression")
        self.prepare_fit(data_instances, validate_data)

        self.fit_single_model(data_instances, validate_data)

    def get_metrics_param(self):
        return EvaluateParam(eval_type="regression", metrics=self.metrics)
