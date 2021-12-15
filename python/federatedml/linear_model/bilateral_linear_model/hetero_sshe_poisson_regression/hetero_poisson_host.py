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

from federatedml.framework.hetero.procedure import batch_generator
from federatedml.transfer_variable.transfer_class.batch_generator_transfer_variable import \
    BatchGeneratorTransferVariable
from federatedml.framework.hetero.procedure.hetero_sshe_linear_model import HeteroSSHEHostBase
from federatedml.linear_model.coordinated_linear_model.poisson_regression.\
    base_poisson_regression import BasePoissonRegression
from federatedml.param.hetero_sshe_poisson_param import HeteroSSHEPoissonParam
from federatedml.protobuf.generated import poisson_model_param_pb2, poisson_model_meta_pb2
from federatedml.secureprotol.spdz.secure_matrix.secure_matrix import SecureMatrix
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.util import consts, fate_operator, LOGGER


class HeteroPoissonHost(HeteroSSHEHostBase):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroPoissonRegression'
        self.model_param_name = 'HeteroPoissonRegressionParam'
        self.model_meta_name = 'HeteroPoissonRegressionMeta'
        self.model_param = HeteroSSHEPoissonParam()
        self.labels = None
        self.wx_self = None

    def forward(self, weights, features, suffix, cipher):
        """if not self.reveal_every_iter:
            LOGGER.info(f"[forward]: Calculate z in share...")
            w_self, w_remote = weights
            # z = self._cal_z_in_share(w_self, w_remote, features, suffix, self.cipher)
            # w_self_value = self.fixedpoint_encoder.decode(w_self.value)
            # mu_self = self.fixedpoint_encoder.decode(features.value).mapValues(lambda x: np.array([np.exp(x.dot(w_self_value))]))
            wx = self._cal_z_in_share(w_self, w_remote, features, suffix, self.cipher)
            # z = self.fixedpoint_encoder.decode(wx).mapValues(lambda x: np.exp(x))
            # mu_self = self.fixedpoint_encoder.decode(features.value).mapValues(lambda x: np.array([np.exp(x.dot(w_self_value))]))
        """
        if not self.reveal_every_iter:
            raise ValueError(f"Hetero SSHE Poisson does not support non reveal_every_iter")

        else:
            LOGGER.info(f"[forward]: Calculate z directly...")
            w = weights.unboxed
            # z = features.dot_local(w)
            # mu_self = self.fixedpoint_encoder.decode(features.value).mapValues(lambda x: np.array([np.exp(x.dot(w))]))
            wx = features.dot_local(w)
        z = self.fixedpoint_encoder.decode(wx.value).mapValues(lambda x: np.exp(np.array(x, dtype=float)))
        z = fixedpoint_table.FixedPointTensor(self.fixedpoint_encoder.encode(z),
                                              q_field=self.fixedpoint_encoder.n,
                                              endec=self.fixedpoint_encoder)
        self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                      is_remote=True,
                                                      cipher=cipher,
                                                      z=z)
        """self.secure_matrix_obj.share_encrypted_matrix(suffix=f"wx.{suffix}",
                                                      is_remote=True,
                                                      cipher=cipher,
                                                      z=wx)"""
        # self.wx_self = wx
        self.wx_self = wx

        tensor_name = ".".join(("complete_z",) + suffix)
        shared_z = SecureMatrix.from_source(tensor_name,
                                            self.other_party,
                                            cipher,
                                            self.fixedpoint_encoder.n,
                                            self.fixedpoint_encoder)

        return shared_z

    def compute_loss(self, weights=None, suffix=None, cipher=None):
        """
         Compute loss:
           loss = 1/N * sum(exp(wx_g)*exp(wx_h) - y(wx_g + wx_h) + log(exposure))
           loss = 1/N * sum(complete_mu - y(complete_wx) + log(exposure)
        """
        LOGGER.info(f"[compute_loss]: Calculate loss ...")

        self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                      is_remote=True,
                                                      cipher=cipher,
                                                      wx_self=self.wx_self)

        tensor_name = ".".join(("shared_loss",) + suffix)
        share_loss = SecureMatrix.from_source(tensor_name=tensor_name,
                                              source=self.other_party,
                                              cipher=cipher,
                                              q_field=self.fixedpoint_encoder.n,
                                              encoder=self.fixedpoint_encoder,
                                              is_fixedpoint_table=False)

        # if self.reveal_every_iter:
        loss_norm = self.optimizer.loss_norm(weights)
        if loss_norm:
            share_loss += loss_norm
        LOGGER.debug(f"share_loss+loss_norm: {share_loss}")
        tensor_name = ".".join(("loss",) + suffix)
        share_loss.broadcast_reconstruct_share(tensor_name=tensor_name)
        """else:
            tensor_name = ".".join(("loss",) + suffix)
            share_loss.broadcast_reconstruct_share(tensor_name=tensor_name)
            if self.optimizer.penalty == consts.L2_PENALTY:
                w_self, w_remote = weights

                w_encode = np.hstack((w_self.value, w_remote.value))

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

                loss_norm = w_tensor.dot(w_tensor_transpose, target_name=loss_norm_tensor_name)
                loss_norm.broadcast_reconstruct_share()"""

    def predict(self, data_instances):
        LOGGER.info("Start predict ...")
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)

        LOGGER.debug(f"Before_predict_reveal_strategy: {self.model_param.reveal_strategy},"
                     f" {self.is_respectively_reveal}")

        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)
        prob_host = data_instances.mapValues(f)
        self.transfer_variable.host_prob.remote(prob_host, role=consts.GUEST, idx=0)
        LOGGER.info("Remote prediction to Guest")

    """def get_single_encrypted_model_weight_dict(self, model_weights=None, header=None):
        weight_dict = {}
        model_weights = model_weights if model_weights else self.model_weights
        header = header if header else self.header
        for idx, header_name in enumerate(header):
            coef_i = model_weights.coef_[idx]

            is_obfuscator = False
            if hasattr(coef_i, "__is_obfuscator"):
                is_obfuscator = getattr(coef_i, "__is_obfuscator")

            public_key = poisson_model_param_pb2.CipherPublicKey(n=str(coef_i.public_key.n))
            weight_dict[header_name] = poisson_model_param_pb2.CipherText(public_key=public_key,
                                                                       cipher_text=str(coef_i.ciphertext()),
                                                                       exponent=str(coef_i.exponent),
                                                                       is_obfuscator=is_obfuscator)
        return weight_dict"""

    def _get_param(self):
        if self.need_cv:
            param_protobuf_obj = poisson_model_param_pb2.PoissonModelParam()
            return param_protobuf_obj

        self.header = self.header if self.header else []
        single_result = self.get_single_model_param()
        param_protobuf_obj = poisson_model_param_pb2.PoissonModelParam(**single_result)
        return param_protobuf_obj

    def _get_meta(self):
        meta_protobuf_obj = poisson_model_meta_pb2.PoissonModelMeta(penalty=self.model_param.penalty,
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
        self.batch_generator = batch_generator.Host()
        self.batch_generator.register_batch_generator(BatchGeneratorTransferVariable(), has_arbiter=False)
        self.header = data_instances.schema.get("header", [])
        self._abnormal_detection(data_instances)
        self.check_abnormal_values(data_instances)
        self.check_abnormal_values(validate_data)

        self.fit_single_model(data_instances, validate_data)
