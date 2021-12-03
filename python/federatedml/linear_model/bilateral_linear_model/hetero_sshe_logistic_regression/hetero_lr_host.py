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

from federatedml.framework.hetero.procedure.hetero_sshe_linear_model import HeteroSSHEHostBase
from federatedml.one_vs_rest.one_vs_rest import one_vs_rest_factory
from federatedml.param.hetero_sshe_lr_param import HeteroSSHELRParam
from federatedml.protobuf.generated import lr_model_param_pb2, lr_model_meta_pb2, sshe_cipher_param_pb2
from federatedml.secureprotol.spdz.secure_matrix.secure_matrix import SecureMatrix
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy
from federatedml.util import consts, fate_operator, LOGGER


class HeteroLRHost(HeteroSSHEHostBase):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroLogisticRegression'
        self.model_param_name = 'HeteroLogisticRegressionParam'
        self.model_meta_name = 'HeteroLogisticRegressionMeta'
        self.model_param = HeteroSSHELRParam()
        self.labels = None
        self.one_vs_rest_obj = None

    def _init_model(self, params):
        super()._init_model(params)
        self.one_vs_rest_obj = one_vs_rest_factory(self, role=self.role, mode=self.mode, has_arbiter=False)

    """def _cal_z_in_share(self, w_self, w_remote, features, suffix):
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
    """
    def forward(self, weights, features, suffix, cipher):
        if not self.reveal_every_iter:
            LOGGER.info(f"[forward]: Calculate z in share...")
            w_self, w_remote = weights
            z = self._cal_z_in_share(w_self, w_remote, features, suffix, self.cipher)
        else:
            LOGGER.info(f"[forward]: Calculate z directly...")
            w = weights.unboxed
            z = features.dot_local(w)

        self.wx_self = z

        self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                      is_remote=True,
                                                      cipher=cipher,
                                                      z=z)

        tensor_name = ".".join(("sigmoid_z",) + suffix)
        shared_sigmoid_z = SecureMatrix.from_source(tensor_name,
                                                    self.other_party,
                                                    cipher,
                                                    self.fixedpoint_encoder.n,
                                                    self.fixedpoint_encoder)

        return shared_sigmoid_z

    """
    def backward(self, error: fixedpoint_table.FixedPointTensor, features, suffix, cipher):
        LOGGER.info(f"[backward]: Calculate gradient...")
        batch_num = self.batch_num[int(suffix[1])]

        ga = features.dot_local(error)
        # LOGGER.debug(f"ga: {ga}, batch_num: {batch_num}")
        ga = ga * (1 / batch_num)

        zb_suffix = ("ga2",) + suffix
        ga2_1 = self.secure_matrix_obj.secure_matrix_mul(features,
                                                         tensor_name=".".join(zb_suffix),
                                                         cipher=None,
                                                         suffix=zb_suffix)

        # LOGGER.debug(f"ga2_1: {ga2_1}")

        ga_new = ga + ga2_1

        tensor_name = ".".join(("encrypt_g",) + suffix)
        gb1 = SecureMatrix.from_source(tensor_name,
                                       self.other_party,
                                       cipher,
                                       self.fixedpoint_encoder.n,
                                       self.fixedpoint_encoder,
                                       is_fixedpoint_table=False)

        # LOGGER.debug(f"gb1: {gb1}")

        return ga_new, gb1
    """

    def compute_loss(self, weights=None, suffix=None, cipher=None):
        """
          Use Taylor series expand log loss:
          Loss = - y * log(h(x)) - (1-y) * log(1 - h(x)) where h(x) = 1/(1+exp(-wx))
          Then loss' = - (1/N)*∑(log(1/2) - 1/2*wx + ywx - 1/8(wx)^2)
        """
        LOGGER.info(f"[compute_loss]: Calculate loss ...")
        wx_self_square = (self.wx_self * self.wx_self).reduce(operator.add)

        self.secure_matrix_obj.share_encrypted_matrix(suffix=suffix,
                                                      is_remote=True,
                                                      cipher=cipher,
                                                      wx_self_square=wx_self_square)

        tensor_name = ".".join(("shared_loss",) + suffix)
        share_loss = SecureMatrix.from_source(tensor_name=tensor_name,
                                              source=self.other_party,
                                              cipher=cipher,
                                              q_field=self.fixedpoint_encoder.n,
                                              encoder=self.fixedpoint_encoder,
                                              is_fixedpoint_table=False)

        if self.reveal_every_iter:
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

    """    
    def _reveal_every_iter_weights_check(self, last_w, new_w, suffix):
        square_sum = np.sum((last_w - new_w) ** 2)
        self.converge_transfer_variable.square_sum.remote(square_sum, role=consts.GUEST, idx=0, suffix=suffix)
        return self.converge_transfer_variable.converge_info.get(idx=0, suffix=suffix)
    """

    def predict(self, data_instances):
        LOGGER.info("Start predict ...")
        self._abnormal_detection(data_instances)
        data_instances = self.align_data_header(data_instances, self.header)
        if self.need_one_vs_rest:
            self.one_vs_rest_obj.predict(data_instances)
            return

        LOGGER.debug(f"Before_predict_reveal_strategy: {self.model_param.reveal_strategy},"
                     f" {self.is_respectively_reveal}")

        def _vec_dot(v, coef, intercept):
            return fate_operator.vec_dot(v.features, coef) + intercept

        f = functools.partial(_vec_dot,
                              coef=self.model_weights.coef_,
                              intercept=self.model_weights.intercept_)
        prob_host = data_instances.mapValues(f)
        self.transfer_variable.host_prob.remote(prob_host, role=consts.GUEST, idx=0)
        LOGGER.info("Remote probability to Guest")

    """
    def get_single_model_param(self, model_weights=None, header=None):
        result = super().get_single_model_param(model_weights, header)
        if not self.is_respectively_reveal:
            weight_dict = {}
            model_weights = model_weights if model_weights else self.model_weights
            header = header if header else self.header
            for idx, header_name in enumerate(header):
                coef_i = model_weights.coef_[idx]

                is_obfuscator = False
                if hasattr(coef_i, "__is_obfuscator"):
                    is_obfuscator = getattr(coef_i, "__is_obfuscator")

                public_key = lr_model_param_pb2.CipherPublicKey(n=str(coef_i.public_key.n))
                weight_dict[header_name] = lr_model_param_pb2.CipherText(public_key=public_key,
                                                                         cipher_text=str(coef_i.ciphertext()),
                                                                         exponent=str(coef_i.exponent),
                                                                         is_obfuscator=is_obfuscator)
            result["encrypted_weight"] = weight_dict

        return result
    """
    def get_single_encrypted_model_weight_dict(self, model_weights=None, header=None):
        weight_dict = {}
        model_weights = model_weights if model_weights else self.model_weights
        header = header if header else self.header
        for idx, header_name in enumerate(header):
            coef_i = model_weights.coef_[idx]

            is_obfuscator = False
            if hasattr(coef_i, "__is_obfuscator"):
                is_obfuscator = getattr(coef_i, "__is_obfuscator")

            public_key = sshe_cipher_param_pb2.CipherPublicKey(n=str(coef_i.public_key.n))
            weight_dict[header_name] = sshe_cipher_param_pb2.CipherText(public_key=public_key,
                                                                        cipher_text=str(coef_i.ciphertext()),
                                                                        exponent=str(coef_i.exponent),
                                                                        is_obfuscator=is_obfuscator)
        return weight_dict

    def _get_param(self):
        if self.need_cv:
            param_protobuf_obj = lr_model_param_pb2.LRModelParam()
            return param_protobuf_obj

        self.header = self.header if self.header else []
        LOGGER.debug("In get_param, self.need_one_vs_rest: {}".format(self.need_one_vs_rest))

        if self.need_one_vs_rest:
            one_vs_rest_result = self.one_vs_rest_obj.save(lr_model_param_pb2.SingleModel)
            single_result = {'header': self.header, 'need_one_vs_rest': True, "best_iteration": -1}
        else:
            one_vs_rest_result = None
            single_result = self.get_single_model_param()
            single_result['need_one_vs_rest'] = False
        single_result['one_vs_rest_result'] = one_vs_rest_result

        param_protobuf_obj = lr_model_param_pb2.LRModelParam(**single_result)

        return param_protobuf_obj

    def _get_meta(self):
        meta_protobuf_obj = lr_model_meta_pb2.LRModelMeta(penalty=self.model_param.penalty,
                                                          tol=self.model_param.tol,
                                                          alpha=self.alpha,
                                                          optimizer=self.model_param.optimizer,
                                                          batch_size=self.batch_size,
                                                          learning_rate=self.model_param.learning_rate,
                                                          max_iter=self.max_iter,
                                                          early_stop=self.model_param.early_stop,
                                                          fit_intercept=self.fit_intercept,
                                                          need_one_vs_rest=self.need_one_vs_rest,
                                                          reveal_strategy=self.model_param.reveal_strategy)
        return meta_protobuf_obj

    """
    def load_single_model(self, single_model_obj):
        super(HeteroLRHost, self).load_single_model(single_model_obj)
        if not self.is_respectively_reveal:
            feature_shape = len(self.header)
            tmp_vars = [None] * feature_shape
            weight_dict = dict(single_model_obj.encrypted_weight)
            for idx, header_name in enumerate(self.header):
                cipher_weight = weight_dict.get(header_name)
                public_key = PaillierPublicKey(int(cipher_weight.public_key.n))
                cipher_text = int(cipher_weight.cipher_text)
                exponent = int(cipher_weight.exponent)
                is_obfuscator = cipher_weight.is_obfuscator
                coef_i = PaillierEncryptedNumber(public_key, cipher_text, exponent)
                if is_obfuscator:
                    coef_i.apply_obfuscator()

                tmp_vars[idx] = coef_i

            self.model_weights = LinearModelWeights(tmp_vars, fit_intercept=self.fit_intercept)
    """
    def load_model(self, model_dict):
        result_obj, _ = super().load_model(model_dict)
        need_one_vs_rest = result_obj.need_one_vs_rest
        LOGGER.info("in _load_model need_one_vs_rest: {}".format(need_one_vs_rest))
        if need_one_vs_rest:
            one_vs_rest_result = result_obj.one_vs_rest_result
            self.one_vs_rest_obj = one_vs_rest_factory(classifier=self, role=consts.HOST,
                                                       mode=self.mode, has_arbiter=False)
            self.one_vs_rest_obj.load_model(one_vs_rest_result)
            self.need_one_vs_rest = True
        else:
            self.load_single_model(result_obj)
            self.need_one_vs_rest = False

    def fit(self, data_instances, validate_data=None):
        LOGGER.info("Starting to fit hetero_sshe_logistic_regression")
        """
        self.batch_generator = batch_generator.Host()
        self.batch_generator.register_batch_generator(BatchGeneratorTransferVariable(), has_arbiter=False)
        self.header = data_instances.schema.get("header", [])
        self._abnormal_detection(data_instances)
        self.check_abnormal_values(data_instances)
        self.check_abnormal_values(validate_data)
        """
        self.prepare_fit(data_instances, validate_data)
        classes = self.one_vs_rest_obj.get_data_classes(data_instances)

        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.need_call_back_loss = False
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    def one_vs_rest_fit(self, train_data=None, validate_data=None):
        LOGGER.info("Class num larger than 2, do one_vs_rest")
        self.one_vs_rest_obj.fit(data_instances=train_data, validate_data=validate_data)

    def fit_binary(self, data_instances, validate_data=None):
        self.fit_single_model(data_instances, validate_data)

    def get_model_summary(self):
        summary = super().get_model_summary()
        summary["one_vs_rest"] = self.need_one_vs_rest
        return summary
