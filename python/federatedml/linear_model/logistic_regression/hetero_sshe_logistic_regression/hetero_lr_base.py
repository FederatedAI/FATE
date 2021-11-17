#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import copy
from abc import ABC

import numpy as np

from fate_arch.session import get_parties
from federatedml.framework.hetero.procedure import batch_generator
from federatedml.linear_model.linear_model_base import BaseLinearModel
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.one_vs_rest.one_vs_rest import one_vs_rest_factory
from federatedml.param.hetero_sshe_lr_param import LogisticRegressionParam
from federatedml.param.logistic_regression_param import InitParam
from federatedml.protobuf.generated import lr_model_meta_pb2
from federatedml.secureprotol import EncryptModeCalculator
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fixedpoint import FixedPointEndec
from federatedml.secureprotol.spdz import SPDZ
from federatedml.secureprotol.spdz.secure_matrix.secure_matrix import SecureMatrix
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.transfer_variable.transfer_class.batch_generator_transfer_variable import \
    BatchGeneratorTransferVariable
from federatedml.transfer_variable.transfer_class.converge_checker_transfer_variable import \
    ConvergeCheckerTransferVariable
from federatedml.transfer_variable.transfer_class.sshe_model_transfer_variable import SSHEModelTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroLRBase(BaseLinearModel, ABC):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroSSHELogisticRegression'
        self.model_param_name = 'HeteroSSHELogisticRegressionParam'
        self.model_meta_name = 'HeteroSSHELogisticRegressionMeta'
        self.mode = consts.HETERO
        self.cipher = None
        self.q_field = None
        self.model_param = LogisticRegressionParam()
        self.labels = None
        self.batch_num = []
        self.one_vs_rest_obj = None
        self.secure_matrix_obj: SecureMatrix
        self._set_parties()
        self.cipher_tool = None

    def _transfer_q_field(self):
        if self.role == consts.GUEST:
            q_field = self.cipher.public_key.n
            self.transfer_variable.q_field.remote(q_field, role=consts.HOST, suffix=("q_field",))

        else:
            q_field = self.transfer_variable.q_field.get(role=consts.GUEST, idx=0,
                                                          suffix=("q_field",))

        return q_field

    def _init_model(self, params: LogisticRegressionParam):
        super()._init_model(params)
        self.encrypted_mode_calculator_param = params.encrypted_mode_calculator_param
        if self.role == consts.HOST:
            self.init_param_obj.fit_intercept = False
        self.cipher = PaillierEncrypt()
        self.cipher.generate_key(self.model_param.encrypt_param.key_length)
        self.transfer_variable = SSHEModelTransferVariable()
        self.one_vs_rest_obj = one_vs_rest_factory(self, role=self.role, mode=self.mode, has_arbiter=False)

        self.converge_func_name = params.early_stop
        self.reveal_every_iter = params.reveal_every_iter

        self.q_field = self._transfer_q_field()

        LOGGER.debug(f"q_field: {self.q_field}")

        if not self.reveal_every_iter:
            self.self_optimizer = copy.deepcopy(self.optimizer)
            self.remote_optimizer = copy.deepcopy(self.optimizer)

        self.batch_generator = batch_generator.Guest() if self.role == consts.GUEST else batch_generator.Host()
        self.batch_generator.register_batch_generator(BatchGeneratorTransferVariable(), has_arbiter=False)
        self.fixedpoint_encoder = FixedPointEndec(n=self.q_field)
        self.converge_transfer_variable = ConvergeCheckerTransferVariable()
        self.secure_matrix_obj = SecureMatrix(party=self.local_party,
                                              q_field=self.q_field,
                                              other_party=self.other_party)

    def _init_weights(self, model_shape):
        return self.initializer.init_model(model_shape, init_params=self.init_param_obj)

    def _set_parties(self):
        parties = []
        guest_parties = get_parties().roles_to_parties(["guest"])
        host_parties = get_parties().roles_to_parties(["host"])
        parties.extend(guest_parties)
        parties.extend(host_parties)

        local_party = get_parties().local_party
        other_party = parties[0] if parties[0] != local_party else parties[1]

        self.parties = parties
        self.local_party = local_party
        self.other_party = other_party

    @property
    def is_respectively_reveal(self):
        return self.model_param.reveal_strategy == "respectively"

    def share_model(self, w, suffix):
        source = [w, self.other_party]
        if self.local_party.role == consts.GUEST:
            wb, wa = (
                fixedpoint_numpy.FixedPointTensor.from_source(f"wb_{suffix}", source[0],
                                                              encoder=self.fixedpoint_encoder,
                                                              q_field=self.q_field),
                fixedpoint_numpy.FixedPointTensor.from_source(f"wa_{suffix}", source[1],
                                                              encoder=self.fixedpoint_encoder,
                                                              q_field=self.q_field),
            )
            return wb, wa
        else:
            wa, wb = (
                fixedpoint_numpy.FixedPointTensor.from_source(f"wa_{suffix}", source[0],
                                                              encoder=self.fixedpoint_encoder,
                                                              q_field=self.q_field),
                fixedpoint_numpy.FixedPointTensor.from_source(f"wb_{suffix}", source[1],
                                                              encoder=self.fixedpoint_encoder,
                                                              q_field=self.q_field),
            )
            return wa, wb

    def forward(self, weights, features, suffix, cipher):
        raise NotImplementedError("Should not call here")

    def backward(self, error, features, suffix, cipher):
        raise NotImplementedError("Should not call here")

    def compute_loss(self, weights, suffix, cipher):
        raise NotImplementedError("Should not call here")

    def fit(self, data_instances, validate_data=None):
        self.header = data_instances.schema.get("header", [])
        self._abnormal_detection(data_instances)
        self.check_abnormal_values(data_instances)
        self.check_abnormal_values(validate_data)
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
        LOGGER.info("Starting to hetero_sshe_logistic_regression")
        self.callback_list.on_train_begin(data_instances, validate_data)

        model_shape = self.get_features_shape(data_instances)
        instances_count = data_instances.count()

        if not self.component_properties.is_warm_start:
            w = self._init_weights(model_shape)
            self.model_weights = LinearModelWeights(l=w,
                                                    fit_intercept=self.model_param.init_param.fit_intercept)
            last_models = copy.deepcopy(self.model_weights)
        else:
            last_models = copy.deepcopy(self.model_weights)
            w = last_models.unboxed
            self.callback_warm_start_init_iter(self.n_iter_)

        self.batch_generator.initialize_batch_generator(data_instances, batch_size=self.batch_size)

        with SPDZ(
                "sshe_lr",
                local_party=self.local_party,
                all_parties=self.parties,
                q_field=self.q_field,
                use_mix_rand=self.model_param.use_mix_rand,
        ) as spdz:
            spdz.set_flowid(self.flowid)
            self.secure_matrix_obj.set_flowid(self.flowid)
            if self.role == consts.GUEST:
                self.labels = data_instances.mapValues(lambda x: np.array([x.label], dtype=int))

            w_self, w_remote = self.share_model(w, suffix="init")
            last_w_self, last_w_remote = w_self, w_remote
            LOGGER.debug(f"first_w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

            batch_data_generator = self.batch_generator.generate_batch_data()

            self.cipher_tool = []
            encoded_batch_data = []
            for batch_data in batch_data_generator:
                if self.fit_intercept:
                    batch_features = batch_data.mapValues(lambda x: np.hstack((x.features, 1.0)))
                else:
                    batch_features = batch_data.mapValues(lambda x: x.features)
                self.batch_num.append(batch_data.count())

                encoded_batch_data.append(
                    fixedpoint_table.FixedPointTensor(self.fixedpoint_encoder.encode(batch_features),
                                                      q_field=self.fixedpoint_encoder.n,
                                                      endec=self.fixedpoint_encoder))

                self.cipher_tool.append(EncryptModeCalculator(self.cipher,
                                                              self.encrypted_mode_calculator_param.mode,
                                                              self.encrypted_mode_calculator_param.re_encrypted_rate))

            while self.n_iter_ < self.max_iter:
                self.callback_list.on_epoch_begin(self.n_iter_)
                LOGGER.info(f"start to n_iter: {self.n_iter_}")

                loss_list = []

                self.optimizer.set_iters(self.n_iter_)
                if not self.reveal_every_iter:
                    self.self_optimizer.set_iters(self.n_iter_)
                    self.remote_optimizer.set_iters(self.n_iter_)

                for batch_idx, batch_data in enumerate(encoded_batch_data):
                    current_suffix = (str(self.n_iter_), str(batch_idx))

                    if self.reveal_every_iter:
                        y = self.forward(weights=self.model_weights,
                                         features=batch_data,
                                         suffix=current_suffix,
                                         cipher=self.cipher_tool[batch_idx])
                    else:
                        y = self.forward(weights=(w_self, w_remote),
                                         features=batch_data,
                                         suffix=current_suffix,
                                         cipher=self.cipher_tool[batch_idx])

                    if self.role == consts.GUEST:
                        error = y - self.labels

                        self_g, remote_g = self.backward(error=error,
                                                         features=batch_data,
                                                         suffix=current_suffix,
                                                         cipher=self.cipher_tool[batch_idx])
                    else:
                        self_g, remote_g = self.backward(error=y,
                                                         features=batch_data,
                                                         suffix=current_suffix,
                                                         cipher=self.cipher_tool[batch_idx])

                    # loss computing;
                    suffix = ("loss",) + current_suffix
                    if self.reveal_every_iter:
                        batch_loss = self.compute_loss(weights=self.model_weights, suffix=suffix, cipher=self.cipher_tool[batch_idx])
                    else:
                        batch_loss = self.compute_loss(weights=(w_self, w_remote), suffix=suffix, cipher=self.cipher_tool[batch_idx])

                    if batch_loss is not None:
                        batch_loss = batch_loss * self.batch_num[batch_idx]
                    loss_list.append(batch_loss)

                    if self.reveal_every_iter:
                        # LOGGER.debug(f"before reveal: self_g shape: {self_g.shape}, remote_g_shape: {remote_g}，"
                        #              f"self_g: {self_g}")

                        new_g = self.reveal_models(self_g, remote_g, suffix=current_suffix)

                        # LOGGER.debug(f"after reveal: new_g shape: {new_g.shape}, new_g: {new_g}"
                        #              f"self.model_param.reveal_strategy: {self.model_param.reveal_strategy}")

                        if new_g is not None:
                            self.model_weights = self.optimizer.update_model(self.model_weights, new_g,
                                                                             has_applied=False)

                        else:
                            self.model_weights = LinearModelWeights(
                                l=np.zeros(self_g.shape),
                                fit_intercept=self.model_param.init_param.fit_intercept)
                    else:
                        if self.optimizer.penalty == consts.L2_PENALTY:
                            self_g = self_g + self.self_optimizer.alpha * w_self
                            remote_g = remote_g + self.remote_optimizer.alpha * w_remote

                        # LOGGER.debug(f"before optimizer: {self_g}, {remote_g}")

                        self_g = self.self_optimizer.apply_gradients(self_g)
                        remote_g = self.remote_optimizer.apply_gradients(remote_g)

                        # LOGGER.debug(f"after optimizer: {self_g}, {remote_g}")
                        w_self -= self_g
                        w_remote -= remote_g

                    LOGGER.debug(f"w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

                if self.role == consts.GUEST:
                    loss = np.sum(loss_list) / instances_count
                    self.loss_history.append(loss)
                    if self.need_call_back_loss:
                        self.callback_loss(self.n_iter_, loss)
                else:
                    loss = None

                if self.converge_func_name in ["diff", "abs"]:
                    self.is_converged = self.check_converge_by_loss(loss, suffix=(str(self.n_iter_),))
                elif self.converge_func_name == "weight_diff":
                    if self.reveal_every_iter:
                        self.is_converged = self.check_converge_by_weights(
                            last_w=last_models.unboxed,
                            new_w=self.model_weights.unboxed,
                            suffix=(str(self.n_iter_),))
                        last_models = copy.deepcopy(self.model_weights)
                    else:
                        self.is_converged = self.check_converge_by_weights(
                            last_w=(last_w_self, last_w_remote),
                            new_w=(w_self, w_remote),
                            suffix=(str(self.n_iter_),))
                        last_w_self, last_w_remote = copy.deepcopy(w_self), copy.deepcopy(w_remote)
                else:
                    raise ValueError(f"Cannot recognize early_stop function: {self.converge_func_name}")

                LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))
                self.callback_list.on_epoch_end(self.n_iter_)
                self.n_iter_ += 1

                if self.stop_training:
                    break

                if self.is_converged:
                    break

            # Finally reconstruct
            if not self.reveal_every_iter:
                new_w = self.reveal_models(w_self, w_remote, suffix=("final",))
                if new_w is not None:
                    self.model_weights = LinearModelWeights(
                        l=new_w,
                        fit_intercept=self.model_param.init_param.fit_intercept)

        LOGGER.debug(f"loss_history: {self.loss_history}")
        self.set_summary(self.get_model_summary())

    def reveal_models(self, w_self, w_remote, suffix=None):
        if suffix is None:
            suffix = self.n_iter_

        if self.model_param.reveal_strategy == "respectively":

            if self.role == consts.GUEST:
                new_w = w_self.get(tensor_name=f"wb_{suffix}",
                                   broadcast=False)
                w_remote.broadcast_reconstruct_share(tensor_name=f"wa_{suffix}")

            else:
                w_remote.broadcast_reconstruct_share(tensor_name=f"wb_{suffix}")
                new_w = w_self.get(tensor_name=f"wa_{suffix}",
                                   broadcast=False)

        elif self.model_param.reveal_strategy == "encrypted_reveal_in_host":

            if self.role == consts.GUEST:
                new_w = w_self.get(tensor_name=f"wb_{suffix}",
                                   broadcast=False)
                encrypted_w_remote = self.cipher.recursive_encrypt(self.fixedpoint_encoder.decode(w_remote.value))
                encrypted_w_remote_tensor = fixedpoint_numpy.PaillierFixedPointTensor(value=encrypted_w_remote)
                encrypted_w_remote_tensor.broadcast_reconstruct_share(tensor_name=f"wa_{suffix}")
            else:
                w_remote.broadcast_reconstruct_share(tensor_name=f"wb_{suffix}")

                new_w = w_self.reconstruct(tensor_name=f"wa_{suffix}", broadcast=False)

        else:
            raise NotImplementedError(f"reveal strategy: {self.model_param.reveal_strategy} has not been implemented.")
        return new_w

    def check_converge_by_loss(self, loss, suffix):
        if self.role == consts.GUEST:
            self.is_converged = self.converge_func.is_converge(loss)
            self.transfer_variable.is_converged.remote(self.is_converged, suffix=suffix)
        else:
            self.is_converged = self.transfer_variable.is_converged.get(idx=0, suffix=suffix)
        return self.is_converged

    def check_converge_by_weights(self, last_w, new_w, suffix):
        if self.reveal_every_iter:
            return self._reveal_every_iter_weights_check(last_w, new_w, suffix)
        else:
            return self._not_reveal_every_iter_weights_check(last_w, new_w, suffix)

    def _reveal_every_iter_weights_check(self, last_w, new_w, suffix):
        raise NotImplementedError()

    def _not_reveal_every_iter_weights_check(self, last_w, new_w, suffix):
        last_w_self, last_w_remote = last_w
        w_self, w_remote = new_w
        grad_self = w_self - last_w_self
        grad_remote = w_remote - last_w_remote

        if self.role == consts.GUEST:
            grad_encode = np.hstack((grad_remote.value, grad_self.value))
        else:
            grad_encode = np.hstack((grad_self.value, grad_remote.value))

        grad_encode = np.array([grad_encode])

        grad_tensor_name = ".".join(("check_converge_grad",) + suffix)
        grad_tensor = fixedpoint_numpy.FixedPointTensor(value=grad_encode,
                                                        q_field=self.fixedpoint_encoder.n,
                                                        endec=self.fixedpoint_encoder,
                                                        tensor_name=grad_tensor_name)

        grad_tensor_transpose_name = ".".join(("check_converge_grad_transpose",) + suffix)
        grad_tensor_transpose = fixedpoint_numpy.FixedPointTensor(value=grad_encode.T,
                                                                  q_field=self.fixedpoint_encoder.n,
                                                                  endec=self.fixedpoint_encoder,
                                                                  tensor_name=grad_tensor_transpose_name)

        grad_norm_tensor_name = ".".join(("check_converge_grad_norm",) + suffix)

        grad_norm = grad_tensor.dot(grad_tensor_transpose, target_name=grad_norm_tensor_name).get()

        weight_diff = np.sqrt(grad_norm[0][0])
        LOGGER.info("iter: {}, weight_diff:{}, is_converged: {}".format(self.n_iter_,
                                                                        weight_diff, self.is_converged))
        is_converge = False
        if weight_diff < self.model_param.tol:
            is_converge = True
        return is_converge

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

    def get_single_model_param(self, model_weights=None, header=None):
        header = header if header else self.header
        result = {'iters': self.n_iter_,
                  'loss_history': self.loss_history,
                  'is_converged': self.is_converged,
                  # 'weight': weight_dict,
                  'intercept': self.model_weights.intercept_,
                  'header': header,
                  'best_iteration': -1 if self.validation_strategy is None else
                  self.validation_strategy.best_iteration
                  }

        if self.role == consts.GUEST or self.is_respectively_reveal:
            model_weights = model_weights if model_weights else self.model_weights
            weight_dict = {}
            for idx, header_name in enumerate(header):
                coef_i = model_weights.coef_[idx]
                weight_dict[header_name] = coef_i

            result['weight'] = weight_dict

        return result

    def get_model_summary(self):
        header = self.header
        if header is None:
            return {}
        weight_dict, intercept_ = self.get_weight_intercept_dict(header)
        best_iteration = -1 if self.validation_strategy is None else self.validation_strategy.best_iteration

        summary = {"coef": weight_dict,
                   "intercept": intercept_,
                   "is_converged": self.is_converged,
                   "one_vs_rest": self.need_one_vs_rest,
                   "best_iteration": best_iteration}

        if not self.is_respectively_reveal:
            del summary["intercept"]
            del summary["coef"]

        if self.validation_strategy:
            validation_summary = self.validation_strategy.summary()
            if validation_summary:
                summary["validation_metrics"] = validation_summary
        return summary

    def load_model(self, model_dict):
        LOGGER.debug("Start Loading model")
        result_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)

        if self.init_param_obj is None:
            self.init_param_obj = InitParam()
        self.init_param_obj.fit_intercept = meta_obj.fit_intercept
        self.model_param.reveal_strategy = meta_obj.reveal_strategy
        LOGGER.debug(f"reveal_strategy: {self.model_param.reveal_strategy}, {self.is_respectively_reveal}")
        self.header = list(result_obj.header)

        need_one_vs_rest = result_obj.need_one_vs_rest
        LOGGER.info("in _load_model need_one_vs_rest: {}".format(need_one_vs_rest))
        if need_one_vs_rest:
            one_vs_rest_result = result_obj.one_vs_rest_result
            self.one_vs_rest_obj = one_vs_rest_factory(classifier=self, role=self.role,
                                                       mode=self.mode, has_arbiter=False)
            self.one_vs_rest_obj.load_model(one_vs_rest_result)
            self.need_one_vs_rest = True
        else:
            self.load_single_model(result_obj)
            self.need_one_vs_rest = False

    def load_single_model(self, single_model_obj):
        LOGGER.info("It's a binary task, start to load single model")

        if self.role == consts.GUEST or self.is_respectively_reveal:
            feature_shape = len(self.header)
            tmp_vars = np.zeros(feature_shape)
            weight_dict = dict(single_model_obj.weight)

            for idx, header_name in enumerate(self.header):
                tmp_vars[idx] = weight_dict.get(header_name)

            if self.fit_intercept:
                tmp_vars = np.append(tmp_vars, single_model_obj.intercept)
            self.model_weights = LinearModelWeights(tmp_vars, fit_intercept=self.fit_intercept)

        self.n_iter_ = single_model_obj.iters
        return self




