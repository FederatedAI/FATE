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
import operator
from abc import ABC

import numpy as np

from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.linear_model.sshe_model.sshe_model_base import SSHEModelBase
from federatedml.one_vs_rest.one_vs_rest import one_vs_rest_factory
from federatedml.param.hetero_sshe_lr_param import LogisticRegressionParam
from federatedml.param.logistic_regression_param import InitParam
from federatedml.protobuf.generated import lr_model_meta_pb2
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.spdz import SPDZ
from federatedml.secureprotol.spdz.tensor import fixedpoint_numpy, fixedpoint_table
from federatedml.transfer_variable.transfer_class.sshe_model_transfer_variable import SSHEModelTransferVariable
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroLRBase(SSHEModelBase, ABC):
    def __init__(self):
        super().__init__()
        self.model_name = 'HeteroLogisticRegression'
        self.model_param_name = 'HeteroLogisticRegressionParam'
        self.model_meta_name = 'HeteroLogisticRegressionMeta'
        self.mode = consts.HETERO
        self.cipher = None
        self.gradient_loss_operator = None
        self.converge_procedure = None
        self.model_param = LogisticRegressionParam()
        # self.features = None
        self.labels = None
        self.label_tensor = None
        self.host_model_weights = None
        self.encoded_batch_num = []
        self.one_vs_rest_obj = None
        self.shared_y = None

    def _init_model(self, params: LogisticRegressionParam):
        super()._init_model(params)
        self.cipher = PaillierEncrypt()
        self.cipher.generate_key(self.model_param.encrypt_param.key_length)
        self.transfer_variable = SSHEModelTransferVariable()
        self.one_vs_rest_obj = one_vs_rest_factory(self, role=self.role, mode=self.mode, has_arbiter=False)
        self.cal_loss = self.model_param.compute_loss
        self.converge_func_name = params.early_stop
        self.review_every_iter = params.reveal_every_iter

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

        if self.validation_strategy:
            validation_summary = self.validation_strategy.summary()
            if validation_summary:
                summary["validation_metrics"] = validation_summary
        return summary

    @property
    def is_respectively_reveal(self):
        return self.model_param.reveal_strategy == "respectively"

    def share_table(self, fix_point_encoder, value=None, tensor_name=''):
        if value is None:
            value = self.other_party

        return fixedpoint_table.FixedPointTensor.from_source(tensor_name, value,
                                                             encoder=fix_point_encoder,
                                                             q_field=self.random_field)

    def share_model(self, w, fix_point_encoder, suffix):
        source = [w, self.other_party]
        if self.local_party.role == consts.GUEST:
            wb, wa = (
                fixedpoint_numpy.FixedPointTensor.from_source(f"wb_{suffix}", source[0],
                                                              encoder=fix_point_encoder,
                                                              q_field=self.random_field),
                fixedpoint_numpy.FixedPointTensor.from_source(f"wa_{suffix}", source[1],
                                                              encoder=fix_point_encoder,
                                                              q_field=self.random_field),
            )
            return wb, wa
        else:
            wa, wb = (
                fixedpoint_numpy.FixedPointTensor.from_source(f"wa_{suffix}", source[0],
                                                              encoder=fix_point_encoder,
                                                              q_field=self.random_field),
                fixedpoint_numpy.FixedPointTensor.from_source(f"wb_{suffix}", source[1],
                                                              encoder=fix_point_encoder,
                                                              q_field=self.random_field),
            )
            return wa, wb

    def cal_prediction(self, w_self, w_remote, features, spdz, suffix):
        raise NotImplementedError("Should not call here")

    def compute_gradient(self, wa, wb, error, features, suffix):
        raise NotImplementedError("Should not call here")

    def transfer_pubkey(self):
        raise NotImplementedError("Should not call here")

    def compute_loss(self, spdz, suffix):
        raise NotImplementedError("Should not call here")

    def fit(self, data_instances, validate_data=None):
        self.header = data_instances.schema.get("header", [])
        self._abnormal_detection(data_instances)
        self.check_abnormal_values(data_instances)
        self.check_abnormal_values(validate_data)
        classes = self.one_vs_rest_obj.get_data_classes(data_instances)

        # self.fit_binary(data_instances, validate_data)

        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.need_call_back_loss = False
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            self.fit_binary(data_instances, validate_data)

    def _init_weights(self, model_shape):
        return self.initializer.init_model(model_shape, init_params=self.init_param_obj)

    def check_converge_by_loss(self, loss, suffix):
        if self.role == consts.GUEST:
            self.is_converged = self.converge_func.is_converge(loss)
            self.transfer_variable.is_converged.remote(self.is_converged, suffix=suffix)
        else:
            self.is_converged = self.transfer_variable.is_converged.get(idx=0, suffix=suffix)
        return self.is_converged

    def fit_binary(self, data_instances, validate_data=None):
        LOGGER.info("Start to hetero_sshe_logistic_regression")
        self.callback_list.on_train_begin(data_instances, validate_data)

        # self.validation_strategy = self.init_validation_strategy(data_instances, validate_data)
        # self.batch_generator.initialize_batch_generator(data_instances, self.batch_size)
        model_shape = self.get_features_shape(data_instances)
        if not self.component_properties.is_warm_start:
            w = self._init_weights(model_shape)
            last_models = w
            self.model_weights = LinearModelWeights(l=w,
                                                    fit_intercept=self.model_param.init_param.fit_intercept)
        else:
            last_models = self.model_weights.unboxed
            w = last_models
            self.callback_warm_start_init_iter(self.n_iter_)
            self.n_iter_ += 1

        self.batch_generator.initialize_batch_generator(data_instances, batch_size=self.batch_size)

        remote_pubkey = self.transfer_pubkey()
        with SPDZ(
                "sshe_lr",
                local_party=self.local_party,
                all_parties=self.parties,
                q_field=self.random_field,
                use_mix_rand=self.model_param.use_mix_rand,
        ) as spdz:
            self.fixpoint_encoder = self.create_fixpoint_encoder(remote_pubkey.n)
            if self.role == consts.GUEST:
                self.labels = data_instances.mapValues(lambda x: np.array([x.label], dtype=int))
                self.label_tensor = fixedpoint_table.FixedPointTensor.from_value(self.labels,
                                                                                 q_field=self.fixpoint_encoder.n,
                                                                                 encoder=self.fixpoint_encoder)
            if self.cal_loss:
                value = self.label_tensor.value if self.role == consts.GUEST else None
                self.shared_y = self.share_table(self.fixpoint_encoder, value=value, tensor_name="label")
                LOGGER.debug(f"shared_y: {self.shared_y}, type: {type(self.shared_y)}")

            w_self, w_remote = self.share_model(w, self.fixpoint_encoder, suffix="init")
            LOGGER.debug(f"first_w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

            batch_data_generator = self.batch_generator.generate_batch_data()
            encoded_batch_data = []
            for batch_data in batch_data_generator:
                batch_features = batch_data.mapValues(lambda x: x.features)
                self.encoded_batch_num.append(self.fixpoint_encoder.encode(1 / batch_data.count()))
                encoded_batch_data.append(
                    fixedpoint_table.FixedPointTensor(self.fixpoint_encoder.encode(batch_features),
                                                      q_field=self.fixpoint_encoder.n,
                                                      endec=self.fixpoint_encoder))

            while self.n_iter_ < self.max_iter:
                self.callback_list.on_epoch_begin(self.n_iter_)

                loss_list = []
                self.optimizer.set_iters(self.n_iter_)
                for batch_idx, batch_data in enumerate(encoded_batch_data):

                    LOGGER.debug(f"n_iter: {self.n_iter_}")
                    current_suffix = (self.n_iter_, batch_idx)
                    y = self.cal_prediction(w_self, w_remote, features=batch_data, spdz=spdz, suffix=current_suffix)

                    if self.role == consts.GUEST:
                        error = y.value.join(self.labels, operator.sub)
                        error = fixedpoint_table.FixedPointTensor.from_value(error,
                                                                             q_field=self.fixpoint_encoder.n,
                                                                             encoder=self.fixpoint_encoder)
                        remote_g, self_g = self.compute_gradient(wa=w_remote, wb=w_self, error=error,
                                                                 features=batch_data,
                                                                 suffix=current_suffix)
                    else:
                        self_g, remote_g = self.compute_gradient(wa=w_self, wb=w_remote, error=y,
                                                                 features=batch_data, suffix=current_suffix)

                    if self.review_every_iter:
                        LOGGER.debug(f"self_g shape: {self_g.shape}, remote_g_shape: {remote_g}")

                        new_g, host_g = self.review_models(self_g, remote_g, suffix=(self.n_iter_, batch_idx))
                        LOGGER.debug(f"new_g shape: {new_g.shape}, host_g_shape: {host_g}")

                        if new_g is not None:
                            self.model_weights = self.optimizer.update_model(self.model_weights, new_g,
                                                                             has_applied=False)
                        else:
                            self.model_weights = LinearModelWeights(
                                l=np.zeros(self_g.shape),
                                fit_intercept=self.model_param.init_param.fit_intercept)
                        if host_g is not None:
                            self.host_model_weights = LinearModelWeights(
                                l=host_g,
                                fit_intercept=False)
                        w_self, w_remote = self.share_model(self.model_weights.unboxed, self.fixpoint_encoder,
                                                            suffix=(self.n_iter_, batch_idx))
                    else:
                        w_self -= self_g * self.optimizer.decay_learning_rate()
                        w_remote -= remote_g * self.optimizer.decay_learning_rate()

                    LOGGER.debug(f"other_w_self shape: {w_self.shape}, w_remote_shape: {w_remote.shape}")

                    if self.cal_loss:
                        suffix = ("loss",) + current_suffix
                        loss_list.append(self.compute_loss(spdz=spdz, suffix=suffix))

                if self.cal_loss:
                    if self.role == consts.GUEST:
                        loss = np.sum(loss_list) / self.batch_generator.batch_nums
                        self.loss_history.append(loss)
                        self.callback_loss(self.n_iter_, loss)
                    else:
                        loss = None

                if self.converge_func_name in ["diff", "abs"]:
                    self.is_converged = self.check_converge_by_loss(loss, suffix=(self.n_iter_,))
                elif self.converge_func_name == "weight_diff":
                    self.is_converged = self.check_converge_by_weights(
                        last_w=last_models,
                        new_w=self.model_weights.unboxed,
                        suffix=(self.n_iter_,))
                    last_models = copy.deepcopy(self.model_weights.unboxed)
                else:
                    raise ValueError(f"Cannot recognize early_stop function: {self.converge_func_name}")

                LOGGER.info("iter: {},  is_converged: {}".format(self.n_iter_, self.is_converged))
                self.callback_list.on_epoch_end(self.n_iter_)
                if self.stop_training:
                    break

                if self.is_converged:
                    break
                self.n_iter_ += 1

            # Finally reconstruct
            if not self.review_every_iter:
                new_w, host_weights = self.review_models(w_self, w_remote, suffix=("final",))
                if new_w is not None:
                    self.model_weights = LinearModelWeights(
                        l=new_w,
                        fit_intercept=self.model_param.init_param.fit_intercept)

                if host_weights is not None:
                    self.host_model_weights = LinearModelWeights(
                        l=host_weights,
                        fit_intercept=False)

        LOGGER.debug(f"loss_history: {self.loss_history}")
        self.set_summary(self.get_model_summary())

    def review_models(self, w_self, w_remote, suffix=None):
        if suffix is None:
             suffix = self.n_iter_
        host_weights = None
        if self.model_param.reveal_strategy == "respectively":
            if self.role == consts.GUEST:
                new_w = w_self.reconstruct_unilateral(tensor_name=f"wb_{suffix}")
                w_remote.broadcast_reconstruct_share(tensor_name=f"wa_{suffix}")
            else:
                w_remote.broadcast_reconstruct_share(tensor_name=f"wb_{suffix}")
                new_w = w_self.reconstruct_unilateral(tensor_name=f"wa_{suffix}")

        elif self.model_param.reveal_strategy == "all_reveal_in_guest":

            if self.role == consts.GUEST:
                new_w = w_self.reconstruct_unilateral(tensor_name=f"wb_{suffix}")
                host_weights = w_remote.reconstruct_unilateral(tensor_name=f"wa_{suffix}")
                # self.host_model_weights = [LinearModelWeights(l=hosted_weights, fit_intercept=False)]
            else:
                if w_remote.shape[0] > 2:
                    raise ValueError("Too many features in Guest. Review strategy: 'all_reveal_in_guest' "
                                     "should not be used.")
                w_remote.broadcast_reconstruct_share(tensor_name=f"wb_{suffix}")

                w_self.broadcast_reconstruct_share(tensor_name=f"wa_{suffix}")
                # new_w = np.zeros(w_self.shape)
                new_w = None
        else:
            raise NotImplementedError(f"review strategy: {self.model_param.reveal_strategy} has not been implemented.")
        return new_w, host_weights

    def share_encrypted_value(self, suffix, is_remote, **kwargs):
        if is_remote:
            for var_name, var in kwargs.items():
                dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST
                encrypt_var = self.cipher.distribute_encrypt(var.value)
                self.transfer_variable.encrypted_share_matrix.remote(encrypt_var, role=dest_role,
                                                                     suffix=(var_name,) + suffix)
        else:
            res = []
            for var_name in kwargs.keys():
                dest_role = consts.GUEST if self.role == consts.HOST else consts.HOST
                z_table = self.transfer_variable.encrypted_share_matrix.get(role=dest_role, idx=0,
                                                                            suffix=(var_name,) + suffix)
                res.append(fixedpoint_table.PaillierFixedPointTensor(
                    z_table, q_field=self.fixpoint_encoder.n, endec=self.fixpoint_encoder))
            return tuple(res)

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
        weight_dict = {}
        model_weights = model_weights if model_weights else self.model_weights
        header = header if header else self.header
        for idx, header_name in enumerate(header):
            coef_i = model_weights.coef_[idx]
            weight_dict[header_name] = coef_i

        result = {'iters': self.n_iter_,
                  'loss_history': self.loss_history,
                  'is_converged': self.is_converged,
                  'weight': weight_dict,
                  'intercept': self.model_weights.intercept_,
                  'header': header,
                  'best_iteration': -1 if self.validation_strategy is None else
                  self.validation_strategy.best_iteration
                  }
        return result

    def load_model(self, model_dict):
        LOGGER.debug("Start Loading model")
        result_obj = list(model_dict.get('model').values())[0].get(self.model_param_name)
        meta_obj = list(model_dict.get('model').values())[0].get(self.model_meta_name)
        # self.fit_intercept = meta_obj.fit_intercept
        if self.init_param_obj is None:
            self.init_param_obj = InitParam()
        self.init_param_obj.fit_intercept = meta_obj.fit_intercept
        self.model_param.reveal_strategy = meta_obj.reveal_strategy
        LOGGER.debug(f"review_strategy: {self.model_param.reveal_strategy}, {self.is_respectively_reveal}")
        self.header = list(result_obj.header)
        # For hetero-lr arbiter predict function
        need_one_vs_rest = result_obj.need_one_vs_rest
        LOGGER.debug("in _load_model need_one_vs_rest: {}".format(need_one_vs_rest))
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

    def one_vs_rest_fit(self, train_data=None, validate_data=None):
        LOGGER.debug("Class num larger than 2, need to do one_vs_rest")
        self.one_vs_rest_obj.fit(data_instances=train_data, validate_data=validate_data)
