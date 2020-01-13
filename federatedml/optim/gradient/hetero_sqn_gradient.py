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

import numpy as np

from arch.api.utils import log_utils
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.optim.gradient import hetero_linear_model_gradient
from federatedml.optim.gradient import sqn_sync
from federatedml.param.sqn_param import StochasticQuasiNewtonParam
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroStochasticQuansiNewton(hetero_linear_model_gradient.HeteroGradientBase):
    def __init__(self, sqn_param: StochasticQuasiNewtonParam):
        self.gradient_computer = None
        self.transfer_variable = None
        self.sqn_sync = None
        self.n_iter = 0
        self.count_t = -1
        self.__total_batch_nums = 0
        self.batch_index = 0
        self.last_w_tilde: LinearModelWeights = None
        self.this_w_tilde: LinearModelWeights = None
        # self.sqn_param = sqn_param
        self.update_interval_L = sqn_param.update_interval_L
        self.memory_M = sqn_param.memory_M
        self.sample_size = sqn_param.sample_size
        self.random_seed = sqn_param.random_seed

    @property
    def iter_k(self):
        return self.n_iter * self.__total_batch_nums + self.batch_index + 1

    def set_total_batch_nums(self, total_batch_nums):
        self.__total_batch_nums = total_batch_nums

    def register_gradient_computer(self, gradient_computer):
        self.gradient_computer = copy.deepcopy(gradient_computer)

    def register_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable
        self.sqn_sync.register_transfer_variable(self.transfer_variable)

    def _renew_w_tilde(self):
        self.last_w_tilde = self.this_w_tilde
        self.this_w_tilde = LinearModelWeights(np.zeros_like(self.last_w_tilde.unboxed),
                                               self.last_w_tilde.fit_intercept)

    def _update_hessian(self, *args):
        raise NotImplementedError("Should not call here")

    def _update_w_tilde(self, model_weights):
        if self.this_w_tilde is None:
            self.this_w_tilde = copy.deepcopy(model_weights)
        else:
            self.this_w_tilde += model_weights

    def compute_gradient_procedure(self, *args):
        data_instances = args[0]
        encrypted_calculator = args[1]
        model_weights = args[2]
        optimizer = args[3]
        self.batch_index = args[5]
        self.n_iter = args[4]
        cipher_operator = encrypted_calculator[0].encrypter
        # one_data = data_instances.first()
        # LOGGER.debug("data shape: {}, model weights shape: {}, model weights coef: {}, intercept: {}".format(
        #     one_data[1].features.shape, model_weights.unboxed.shape, model_weights.coef_, model_weights.intercept_
        # ))

        gradient_results = self.gradient_computer.compute_gradient_procedure(*args)
        self._update_w_tilde(model_weights)

        if self.iter_k % self.update_interval_L == 0:
            self.count_t += 1
            LOGGER.debug("Before division, this_w_tilde: {}".format(self.this_w_tilde.unboxed))
            self.this_w_tilde /= self.update_interval_L
            LOGGER.debug("After division, this_w_tilde: {}".format(self.this_w_tilde.unboxed))

            if self.count_t > 0:
                LOGGER.info("iter_k: {}, count_t: {}, start to update hessian".format(self.iter_k, self.count_t))
                self._update_hessian(data_instances, optimizer, cipher_operator)
            self.last_w_tilde = self.this_w_tilde
            self.this_w_tilde = LinearModelWeights(np.zeros_like(self.last_w_tilde.unboxed),
                                                   self.last_w_tilde.fit_intercept)
            LOGGER.debug("After replace, last_w_tilde: {}, this_w_tilde: {}".format(self.last_w_tilde.unboxed,
                                                                                    self.this_w_tilde.unboxed))

        return gradient_results

    def compute_loss(self, *args):
        loss = self.gradient_computer.compute_loss(*args)
        return loss


class HeteroStochasticQuansiNewtonGuest(HeteroStochasticQuansiNewton):
    def __init__(self, sqn_param):
        super().__init__(sqn_param)
        self.sqn_sync = sqn_sync.Guest()

    def _update_hessian(self, data_instances, optimizer, cipher_operator):
        suffix = (self.n_iter, self.batch_index)

        sampled_data = self.sqn_sync.sync_sample_data(data_instances, self.sample_size, self.random_seed, suffix=suffix)
        delta_s = self.this_w_tilde - self.last_w_tilde
        host_forwards = self.sqn_sync.get_host_forwards(suffix=suffix)
        forward_hess, hess_vector = self.gradient_computer.compute_forward_hess(sampled_data, delta_s, host_forwards)
        self.sqn_sync.remote_forward_hess(forward_hess, suffix)
        hess_norm = optimizer.hess_vector_norm(delta_s)
        LOGGER.debug("In _update_hessian, hess_norm: {}".format(hess_norm.unboxed))
        hess_vector = hess_vector + hess_norm.unboxed
        self.sqn_sync.sync_hess_vector(hess_vector, suffix)


class HeteroStochasticQuansiNewtonHost(HeteroStochasticQuansiNewton):
    def __init__(self, sqn_param):
        super().__init__(sqn_param)
        self.sqn_sync = sqn_sync.Host()

    def _update_hessian(self, data_instances, optimizer, cipher_operator):
        suffix = (self.n_iter, self.batch_index)
        sampled_data = self.sqn_sync.sync_sample_data(data_instances, suffix=suffix)
        delta_s = self.this_w_tilde - self.last_w_tilde
        LOGGER.debug("In _update_hessian, delta_s: {}".format(delta_s.unboxed))
        host_forwards = self.gradient_computer.compute_sqn_forwards(sampled_data, delta_s, cipher_operator)
        # host_forwards = cipher_operator.encrypt_list(host_forwards)
        self.sqn_sync.remote_host_forwards(host_forwards, suffix=suffix)
        forward_hess = self.sqn_sync.get_forward_hess(suffix=suffix)
        hess_vector = self.gradient_computer.compute_forward_hess(sampled_data, delta_s, forward_hess)
        hess_vector += optimizer.hess_vector_norm(delta_s).unboxed
        self.sqn_sync.sync_hess_vector(hess_vector, suffix)


class HeteroStochasticQuansiNewtonArbiter(HeteroStochasticQuansiNewton):
    def __init__(self, sqn_param):
        super().__init__(sqn_param)
        self.opt_Hess = None
        self.opt_v = None
        self.opt_s = None
        self.sqn_sync = sqn_sync.Arbiter()
        self.model_weight: LinearModelWeights = None

    def _update_w_tilde(self, gradient: LinearModelWeights):
        if self.model_weight is None:
            self.model_weight = copy.deepcopy(gradient)
        else:
            self.model_weight -= gradient

        if self.this_w_tilde is None:
            self.this_w_tilde = copy.deepcopy(self.model_weight)
        else:
            self.this_w_tilde += self.model_weight

    def compute_gradient_procedure(self, cipher_operator, optimizer, n_iter_, batch_index):
        self.batch_index = batch_index
        self.n_iter = n_iter_
        # LOGGER.debug("In compute_gradient_procedure, n_iter: {}, batch_index: {}, iter_k: {}".format(
        #     self.n_iter, self.batch_index, self.iter_k
        # ))

        optimizer.set_hess_matrix(self.opt_Hess)
        delta_grad = self.gradient_computer.compute_gradient_procedure(
            cipher_operator, optimizer, n_iter_, batch_index)
        self._update_w_tilde(LinearModelWeights(delta_grad, fit_intercept=False))
        if self.iter_k % self.update_interval_L == 0:
            self.count_t += 1
            LOGGER.debug("Before division, this_w_tilde: {}".format(self.this_w_tilde.unboxed))
            self.this_w_tilde /= self.update_interval_L
            LOGGER.debug("After division, this_w_tilde: {}".format(self.this_w_tilde.unboxed))

            if self.count_t > 0:
                LOGGER.info("iter_k: {}, count_t: {}, start to update hessian".format(self.iter_k, self.count_t))
                self._update_hessian(cipher_operator)
            self.last_w_tilde = self.this_w_tilde
            self.this_w_tilde = LinearModelWeights(np.zeros_like(self.last_w_tilde.unboxed),
                                                   self.last_w_tilde.fit_intercept)
        return delta_grad

        # self._update_w_tilde(cipher_operator)

    def _update_hessian(self, cipher_operator):
        suffix = (self.n_iter, self.batch_index)
        hess_vectors = self.sqn_sync.sync_hess_vector(suffix)
        hess_vectors = np.array(cipher_operator.decrypt_list(hess_vectors))
        delta_s = self.this_w_tilde - self.last_w_tilde
        LOGGER.debug("In update hessian, hess_vectors: {}, delta_s: {}".format(
            hess_vectors, delta_s.unboxed
        ))
        self.opt_v = self._update_memory_vars(hess_vectors, self.opt_v)
        self.opt_s = self._update_memory_vars(delta_s.unboxed, self.opt_s)
        self._compute_hess_matrix()

    def _update_memory_vars(self, new_vars, memory_vars):
        if isinstance(new_vars, list):
            new_vars = np.array(new_vars)
        if memory_vars is None:
            memory_vars = [0, ]
            memory_vars[0] = new_vars.reshape(-1, 1)
        elif len(memory_vars) < self.memory_M:
            memory_vars.append(new_vars.reshape(-1, 1))
        else:
            memory_vars.pop(0)
            memory_vars.append(new_vars.reshape(-1, 1))
        return memory_vars

    def _compute_hess_matrix(self):
        LOGGER.debug("opt_v: {}, opt_s: {}".format(self.opt_v, self.opt_s))
        rho = sum(self.opt_v[-1] * self.opt_s[-1]) / sum(self.opt_v[-1] * self.opt_v[-1])
        LOGGER.debug("in _compute_hess_matrix, rho0 = {}".format(rho))
        n = self.opt_s[0].shape[0]
        Hess = rho * np.identity(n)
        iter_num = 0
        for y, s in zip(self.opt_v, self.opt_s):
            rho = 1.0 / (y.T.dot(s))
            Hess = (np.identity(n) - rho * s.dot(y.T)).dot(Hess).dot(np.identity(n) - rho * y.dot(s.T)) + rho * s.dot(
                s.T)
            iter_num += 1
            LOGGER.info(
                "hessian updating algorithm iter_num = {}, rho = {} \n ||s|| is {} \n ||y|| is {}".format(iter_num, rho,
                                                                                                          np.linalg.norm(
                                                                                                              s),
                                                                                                          np.linalg.norm(
                                                                                                              y)))

        self.opt_Hess = Hess


def sqn_factory(role, sqn_param):
    if role == consts.GUEST:
        return HeteroStochasticQuansiNewtonGuest(sqn_param)

    if role == consts.HOST:
        return HeteroStochasticQuansiNewtonHost(sqn_param)

    return HeteroStochasticQuansiNewtonArbiter(sqn_param)
