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

from federatedml.param.sqn_param import StochasticQuasiNewtonParam
from federatedml.optim.gradient import hetero_linear_model_gradient
from federatedml.optim.gradient import sqn_sync
from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.util import consts
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HeteroStochasticQuansiNewton(hetero_linear_model_gradient.HeteroGradientBase):
    def __init__(self, sqn_param: StochasticQuasiNewtonParam):
        self.gradient_computer = None
        self.transfer_variable = None
        self.sqn_sync = None
        self.n_iter = 0
        self.count_t = -1
        self.total_batch_nums = 0
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
        return self.n_iter * self.total_batch_nums + self.batch_index + 1

    def set_total_batch_nums(self, total_batch_nums):
        self.total_batch_nums = total_batch_nums

    def register_gradient_computer(self, gradient_computer):
        self.gradient_computer = copy.deepcopy(gradient_computer)

    def register_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable
        self.sqn_sync.register_transfer_variable(self.transfer_variable)

    def compute_gradient_procedure(self, *args):
        raise NotImplementedError("Should not call here")

    def compute_loss(self, *args):
        raise NotImplementedError("Should not call here")


class HeteroStochasticQuansiNewtonGuest(HeteroStochasticQuansiNewton):
    def __init__(self, sqn_param):
        super().__init__(sqn_param)
        self.sqn_sync = sqn_sync.Guest()

    def compute_gradient_procedure(self, data_instances, encrypted_calculator, model_weights, optimizer,
                                   n_iter_, batch_index, offset=None):
        self.batch_index = batch_index
        self.n_iter = n_iter_
        if self.this_w_tilde is None:
            self.this_w_tilde = copy.deepcopy(model_weights)
        else:
            self.this_w_tilde += model_weights

        if self.iter_k % self.update_interval_L != 0:
            assert isinstance(self.gradient_computer, hetero_linear_model_gradient.Guest)
            optim_guest_gradient, fore_gradient, host_forwards = self.gradient_computer.compute_gradient_procedure(
                data_instances,
                encrypted_calculator,
                model_weights,
                optimizer,
                n_iter_,
                batch_index
            )
        else:
            self.count_t += 1
            self.this_w_tilde /= self.update_interval_L
            if self.count_t > 0:
                LOGGER.info("iter_k: {}, count_t: {}, start to update hessian".format(self.iter_k, self.count_t))
                self._update_hessian(data_instances, optimizer)
            self.last_w_tilde = self.this_w_tilde
            self.this_w_tilde = LinearModelWeights(np.zeros_like(self.last_w_tilde.unboxed),
                                                   self.last_w_tilde.fit_intercept)

        # static w_tilde i.e. sum(w)
        # if iter_k % L == 0:
        #   if count > 0: (i.e. iter_k > 2L)
        #       1. sample data
        #       2. get delta_s = self.this_w_tilde - self.last_w_tilde
        #       3. Compute hess_vector = 1/4 * (W * delta_s) * batch_size / sample_size + alpha * delta_s(if L2)
        #       4. Send hess_vector to arbiter
        #   else:
        #       last_w_tilde = this_w_tilde
        #       this_w_tilde = zero_like()

    def _update_hessian(self, data_instances, optimizer):
        suffix = (self.n_iter, self.batch_index)

        sampled_data = self.sqn_sync.sync_sample_data(data_instances, self.sample_size, self.random_seed, suffix=suffix)
        delta_s = self.this_w_tilde - self.last_w_tilde
        host_forwards = self.sqn_sync.get_host_forwards(suffix=suffix)
        forward_hess, hess_vector = self.gradient_computer.compute_forward_hess(sampled_data, delta_s, host_forwards)
        self.sqn_sync.remote_forward_hess(forward_hess, suffix)
        hess_vector += optimizer.hess_vector_norm(delta_s)
        self.sqn_sync.sync_hess_vector(hess_vector, suffix)


class HeteroStochasticQuansiNewtonHost(HeteroStochasticQuansiNewton):
    def __init__(self, sqn_param):
        super().__init__(sqn_param)
        self.sqn_sync = sqn_sync.Host()

    def compute_gradient_procedure(self):
        # static w_tilde i.e. sum(w)
        # if iter_k % L == 0:
        #   if count > 0: (i.e. iter_k > 2L)
        #       1. sample data
        #       2. get delta_s = self.this_w_tilde - self.last_w_tilde
        #       3. Compute hess_vector = 1/4 * (W * delta_s) * batch_size / sample_size + alpha * delta_s(if L2)
        #       4. Send hess_vector to arbiter
        #   else:
        #       last_w_tilde = this_w_tilde
        #       this_w_tilde = zero_like()

        pass

    def _update_hessian(self, data_instances, optimizer):
        suffix = (self.n_iter, self.batch_index)
        sampled_data = self.sqn_sync.sync_sample_data(data_instances, suffix=suffix)
        delta_s = self.this_w_tilde - self.last_w_tilde
        host_forwards = self.gradient_computer.compute_sqn_forwards(sampled_data, delta_s)
        self.sqn_sync.remote_host_forwards(host_forwards, suffix=suffix)
        forward_hess = self.sqn_sync.get_forward_hess(suffix=suffix)
        hess_vector = self.gradient_computer.compute_forward_hess(sampled_data, delta_s, forward_hess)
        hess_vector += optimizer.hess_vector_norm(delta_s)
        self.sqn_sync.sync_hess_vector(hess_vector, suffix)



class HeteroStochasticQuansiNewtonArbiter(HeteroStochasticQuansiNewton):
    def __init__(self, sqn_param):
        super().__init__(sqn_param)
        self.opt_Hess = None
        self.opt_v = None
        self.opt_s = None
        self.sqn_sync = sqn_sync.Arbiter()

    def compute_gradient_procedure(self, cipher_operator, optimizer, n_iter_, batch_index):
        # Step 1: add w_tilde

        # step 2: Accept gradient from guest and hosts

        # if iter_k < 2L:
        #   sgd update directly
        # else:
        #   Hess update

        # if iter_k % L == 0:
        #   Update Hessian Matrix
        pass

    def _update_hessian(self, cipher_operator):
        suffix = (self.n_iter, self.batch_index)
        hess_vectors = self.sqn_sync.sync_hess_vector(suffix)
        hess_vectors = cipher_operator.decrypt_list(hess_vectors)
        delta_s = self.this_w_tilde - self.last_w_tilde
        self.update_memory_vars(hess_vectors, self.opt_v)
        self.update_memory_vars(delta_s, self.opt_s)

    def update_memory_vars(self, new_vars, memory_vars):
        if memory_vars is None:
            memory_vars = [0, ]
            memory_vars[0] = new_vars.reshape(-1, 1)
        elif len(memory_vars) < self.memory_M:
            memory_vars.append(new_vars.reshape(-1, 1))
        else:
            memory_vars.pop(0)
            memory_vars.append(new_vars.reshape(-1, 1))




def sqn_factory(role, sqn_param):
    if role == consts.GUEST:
        return HeteroStochasticQuansiNewtonGuest(sqn_param)

    if role == consts.HOST:
        return HeteroStochasticQuansiNewtonHost(sqn_param)

    return HeteroStochasticQuansiNewtonArbiter(sqn_param)
