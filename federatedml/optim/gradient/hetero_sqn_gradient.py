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

from federatedml.linear_model.linear_model_weight import LinearModelWeights
from federatedml.optim.gradient import hetero_linr_gradient_and_loss
from federatedml.param.sqn_param import StochasticQuasiNewtonParam


class HeteroStochasticQuansiNewton(object):
    def __init__(self, sqn_param: StochasticQuasiNewtonParam):
        self.gradient_computer = None
        self.n_iter = 0
        self.total_batch_nums = 0
        self.batch_index = 0
        self.last_w_tilde: LinearModelWeights = None
        self.this_w_tilde: LinearModelWeights = None
        self.sqn_param = sqn_param

    @property
    def iter_k(self):
        return self.n_iter * self.total_batch_nums + self.batch_index + 1

    def set_n_iter(self, n_iter):
        self.n_iter = n_iter

    def set_total_batch_nums(self, total_batch_nums):
        self.total_batch_nums = total_batch_nums

    def set_batch_index(self, batch_index):
        self.batch_index = batch_index

    def register_gradient_computer(self, gradient_computer):
        self.gradient_computer = gradient_computer


class HeteroStochasticQuansiNewtonGuest(object):

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


class HeteroStochasticQuansiNewtonArbiter(object):
    def __init__(self):
        self.opt_Hess = None
        self.opt_v = None
        self.opt_s = None
        self.counter_t = -1

    def compute_gradient_procedure(self):
        # Step 1: add w_tilde

        # step 2: Accept gradient from guest and hosts

        # if iter_k < 2L:
        #   sgd update directly
        # else:
        #   Hess update

        # if iter_k % L == 0:
        #   Update Hessian Matrix
        pass