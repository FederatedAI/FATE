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

import functools

import numpy as np
import scipy.sparse as sp

from federatedml.feature.sparse_vector import SparseVector
from federatedml.statistic import data_overview
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util import fate_operator


def __compute_partition_gradient(data, fit_intercept=True, is_sparse=False):
    """
    Compute hetero regression gradient for:
    gradient = ∑d*x, where d is fore_gradient which differ from different algorithm
    Parameters
    ----------
    data: DTable, include fore_gradient and features
    fit_intercept: bool, if model has interception or not. Default True

    Returns
    ----------
    numpy.ndarray
        hetero regression model gradient
    """
    feature = []
    fore_gradient = []

    if is_sparse:
        row_indice = []
        col_indice = []
        data_value = []

        row = 0
        feature_shape = None
        for key, (sparse_features, d) in data:
            fore_gradient.append(d)
            assert isinstance(sparse_features, SparseVector)
            if feature_shape is None:
                feature_shape = sparse_features.get_shape()
            for idx, v in sparse_features.get_all_data():
                col_indice.append(idx)
                row_indice.append(row)
                data_value.append(v)
            row += 1
        if feature_shape is None or feature_shape == 0:
            return 0
        sparse_matrix = sp.csr_matrix((data_value, (row_indice, col_indice)), shape=(row, feature_shape))
        fore_gradient = np.array(fore_gradient)

        # gradient = sparse_matrix.transpose().dot(fore_gradient).tolist()
        gradient = fate_operator.dot(sparse_matrix.transpose(), fore_gradient).tolist()
        if fit_intercept:
            bias_grad = np.sum(fore_gradient)
            gradient.append(bias_grad)
            # LOGGER.debug("In first method, gradient: {}, bias_grad: {}".format(gradient, bias_grad))
        return np.array(gradient)

    else:
        for key, value in data:
            feature.append(value[0])
            fore_gradient.append(value[1])
        feature = np.array(feature)
        fore_gradient = np.array(fore_gradient)
        if feature.shape[0] <= 0:
            return 0

        gradient = fate_operator.dot(feature.transpose(), fore_gradient)
        gradient = gradient.tolist()
        if fit_intercept:
            bias_grad = np.sum(fore_gradient)
            gradient.append(bias_grad)
        return np.array(gradient)


def compute_gradient(data_instances, fore_gradient, fit_intercept):
    """
    Compute hetero-regression gradient
    Parameters
    ----------
    data_instances: DTable, input data
    fore_gradient: DTable, fore_gradient
    fit_intercept: bool, if model has intercept or not

    Returns
    ----------
    DTable
        the hetero regression model's gradient
    """
    feat_join_grad = data_instances.join(fore_gradient,
                                         lambda d, g: (d.features, g))
    is_sparse = data_overview.is_sparse_data(data_instances)
    f = functools.partial(__compute_partition_gradient,
                          fit_intercept=fit_intercept,
                          is_sparse=is_sparse)
    gradient_partition = feat_join_grad.applyPartitions(f)
    gradient_partition = gradient_partition.reduce(lambda x, y: x + y)

    gradient = gradient_partition / data_instances.count()

    return gradient


class HeteroGradientBase(object):
    def compute_gradient_procedure(self, *args):
        raise NotImplementedError("Should not call here")

    def set_total_batch_nums(self, total_batch_nums):
        """	
        Use for sqn gradient.	
        """
        pass


class Guest(HeteroGradientBase):
    def __init__(self):
        self.host_forwards = None
        self.forwards = None
        self.aggregated_forwards = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                guest_gradient_transfer, guest_optim_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = guest_gradient_transfer
        self.unilateral_optim_gradient_transfer = guest_optim_gradient_transfer

    def compute_and_aggregate_forwards(self, data_instances, model_weights,
                                       encrypted_calculator, batch_index, current_suffix, offset=None):
        raise NotImplementedError("Function should not be called here")

    def compute_gradient_procedure(self, data_instances, encrypted_calculator, model_weights, optimizer,
                                   n_iter_, batch_index, offset=None):
        """
          Linear model gradient procedure
          Step 1: get host forwards which differ from different algorithm
                  For Logistic Regression and Linear Regression: forwards = wx
                  For Poisson Regression, forwards = exp(wx)

          Step 2: Compute self forwards and aggregate host forwards and get d = fore_gradient

          Step 3: Compute unilateral gradient = ∑d*x,

          Step 4: Send unilateral gradients to arbiter and received the optimized and decrypted gradient.
          """
        current_suffix = (n_iter_, batch_index)
        # self.host_forwards = self.get_host_forward(suffix=current_suffix)

        fore_gradient = self.compute_and_aggregate_forwards(data_instances, model_weights, encrypted_calculator,
                                                            batch_index, current_suffix, offset)

        self.remote_fore_gradient(fore_gradient, suffix=current_suffix)

        unilateral_gradient = compute_gradient(data_instances,
                                               fore_gradient,
                                               model_weights.fit_intercept)
        if optimizer is not None:
            unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, model_weights)

        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        return optimized_gradient, fore_gradient, self.host_forwards

    def get_host_forward(self, suffix=tuple()):
        host_forward = self.host_forward_transfer.get(idx=-1, suffix=suffix)
        return host_forward

    def remote_fore_gradient(self, fore_gradient, suffix=tuple()):
        self.fore_gradient_transfer.remote(obj=fore_gradient, role=consts.HOST, idx=-1, suffix=suffix)

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient


class Host(HeteroGradientBase):
    def __init__(self):
        self.forwards = None
        self.fore_gradient = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                host_gradient_transfer, host_optim_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = host_gradient_transfer
        self.unilateral_optim_gradient_transfer = host_optim_gradient_transfer

    def compute_forwards(self, data_instances, model_weights):
        raise NotImplementedError("Function should not be called here")

    def compute_unilateral_gradient(self, data_instances, fore_gradient, model_weights, optimizer):
        raise NotImplementedError("Function should not be called here")

    def compute_gradient_procedure(self, data_instances, encrypted_calculator, model_weights,
                                   optimizer,
                                   n_iter_, batch_index):
        """
        Linear model gradient procedure
        Step 1: get host forwards which differ from different algorithm
                For Logistic Regression: forwards = wx


        """
        current_suffix = (n_iter_, batch_index)

        self.forwards = self.compute_forwards(data_instances, model_weights)
        encrypted_forward = encrypted_calculator[batch_index].encrypt(self.forwards)

        self.remote_host_forward(encrypted_forward, suffix=current_suffix)
        fore_gradient = self.get_fore_gradient(suffix=current_suffix)

        unilateral_gradient = compute_gradient(data_instances,
                                               fore_gradient,
                                               model_weights.fit_intercept)
        if optimizer is not None:
            unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, model_weights)

        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        return optimized_gradient, fore_gradient

    def compute_sqn_forwards(self, data_instances, delta_s, cipher_operator):
        """
        To compute Hessian matrix, y, s are needed.
        g = (1/N)*∑(0.25 * wx - 0.5 * y) * x
        y = ∇2^F(w_t)s_t = g' * s = (1/N)*∑(0.25 * x * s) * x
        define forward_hess = ∑(0.25 * x * s)
        """
        sqn_forwards = data_instances.mapValues(
            lambda v: cipher_operator.encrypt(fate_operator.vec_dot(v.features, delta_s.coef_) + delta_s.intercept_))
        # forward_sum = sqn_forwards.reduce(reduce_add)
        return sqn_forwards

    def compute_forward_hess(self, data_instances, delta_s, forward_hess):
        """
        To compute Hessian matrix, y, s are needed.
        g = (1/N)*∑(0.25 * wx - 0.5 * y) * x
        y = ∇2^F(w_t)s_t = g' * s = (1/N)*∑(0.25 * x * s) * x
        define forward_hess = (0.25 * x * s)
        """
        hess_vector = compute_gradient(data_instances,
                                       forward_hess,
                                       delta_s.fit_intercept)
        return np.array(hess_vector)

    def remote_host_forward(self, host_forward, suffix=tuple()):
        self.host_forward_transfer.remote(obj=host_forward, role=consts.GUEST, idx=0, suffix=suffix)

    def get_fore_gradient(self, suffix=tuple()):
        host_forward = self.fore_gradient_transfer.get(idx=0, suffix=suffix)
        return host_forward

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient


class Arbiter(HeteroGradientBase):
    def __init__(self):
        self.has_multiple_hosts = False

    def _register_gradient_sync(self, guest_gradient_transfer, host_gradient_transfer,
                                guest_optim_gradient_transfer, host_optim_gradient_transfer):
        self.guest_gradient_transfer = guest_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.guest_optim_gradient_transfer = guest_optim_gradient_transfer
        self.host_optim_gradient_transfer = host_optim_gradient_transfer

    def compute_gradient_procedure(self, cipher_operator, optimizer, n_iter_, batch_index):
        """
        Compute gradients.
        Received local_gradients from guest and hosts. Merge and optimize, then separate and remote back.
        Parameters
        ----------
        cipher_operator: Use for encryption

        optimizer: optimizer that get delta gradient of this iter

        n_iter_: int, current iter nums

        batch_index: int, use to obtain current encrypted_calculator

        """
        current_suffix = (n_iter_, batch_index)

        host_gradients, guest_gradient = self.get_local_gradient(current_suffix)

        if len(host_gradients) > 1:
            self.has_multiple_hosts = True

        host_gradients = [np.array(h) for h in host_gradients]
        guest_gradient = np.array(guest_gradient)

        size_list = [h_g.shape[0] for h_g in host_gradients]
        size_list.append(guest_gradient.shape[0])

        gradient = np.hstack((h for h in host_gradients))
        gradient = np.hstack((gradient, guest_gradient))

        grad = np.array(cipher_operator.decrypt_list(gradient))

        # LOGGER.debug("In arbiter compute_gradient_procedure, before apply grad: {}, size_list: {}".format(
        #     grad, size_list
        # ))

        delta_grad = optimizer.apply_gradients(grad)

        # LOGGER.debug("In arbiter compute_gradient_procedure, delta_grad: {}".format(
        #     delta_grad
        # ))
        separate_optim_gradient = self.separate(delta_grad, size_list)
        # LOGGER.debug("In arbiter compute_gradient_procedure, separated gradient: {}".format(
        #     separate_optim_gradient
        # ))
        host_optim_gradients = separate_optim_gradient[: -1]
        guest_optim_gradient = separate_optim_gradient[-1]

        self.remote_local_gradient(host_optim_gradients, guest_optim_gradient, current_suffix)
        return delta_grad

    @staticmethod
    def separate(value, size_list):
        """
        Separate value in order to several set according size_list
        Parameters
        ----------
        value: list or ndarray, input data
        size_list: list, each set size

        Returns
        ----------
        list
            set after separate
        """
        separate_res = []
        cur = 0
        for size in size_list:
            separate_res.append(value[cur:cur + size])
            cur += size
        return separate_res

    def get_local_gradient(self, suffix=tuple()):
        host_gradients = self.host_gradient_transfer.get(idx=-1, suffix=suffix)
        LOGGER.info("Get host_gradient from Host")

        guest_gradient = self.guest_gradient_transfer.get(idx=0, suffix=suffix)
        LOGGER.info("Get guest_gradient from Guest")
        return host_gradients, guest_gradient

    def remote_local_gradient(self, host_optim_gradients, guest_optim_gradient, suffix=tuple()):
        for idx, host_optim_gradient in enumerate(host_optim_gradients):
            self.host_optim_gradient_transfer.remote(host_optim_gradient,
                                                     role=consts.HOST,
                                                     idx=idx,
                                                     suffix=suffix)

        self.guest_optim_gradient_transfer.remote(guest_optim_gradient,
                                                  role=consts.GUEST,
                                                  idx=0,
                                                  suffix=suffix)
