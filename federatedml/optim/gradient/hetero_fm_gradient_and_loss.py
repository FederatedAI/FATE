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

import numpy as np
import functools
from arch.api.utils import log_utils
from federatedml.framework.hetero.sync import loss_sync
from federatedml.optim.gradient import hetero_linear_model_gradient
from federatedml.util.fate_operator import reduce_add
from federatedml.feature.sparse_vector import SparseVector
from scipy.sparse import csr_matrix
from federatedml.util import fate_operator
from federatedml.util import consts
from collections import Iterable
from federatedml.secureprotol.fate_paillier import PaillierEncryptedNumber
LOGGER = log_utils.getLogger()


def __compute_partition_gradient(data, fit_intercept=True):
    feature = []
    fore_gradient = []
    issparsevec = 0

    for key, value in data:
        feature.append(value[0])
        fore_gradient.append(value[1])

    if len(feature) == 0 :
        return 0

    if isinstance(feature[0], SparseVector):
        issparsevec = 1
        feature_num = feature[0].get_shape()

    if issparsevec:
        gradient = [0] * feature_num
        for i in range(len(feature)):
            for idx, v in feature[i].get_all_data():
                tmp = fore_gradient[i] * v
                gradient[idx] = gradient[idx] + tmp

        for i in range(feature_num):
            if gradient[i] == 0 :
                if isinstance(fore_gradient[0], PaillierEncryptedNumber):
                    gradient[i] = 0 * fore_gradient[0]

        if fit_intercept:
            bias_grad = np.sum(fore_gradient)
            gradient.append(bias_grad)

        return np.array(gradient)
    else:
        feature = np.array(feature)
        fore_gradient = np.array(fore_gradient)

        gradient = []
        for j in range(feature.shape[1]):
            feature_col = feature[:, j]
            gradient_j = fate_operator.dot(feature_col, fore_gradient)
            gradient.append(gradient_j)

        if fit_intercept:
            bias_grad = np.sum(fore_gradient)
            gradient.append(bias_grad)
        return np.array(gradient)

def __compute_partition_gradient_1(data,model_feature_emb = None, K = 5, cipher_operator = None):

    feature = []
    fore_gradient = []
    fore_gradient_mul_ui_sum = []
    issparsevec = 0

    for key, value in data:
        feature.append(value[0])
        fore_gradient.append(value[1])
        fore_gradient_mul_ui_sum.append(value[2])

    if len(feature) == 0 :
        return 0

    if isinstance(feature[0], SparseVector):
        issparsevec = 1
        feature_num = feature[0].get_shape()

    if issparsevec:
        # part1
        item = [0] * K
        item = cipher_operator.encrypt_list(item)
        item = np.array(item)
        part1 = [item] * feature_num

        # data : key , value[0] : feature , value[1] : fore_gradient , value[2]: fore_gradient_mul_ui_sum
        for i in range(len(feature)):
            for idx, v in feature[i].get_all_data():
                tmp = v * fore_gradient_mul_ui_sum[i]
                part1[idx] = fate_operator.add(part1[idx],tmp)

        # part2
        item = [0] * K
        item = cipher_operator.encrypt_list(item)
        item = np.array(item)
        part2 = [item] * feature_num

        for i in range(len(feature)):
            for idx ,v in feature[i].get_all_data():
                tmp1 = v * v * fore_gradient[i]
                tmp2 = fate_operator.scalar_mul_array(tmp1,model_feature_emb[idx])
                part2[idx] = fate_operator.add(part2[idx] , tmp2)
        v_gradient = np.array(part1) - np.array(part2)
        return v_gradient
    else:  # dense

        feature = np.array(feature)
        fore_gradient = np.array(fore_gradient)
        fore_gradient_mul_ui_sum = np.array(fore_gradient_mul_ui_sum)

        if feature.shape[0] <= 0:
            return 0
        part1 = feature.transpose().dot(fore_gradient_mul_ui_sum) 

        part2 = []
        feat_square = feature * feature
        feat_square_sum = np.dot(fore_gradient, feat_square) 
        for i in range(feat_square_sum.shape[0]): 
            part2.append(fate_operator.scalar_mul_array(feat_square_sum[i], model_feature_emb[i]))
        part2 = np.array(part2)

        v_gradient = part1 - part2
        return v_gradient


def compute_gradient(data_instances, fore_gradient, fit_intercept, fore_gradient_mul_ui_sum, model_feature_emb, K , cipher_operator):

    # w_gradient 
    feat_join_grad = data_instances.join(fore_gradient,
                                         lambda d, g: (d.features, g))

    f = functools.partial(__compute_partition_gradient,fit_intercept=fit_intercept)

    gradient_partition = feat_join_grad.mapPartitions(f).reduce(lambda x, y: x + y)

    w_gradient = gradient_partition / data_instances.count()

    # v_gradient 
    feat_grad_sum = feat_join_grad.join(fore_gradient_mul_ui_sum, lambda v1, v2 : (v1[0], v1[1], v2))

    f1 = functools.partial(__compute_partition_gradient_1,
                          model_feature_emb=model_feature_emb,
                           K=K,
                           cipher_operator=cipher_operator)
    feature_emb_gradient_partition = feat_grad_sum.mapPartitions(f1).reduce(lambda x,y : x + y)
    v_gradient = feature_emb_gradient_partition / data_instances.count()

    return (w_gradient,v_gradient)


def cal_ui_dot_sum(feature, model_feature_emb):
    feature_num = model_feature_emb.shape[0]
    feature_dim = model_feature_emb.shape[1]
    sum = 0
    if isinstance(feature, SparseVector):
        for idx, v in feature.get_all_data():
            if idx < feature_num:
                sum += np.dot(v * model_feature_emb[idx], v * model_feature_emb[idx])
    else:
        sum = np.sum(np.dot(feature * feature, model_feature_emb * model_feature_emb))
    return sum


def cal_ui_sum(feature, model_feature_emb):
    feature_num = model_feature_emb.shape[0]
    feature_dim = model_feature_emb.shape[1]
    ui_sum = np.zeros(feature_dim)
    if isinstance(feature,SparseVector):
        for idx, v in feature.get_all_data():
            if idx < feature_num:
                ui_sum += model_feature_emb[idx] * v
    else:
        ui_sum = np.dot(feature,model_feature_emb)
    return ui_sum

class Guest(hetero_linear_model_gradient.Guest, loss_sync.Guest):

    def __init__(self):
        super(Guest).__init__()
        self.host_ui_sum = None 
        self.host_ui_sum_square = None
        self.host_ui_dot_sum = None
        self.aggregated_ui_sum = None
        self.f_x = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                guest_gradient_transfer, guest_optim_gradient_transfer,
                                host_ui_sum_transfer,host_ui_sum_square_transfer,
                                host_ui_dot_sum_transfer,aggregated_ui_sum,
                                fore_gradient_mul_ui_sum_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = guest_gradient_transfer
        self.unilateral_optim_gradient_transfer = guest_optim_gradient_transfer
        self.host_ui_sum_transfer = host_ui_sum_transfer
        self.host_ui_sum_square_transfer = host_ui_sum_square_transfer
        self.host_ui_dot_sum_transfer = host_ui_dot_sum_transfer
        self.aggregated_ui_sum_transfer = aggregated_ui_sum
        self.fore_gradient_mul_ui_sum_transfer = fore_gradient_mul_ui_sum_transfer

    def _register_loss_sync(self, host_loss_regular_transfer, loss_transfer, loss_intermediate_transfer,
                            f_x_transfer,en_f_x_square_transfer):
        self.host_loss_regular_transfer = host_loss_regular_transfer
        self.loss_transfer = loss_transfer
        self.loss_intermediate_transfer = loss_intermediate_transfer
        self.f_x_transfer = f_x_transfer
        self.en_f_x_square_transfer = en_f_x_square_transfer

    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward_dict,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.guest_gradient,
                                     transfer_variables.guest_optim_gradient,
                                     transfer_variables.host_ui_sum,
                                     transfer_variables.host_ui_sum_square,
                                     transfer_variables.host_ui_dot_sum,
                                     transfer_variables.aggregated_ui_sum,
                                     transfer_variables.fore_gradient_mul_ui_sum
                                     )

        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_intermediate,
                                 transfer_variables.f_x,
                                 transfer_variables.en_f_x_square
                                 )

    def compute_and_aggregate_forwards(self, data_instances, model_weights,model_feature_emb, encrypted_calculator, batch_index,offset=None):

        # compute f(x)
        # step1: compute [[wx]]
        half_wx = data_instances.mapValues(
            lambda v: fate_operator.vec_dot(v.features, model_weights.coef_) +  model_weights.intercept_)
        self.forwards = half_wx
        self.aggregated_forwards = encrypted_calculator[batch_index].encrypt(half_wx)
        self.aggregated_forwards = self.aggregated_forwards.join(self.host_forwards, lambda g, h: g + h) 
        
        # step2: compute FM part
        half_ui_sum = data_instances.mapValues(lambda v : cal_ui_sum(v.features,model_feature_emb))       
        en_half_ui_sum = encrypted_calculator[batch_index].encrypt(half_ui_sum) 
        self.aggregated_ui_sum = en_half_ui_sum.join(self.host_ui_sum, lambda g,h : g+h)

        half_ui_sum_square = half_ui_sum.mapValues(lambda v : v * v)
        en_half_ui_sum_square = encrypted_calculator[batch_index].encrypt(half_ui_sum_square)
        ui_sum_and_sum_square = half_ui_sum.join(en_half_ui_sum_square,
                                                 lambda ui_sum, en_ui_sum_square: (ui_sum, en_ui_sum_square))  # (ui_sum, en_ui_sum_square)
        host_ui_sum_and_sum_square = self.host_ui_sum.join(self.host_ui_sum_square, lambda u1,u2 : (u1,u2))
        part1 = ui_sum_and_sum_square.join(host_ui_sum_and_sum_square,
                                   lambda g,h : np.sum( g[1] + h[1] + 2 * g[0] * h[0]) ) # (ui_sum_g, en_ui_sum_g, en_ui_sum_square_g, en_ui_sum_h, en_ui_sum_square_h)

        half_ui_dot_sum = data_instances.mapValues(lambda v : cal_ui_dot_sum(v.features, model_feature_emb))
        en_half_ui_dot_sum = encrypted_calculator[batch_index].encrypt(half_ui_dot_sum)
        part2 = en_half_ui_dot_sum.join(self.host_ui_dot_sum, lambda g,h : g + h)

        res1 = part1.join(part2, lambda p1,p2 : 0.5 * (p1 - p2))
        self.f_x = res1.join(self.aggregated_forwards, lambda r1, r2 : r1 + r2)
        
        fore_gradient = data_instances.join(self.f_x , lambda d,r : 0.25 * r - 0.5 * d.label )
        return fore_gradient

    def get_host_ui_sum(self, suffix=tuple()):
        host_ui_sum = self.host_ui_sum_transfer.get(idx=0, suffix=suffix)
        return host_ui_sum

    def get_host_ui_sum_square(self, suffix=tuple()):
        host_ui_sum_square = self.host_ui_sum_square_transfer.get(idx=0, suffix=suffix)
        return host_ui_sum_square

    def get_host_ui_dot_sum(self, suffix=tuple()):
        host_ui_dot_sum = self.host_ui_dot_sum_transfer.get(idx=0, suffix=suffix)
        return host_ui_dot_sum

    def get_fore_gradient_mul_ui_sum(self, suffix=tuple()):
        fore_gradient_mul_ui_sum = self.fore_gradient_mul_ui_sum_transfer.get(idx=0, suffix=suffix)
        return fore_gradient_mul_ui_sum

    def get_host_forward(self, suffix=tuple()):
        host_forward = self.host_forward_transfer.get(idx=0, suffix=suffix)
        return host_forward

    def remote_remote_ui_sum(self, ui_sum, suffix=tuple()):
        self.aggregated_ui_sum_transfer.remote(obj=ui_sum, role=consts.ARBITER, idx=0, suffix=suffix)

    def remote_fore_gradient(self, fore_gradient, suffix=tuple()):
        self.fore_gradient_transfer.remote(obj=fore_gradient, role=consts.HOST, idx=0, suffix=suffix)
        self.fore_gradient_transfer.remote(obj=fore_gradient, role=consts.ARBITER, idx=0, suffix=suffix)

    def get_host_forward(self, suffix=tuple()):
        host_forward = self.host_forward_transfer.get(idx=0, suffix=suffix)
        return host_forward

    def get_host_loss_regular(self, suffix=tuple()):
        loss = self.host_loss_regular_transfer.get(idx=0, suffix=suffix)
        return loss

    def compute_gradient_procedure(self, data_instances, model_weights, model_feature_emb,
                                   encrypted_calculator, cipher_operator, optimizer_w,optimizer_v,
                                   n_iter_, batch_index , K ,offset = None):

        current_suffix = (n_iter_, batch_index)
        self.host_forwards = self.get_host_forward(suffix=current_suffix) 

        self.host_ui_sum = self.get_host_ui_sum(suffix=current_suffix)

        self.host_ui_sum_square = self.get_host_ui_sum_square(suffix=current_suffix)

        self.host_ui_dot_sum = self.get_host_ui_dot_sum(suffix = current_suffix)

        fore_gradient = self.compute_and_aggregate_forwards(data_instances, model_weights, model_feature_emb.coef_,
                                                            encrypted_calculator, batch_index, offset)

        self.remote_fore_gradient(fore_gradient, suffix=current_suffix)
        LOGGER.info("guest remote fore_gradient to host and arbiter!")
        self.aggregated_ui_sum_transfer.remote(obj=self.aggregated_ui_sum, role=consts.ARBITER, idx=0, suffix=current_suffix)
        LOGGER.info("guest remote aggregated_ui_sum to arbiter!")

        fore_gradient_mul_ui_sum = self.get_fore_gradient_mul_ui_sum(suffix=current_suffix)
        LOGGER.info("guest get fore_gradient_mul_ui_sum from arbiter!")

        unilateral_gradient = compute_gradient(data_instances,
                                               fore_gradient,
                                               model_weights.fit_intercept,
                                               fore_gradient_mul_ui_sum,
                                               model_feature_emb.coef_,
                                               K,cipher_operator)
        if optimizer_w is not None and optimizer_v is not None:
            unilateral_gradient_0_tmp = optimizer_w.add_regular_to_grad(unilateral_gradient[0], model_weights)
            unilateral_gradient_1_tmp = optimizer_v.add_regular_to_grad(unilateral_gradient[1], model_feature_emb)
            unilateral_gradient = (unilateral_gradient_0_tmp, unilateral_gradient_1_tmp)

        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)

        return optimized_gradient, fore_gradient, self.host_forwards

    def compute_loss(self, data_instances, n_iter_, batch_index, loss_norm=None):

        current_suffix = (n_iter_, batch_index)
        n = data_instances.count()
        yf = self.f_x.join(data_instances, lambda f, d: f * int(d.label)).reduce(reduce_add)
        self.f_x_transfer.remote(obj=self.f_x, role=consts.ARBITER, idx=0, suffix=current_suffix)
        en_f_x_square = self.en_f_x_square_transfer.get(idx=0, suffix=current_suffix)
        loss_list = []

        if loss_norm is not None:
            host_loss_regular = self.get_host_loss_regular(suffix=current_suffix)
        else:
            host_loss_regular = None

        loss = np.log(2) - 0.5 * (1 / n) * yf + 0.125 * (1 / n) * en_f_x_square
        if loss_norm is not None:
            loss += loss_norm
            loss += host_loss_regular
        loss_list.append(loss)
        LOGGER.debug("In compute_loss, loss list are: {}".format(loss_list))
        self.sync_loss_info(loss_list, suffix=current_suffix)

class Host(hetero_linear_model_gradient.Host, loss_sync.Host):

    def __init__(self):
        super(Host).__init__()
        self.ui_sum_and_sum_square = None
        self.ui_dot_sum = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                host_gradient_transfer, host_optim_gradient_transfer,
                                host_ui_sum_transfer, host_ui_sum_square_transfer,
                                host_ui_dot_sum_transfer ,fore_gradient_mul_ui_sum_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = host_gradient_transfer
        self.unilateral_optim_gradient_transfer = host_optim_gradient_transfer
        self.host_ui_sum_transfer = host_ui_sum_transfer
        self.host_ui_sum_square_transfer = host_ui_sum_square_transfer
        self.host_ui_dot_sum_transfer = host_ui_dot_sum_transfer
        self.fore_gradient_mul_ui_sum_transfer = fore_gradient_mul_ui_sum_transfer

    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward_dict,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.host_optim_gradient,
                                     transfer_variables.host_ui_sum,
                                     transfer_variables.host_ui_sum_square,
                                     transfer_variables.host_ui_dot_sum,
                                     transfer_variables.fore_gradient_mul_ui_sum)

        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_intermediate)

    def compute_forwards(self, data_instances, model_weights, model_feature_emb):

        forward = data_instances.mapValues(
            lambda v: fate_operator.vec_dot(v.features, model_weights.coef_) +  model_weights.intercept_)
        ui_sum = data_instances.mapValues(lambda v : cal_ui_sum(v.features,model_feature_emb)) 
        ui_sum_square = ui_sum.mapValues(lambda u: u * u)
        ui_dot_sum = data_instances.mapValues(lambda v: cal_ui_dot_sum(v.features, model_feature_emb))

        return (forward,ui_sum, ui_sum_square,ui_dot_sum)

    def compute_gradient_procedure(self, data_instances, model_weights, model_feature_emb,
                                   encrypted_calculator, cipher_operator, optimizer_w,optimizer_v,
                                   n_iter_, batch_index , K):

        current_suffix = (n_iter_, batch_index)

        self.forwards, self.ui_sum, self.ui_sum_square, self.ui_dot_sum = self.compute_forwards(data_instances, model_weights, model_feature_emb.coef_)
        encrypted_forward = encrypted_calculator[batch_index].encrypt(self.forwards)
        encrypted_ui_sum = encrypted_calculator[batch_index].encrypt(self.ui_sum)
        encrypted_ui_sum_square = encrypted_calculator[batch_index].encrypt(self.ui_sum_square)
        encrypted_ui_dot_sum = encrypted_calculator[batch_index].encrypt(self.ui_dot_sum)

        self.remote_host_forward(encrypted_forward, suffix=current_suffix)
        self.remote_ui_sum(encrypted_ui_sum, suffix=current_suffix)
        self.remote_ui_sum_square(encrypted_ui_sum_square, suffix=current_suffix)
        self.remote_ui_dot_sum(encrypted_ui_dot_sum, suffix=current_suffix)

        fore_gradient = self.get_fore_gradient(suffix=current_suffix)
        fore_gradient_mul_ui_sum = self.get_fore_gradient_mul_ui_sum(suffix=current_suffix)
        unilateral_gradient = compute_gradient(data_instances,
                                               fore_gradient,
                                               model_weights.fit_intercept,
                                               fore_gradient_mul_ui_sum,
                                               model_feature_emb.coef_,
                                               K,cipher_operator)

        if optimizer_w is not None and optimizer_v is not None :
            unilateral_gradient_0_tmp = optimizer_w.add_regular_to_grad(unilateral_gradient[0], model_weights)
            unilateral_gradient_1_tmp = optimizer_v.add_regular_to_grad(unilateral_gradient[1], model_feature_emb)
            unilateral_gradient = (unilateral_gradient_0_tmp, unilateral_gradient_1_tmp)
 
        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        return optimized_gradient, fore_gradient

    def compute_loss(self, lr_weights, optimizer, n_iter_, batch_index):

        current_suffix = (n_iter_, batch_index)
        loss_regular = optimizer.loss_norm(lr_weights)
        self.remote_loss_regular(loss_regular, suffix=current_suffix)

    def remote_ui_sum(self, ui_sum, suffix=tuple()):
        self.host_ui_sum_transfer.remote(obj=ui_sum, role=consts.GUEST, idx=0, suffix=suffix)

    def remote_ui_sum_square(self, ui_sum_square, suffix=tuple()):
        self.host_ui_sum_square_transfer.remote(obj=ui_sum_square, role=consts.GUEST, idx=0, suffix=suffix)

    def remote_ui_dot_sum(self, ui_dot_sum, suffix=tuple()):
        self.host_ui_dot_sum_transfer.remote(obj=ui_dot_sum, role=consts.GUEST, idx=0, suffix=suffix)

    def get_fore_gradient_mul_ui_sum(self, suffix=tuple()):
        fore_gradient_mul_ui_sum = self.fore_gradient_mul_ui_sum_transfer.get(idx=0, suffix=suffix)
        return fore_gradient_mul_ui_sum

class Arbiter(hetero_linear_model_gradient.Arbiter, loss_sync.Arbiter):
    def __init__(self):
        super().__init__()

    def _register_gradient_sync(self, guest_gradient_transfer, host_gradient_transfer,
                                guest_optim_gradient_transfer, host_optim_gradient_transfer,
                                fore_gradient_transfer,aggregated_ui_sum_transfer,
                                fore_gradient_mul_ui_sum_transfer):
        self.guest_gradient_transfer = guest_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.guest_optim_gradient_transfer = guest_optim_gradient_transfer
        self.host_optim_gradient_transfer = host_optim_gradient_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.aggregated_ui_sum_transfer = aggregated_ui_sum_transfer
        self.fore_gradient_mul_ui_sum_transfer = fore_gradient_mul_ui_sum_transfer

    def _register_loss_sync(self, loss_transfer, f_x_transfer, en_f_x_square_transfer):
        self.loss_transfer = loss_transfer
        self.f_x_transfer = f_x_transfer
        self.en_f_x_square_transfer = en_f_x_square_transfer

    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.guest_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.guest_optim_gradient,
                                     transfer_variables.host_optim_gradient,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.aggregated_ui_sum,
                                     transfer_variables.fore_gradient_mul_ui_sum
                                     )
        self._register_loss_sync(transfer_variables.loss,transfer_variables.f_x,
                                     transfer_variables.en_f_x_square)

    def compute_loss(self, cipher_operator, n_iter_, batch_index):
        if self.has_multiple_hosts:
            LOGGER.info("Has more than one host, loss is not available")
            return []

        current_suffix = (n_iter_, batch_index)
        fx = self.f_x_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get f_x from Guest")
        fx_square = fx.mapValues(lambda x : self.cal_fx_square(x,cipher_operator)).reduce(reduce_add)
        self.en_f_x_square_transfer.remote(fx_square, role=consts.GUEST, idx=0,suffix=current_suffix)

        loss_list = self.sync_loss_info(suffix=current_suffix)
        de_loss_list = cipher_operator.decrypt_list(loss_list)
        return de_loss_list

    def cal_fx_square(self,x,cipher_operator):
        x1 = self.decrypt_row(x,cipher_operator)
        x1_square = cipher_operator.encrypt ( x1 * x1)
        return x1_square

    def decrypt_row(self, row,cipher_operator):
        if type(row).__name__ == "ndarray":
            return np.array([cipher_operator.decrypt(val) for val in row])
        elif isinstance(row, Iterable):
            return type(row)(cipher_operator.decrypt(val) for val in row)
        else:
            return cipher_operator.decrypt(row)

    def get_local_gradient(self, suffix=tuple()):
        host_gradient = self.host_gradient_transfer.get(idx=0, suffix=suffix)
        LOGGER.info("Get host_gradient from Host")

        guest_gradient = self.guest_gradient_transfer.get(idx=0, suffix=suffix)
        LOGGER.info("Get guest_gradient from Guest")
        return host_gradient, guest_gradient

    def compute_gradient_procedure(self, cipher_operator, optimizer_w,optimizer_v, n_iter_, batch_index):
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

        en_fore_gradient = self.fore_gradient_transfer.get(idx=0, suffix=current_suffix) 
        LOGGER.info("Get fore_gradient from Guest")
        en_ui_sum = self.aggregated_ui_sum_transfer.get(idx=0, suffix=current_suffix)
        LOGGER.info("Get ui_sum from Guest")

        # decrypt en_ui_sum
        fore_gradient = en_fore_gradient.mapValues(lambda v : self.decrypt_row(v,cipher_operator))
        fore_gradient_mul_ui_sum = fore_gradient.join(en_ui_sum,lambda f,ui_sum: f * ui_sum) 
        self.fore_gradient_mul_ui_sum_transfer.remote(fore_gradient_mul_ui_sum,role=consts.HOST,idx=-1, suffix=current_suffix)
        LOGGER.info("arbiter remote fore_gradient_mul_ui_sum to host ")
        self.fore_gradient_mul_ui_sum_transfer.remote(fore_gradient_mul_ui_sum, role=consts.GUEST, idx=-1,suffix=current_suffix)
        LOGGER.info("arbiter remote fore_gradient_mul_ui_sum to guest ")

        host_gradient, guest_gradient = self.get_local_gradient(current_suffix)

        host_gradient_w = host_gradient[0]
        host_gradient_v = host_gradient[1]

        guest_gradient_w = guest_gradient[0]
        guest_gradient_v = guest_gradient[1]

        # update w
        host_gradient_w = np.array(host_gradient_w)
        guest_gradient_w = np.array(guest_gradient_w)

        gradient_w = np.hstack((host_gradient_w, guest_gradient_w))
        grad_w = np.array(cipher_operator.decrypt_list(gradient_w))

        delta_grad_w = optimizer_w.apply_gradients(grad_w)

        separate_optim_gradient_w = self.separate(delta_grad_w, [host_gradient_w.shape[0], guest_gradient_w.shape[0]])
 
        host_optim_gradient_w = separate_optim_gradient_w[0]
        guest_optim_gradient_w = separate_optim_gradient_w[1]

        # update v 
        host_gradient_v = np.array(host_gradient_v)
        guest_gradient_v = np.array(guest_gradient_v)

        gradient_v = np.vstack((host_gradient_v, guest_gradient_v))
        grad_v = np.array(cipher_operator.decrypt_2d_array(gradient_v))

        delta_grad_v = optimizer_v.apply_gradients(grad_v)

        separate_optim_gradient_v = self.separate(delta_grad_v, [host_gradient_v.shape[0], guest_gradient_v.shape[0]])
 
        host_optim_gradient_v = separate_optim_gradient_v[0]
        guest_optim_gradient_v = separate_optim_gradient_v[1]

        host_optim_gradient = (host_optim_gradient_w, host_optim_gradient_v)
        guest_optim_gradient = (guest_optim_gradient_w, guest_optim_gradient_v)
        self.host_optim_gradient_transfer.remote(host_optim_gradient,role=consts.HOST,idx=-1,suffix=current_suffix)
        self.guest_optim_gradient_transfer.remote(guest_optim_gradient, role=consts.GUEST, idx=-1, suffix=current_suffix)

        return (delta_grad_w,delta_grad_v)

