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

from arch.api.utils import log_utils
from federatedml.framework.hetero.sync import loss_sync
from federatedml.util.fate_operator import reduce_add
from federatedml.util import consts


LOGGER = log_utils.getLogger()


def __compute_partition_gradient(data, embed, fit_intercept=True):
    """
    Compute hetero factorization machine gradient for:
    gradient_w = ∑d*x, where d is fore_gradient which differ from different algorithm
    gradient_v = x*∑vx - vx^2
    Parameters
    ----------
    data: DTable, include fore_gradient and features
    embed: Numpy array, embedding of FM model

    Returns
    ----------
    numpy.ndarray
        hetero factorization model gradient
    """
    feature = []
    fore_gradient = []
    vx_mul_fg = []
    x_square_mul_fg = []

    for key, value in data:
        feature.append(value[0])
        fore_gradient.append(value[1])
        vx_mul_fg.append(value[2])
        x_square_mul_fg.append(value[3])
    feature = np.array(feature)
    fore_gradient = np.array(fore_gradient)
    vx_mul_fg = np.array(vx_mul_fg)
    x_square_mul_fg = np.array(x_square_mul_fg)

    if feature.shape[0] <= 0:
        return 0

    # calculate: gradient_w, gradient_v, gradient_intercept
    gradient_w = np.multiply(feature, np.expand_dims(fore_gradient, 1))
    gradient_w = np.sum(gradient_w, axis=0)

    gs = []
    for x, vx_fg, x_square_fg in zip(feature, vx_mul_fg, x_square_mul_fg):
        g1 = np.multiply(x, np.expand_dims(vx_fg, 1))
        g2 = np.multiply(embed.T, x_square_fg)
        gradient_v = g1 - g2
        gs.append(gradient_v.T.flatten())
    gradient_vs = np.sum(gs, axis=0)

    if fit_intercept:
        gradient_intercept = np.array([np.sum(fore_gradient)])
        gradient = [gradient_w, gradient_vs, gradient_intercept]
    else:
        gradient = [gradient_w, gradient_vs]

    return np.concatenate(gradient)


def compute_gradient(data_instances, fore_gradient, vx_mul_fg, x_square_mul_fg, model_weights):
    """
    Compute hetero-factorization machine gradient
    Parameters
    ----------
    data_instances: DTable, input data
    fore_gradient: DTable, fore_gradient
    vx_mul_fg: DTable
    x_square_mul_fg: DTable
    model_weights: FM Model Weight

    Returns
    ----------
    DTable
        the hetero FM model's gradient
    """
    feat_join_grad = data_instances.join(fore_gradient, lambda d, g: (d.features, g))
    feat_join_grad = feat_join_grad.join(vx_mul_fg, lambda a, b: (a[0], a[1], b))

    # Now we have (x, fore_gradient, vx_mul_fg, x_square_mul_fg)
    feat_join_grad = feat_join_grad.join(x_square_mul_fg, lambda a, b: (a[0], a[1], a[2], b))

    f = functools.partial(__compute_partition_gradient,
                          embed=model_weights.embed_,
                          fit_intercept=model_weights.fit_intercept)

    gradient_partition = feat_join_grad.mapPartitions(f).reduce(lambda x, y: x + y)
    gradient = gradient_partition / data_instances.count()

    return gradient


class BaseFM(object):
    def compute_fm(self, data_instances, model_weights, fit_intercept=True):
        """ calculate w*x + (v*x)^2 - (v^2)*(x^2)
            data_instances: batch_size * feature_size
            model_weights: FM Model weights

            model_weights.w_'s shape is feature_size
            model_weights.embed_'s shape is feature_size * embed_size
        """

        def fm_func(features, embed):
            re = np.multiply(np.expand_dims(features, 1), embed)
            re = np.sum(re, 0)
            part1 = np.sum(np.power(re, 2))
            features_square = np.power(features, 2)
            embed_square = np.power(embed, 2)
            part2 = np.sum(np.dot(features_square, embed_square))
            return 0.5*(part1 - part2)

        wx = data_instances.mapValues(lambda v: np.dot(v.features, model_weights.w_))
        fm = data_instances.mapValues(lambda v: fm_func(v.features, model_weights.embed_))
        pred = wx.join(fm, lambda wx_, fm_: wx_ + fm_)

        if fit_intercept:
            pred = pred.mapValues(lambda v: v + model_weights.intercept_)

        return pred

    def compute_vx(self, data_instances, embed_):
        """
        calculate v*x, v refers to embeddings
        :param data_instances: x
        :param embed_: v
        :return: v*x
        """
        return data_instances.mapValues(lambda v: np.dot(v.features, embed_))


class Guest(BaseFM, loss_sync.Guest):
    def __init__(self):
        self.host_forwards = None
        self.forwards = None
        self.aggregated_forwards = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                guest_gradient_transfer, guest_optim_gradient_transfer,
                                agg_vx_mul_fg_transfer, aggregated_forwards_transfer,
                                capped_fore_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.capped_fore_gradient_transfer = capped_fore_gradient_transfer
        self.unilateral_gradient_transfer = guest_gradient_transfer
        self.unilateral_optim_gradient_transfer = guest_optim_gradient_transfer
        self.agg_vx_mul_fg_transfer = agg_vx_mul_fg_transfer
        self.aggregated_forwards_transfer = aggregated_forwards_transfer

    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward_dict,
                                     transfer_variables.fore_gradient,
                                     transfer_variables.guest_gradient,
                                     transfer_variables.guest_optim_gradient,
                                     transfer_variables.agg_vx_mul_fg,
                                     transfer_variables.aggregated_forwards,
                                     transfer_variables.capped_fore_gradient)

        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_intermediate)

    def compute_gradient_procedure(self, data_instances, encrypted_calculator, model_weights, optimizer,
                                   n_iter_, batch_index):
        """
          FM model gradient procedure
          Step 1: get host forwards = w*x + (v*x)^2 - (v^2)*(x^2)

          Step 2: Compute self forwards and aggregate host forwards and get d = fore_gradient

          Step 3: Compute unilateral gradient_w = ∑d*x, gradient_v =

          Step 4: Send unilateral gradients to arbiter and received the optimized and decrypted gradient.
          """
        current_suffix = (n_iter_, batch_index)
        # encrypted host forwards (en_fm, en_vx)
        self.host_forwards = self.get_host_forward(suffix=current_suffix)

        # guest forward includes: 1. en_fm 2. vx 3. en_vx
        # raw_guest_forward is just guest side's fm
        self.guest_forward, self.raw_guest_forward, guest_vx = \
            self.compute_forwards(data_instances, model_weights, encrypted_calculator, batch_index)

        # aggregate forward is used to compute fore_gradient
        en_aggregate_forward = self.compute_and_aggregate_forwards()
        fore_gradient = en_aggregate_forward.join(data_instances, lambda pred, d: 0.25 * pred - 0.5 * d.label)

        # send to arbiter and get capped_fore_gradient
        self.remote_fore_gradient(fore_gradient, suffix=current_suffix)
        capped_fore_gradient = self.get_capped_fore_gradient(suffix=current_suffix)

        agg_vx = self.host_forwards[0].join(guest_vx, lambda host_forward, gvx: host_forward[1] + gvx)
        agg_vx_mul_fg = agg_vx.join(capped_fore_gradient, lambda vx, fg: np.multiply(vx, fg))

        # send vx_mul_fg to host
        self.remote_agg_vx_mul_fg(agg_vx_mul_fg, suffix=current_suffix)

        # compute guest gradient and loss
        x_square = data_instances.mapValues(lambda v: np.multiply(v.features, v.features))
        x_square_mul_fg = x_square.join(capped_fore_gradient, lambda xs, fg: xs * fg)

        # compuate gradient
        # en_capped_fore_gradient = capped_fore_gradient.mapValues(lambda v: encrypted_calculator.encrypt(v))
        en_capped_fore_gradient = encrypted_calculator[batch_index].encrypt(capped_fore_gradient)
        unilateral_gradient = compute_gradient(data_instances,
                                               en_capped_fore_gradient,
                                               agg_vx_mul_fg,
                                               x_square_mul_fg,
                                               model_weights)

        if optimizer is not None:
            unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, model_weights)

        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        return optimized_gradient, fore_gradient

    def get_host_forward(self, suffix=tuple()):
        host_forward = self.host_forward_transfer.get(idx=-1, suffix=suffix)
        return host_forward

    def get_capped_fore_gradient(self, suffix=tuple()):
        capped_fore_gradient = self.capped_fore_gradient_transfer.get(idx=0, suffix=suffix)
        LOGGER.info("Get capped fore gradient from arbiter.")
        return capped_fore_gradient

    def remote_agg_vx_mul_fg(self, agg_vx_mul_fg, suffix=tuple()):
        self.agg_vx_mul_fg_transfer.remote(obj=agg_vx_mul_fg, role=consts.HOST, idx=-1, suffix=suffix)

    def remote_fore_gradient(self, fore_gradient, suffix=tuple()):
        self.fore_gradient_transfer.remote(obj=fore_gradient, role=consts.ARBITER, idx=-1, suffix=suffix)

    def remote_aggregated_forwards(self, aggregated_forwards, suffix=tuple()):
        self.aggregated_forwards_transfer.remote(obj=aggregated_forwards, role=consts.ARBITER,
                                                 idx=0, suffix=suffix)

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient

    def compute_forwards(self, data_instances, model_weights, encrypted_calculator, batch_index=-1):
        """
        Compute guest side's fm forward, which is w*x + (v*x)^2 - (v^2)*(x^2)
        Parameters
        ----------
        data_instances: DTable of Instance, input data
        model_weights: FM Model weight
        encrypted_calculator: weight encrypt
        batch_index: int
        """
        # compute w*x + <vi,vj>xi*xj + b
        fm = self.compute_fm(data_instances, model_weights)
        en_fm = encrypted_calculator[batch_index].encrypt(fm)

        # Note that guest side's vx is not encrypted
        vx = self.compute_vx(data_instances, model_weights.embed_)
        en_vx = encrypted_calculator[batch_index].encrypt(vx)

        # (encrypted fm, vx)
        en_fm_join_vx = en_fm.join(vx, lambda wx_p_fm, _vx: (wx_p_fm, _vx))

        # (encrypted fm, vx, encrypted vx)
        guest_forward = en_fm_join_vx.join(en_vx, lambda e, _en_vx: (e[0], e[1], _en_vx))

        return guest_forward, fm, vx

    def compute_and_aggregate_forwards(self):
        """
        Here we compute guest forward and aggregated with host forward.
        Cross feature between host and guest is implemented.
        """
        assert(len(self.host_forwards) == 1, "Current FM only support single host party")
        # encrypted host forwards (en_wx_plus_fm, en_vx)
        host_forward = self.host_forwards[0]

        # Here we compute cross feature between host and guest
        # Guest forward includes: 1. en_wx_plus_fm 2. vx 3. en_vx
        # Aggregate_forward = en_wx_plus_fm(host) + en_wx_plus_fm(guest) + en_vx(host) * vx(guest)
        en_aggregate_forward = self.guest_forward.join(host_forward,
                                                       lambda g, h: g[0] + h[0] + np.dot(h[1], g[1]))
        self.aggregated_forwards = en_aggregate_forward

        return en_aggregate_forward

    def compute_loss(self, data_instances, n_iter_, batch_index, loss_norm=None):
        """
        Compute hetero-fm loss for:
        loss = (1/N)*∑(log2 - 1/2*y*f(x) + 1/8*f(x)^2), where y is label, f(x) is the fm model's predict.
        Note that "1/8*f(x)^2" are computed at Arbiter. 
        """
        current_suffix = (n_iter_, batch_index)
        # n = data_instances.count()
        n = self.aggregated_forwards.count()
        yfx = self.aggregated_forwards.join(data_instances, lambda fx, d: fx * int(d.label)).reduce(reduce_add)

        loss_list = []
        if loss_norm is not None:
            host_loss_regular = self.get_host_loss_regular(suffix=current_suffix)
            LOGGER.debug(f"iter {n_iter_} host loss regular {host_loss_regular}, guest loss regualr {loss_norm}")
        else:
            host_loss_regular = []

        # for host_idx, host_forward in enumerate(self.host_forwards):
        if len(self.host_forwards) > 1:
            LOGGER.info("More than one host exist, loss is not available")
        else:
            # (1/N)*∑(log2 - 1/2*y*f(x))
            loss = np.log(2) - 0.5 * (1 / n) * yfx

            # part 2 is sqrt(1/8)*f(x).
            self.remote_aggregated_forwards(self.aggregated_forwards, suffix=current_suffix)

            if loss_norm is not None:
                loss += loss_norm
                loss += host_loss_regular[0]
                LOGGER.debug(f"Reg loss is {loss_norm + host_loss_regular[0]}")
            loss_list.append(loss)

        LOGGER.debug(f"In compute_loss, loss list are: {loss_list}")
        self.sync_loss_info(loss_list, suffix=current_suffix)


class Host(BaseFM, loss_sync.Host):
    def __init__(self):
        self.forwards = None
        self.fore_gradient = None

    def _register_gradient_sync(self, host_forward_transfer, capped_fore_gradient_transfer,
                                host_gradient_transfer, host_optim_gradient_transfer,
                                agg_vx_mul_fg_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.capped_fore_gradient_transfer = capped_fore_gradient_transfer
        self.unilateral_gradient_transfer = host_gradient_transfer
        self.unilateral_optim_gradient_transfer = host_optim_gradient_transfer
        self.agg_vx_mul_fg_transfer = agg_vx_mul_fg_transfer

    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.host_forward_dict,
                                     transfer_variables.capped_fore_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.host_optim_gradient,
                                     transfer_variables.agg_vx_mul_fg)

        self._register_loss_sync(transfer_variables.host_loss_regular,
                                 transfer_variables.loss,
                                 transfer_variables.loss_intermediate)

    def compute_unilateral_gradient(self, data_instances, fore_gradient, model_weights, optimizer):
        raise NotImplementedError("Function should not be called here")

    def compute_gradient_procedure(self, data_instances, model_weights,
                                   encrypted_calculator, optimizer,
                                   n_iter_, batch_index):
        """
        FM model gradient procedure
        Step 1: get host forwards which differ from different algorithm
                For Factorization Machine: forwards = wx + <v,v>x,x + b
        """
        current_suffix = (n_iter_, batch_index)

        # encrypted host forwards (en_wx_plus_fm, en_vx)
        self.forwards, host_vx = self.compute_forwards(data_instances, model_weights, encrypted_calculator[batch_index])

        self.remote_host_forward(self.forwards, suffix=current_suffix)

        # receive from arbiter: capped_fore_gradient
        capped_fore_gradient = self.get_capped_fore_gradient(suffix=current_suffix)

        # receive from guest: capped_fore_gradient * agg_vx
        agg_vx_mul_fg = self.get_agg_vx_mul_fg(suffix=current_suffix)

        # compute guest gradient and loss
        x_square = data_instances.mapValues(lambda v: np.multiply(v.features, v.features))
        x_square_mul_fg = x_square.join(capped_fore_gradient, lambda xs, fg: xs * fg)

        # compuate gradient
        # en_capped_fore_gradient = capped_fore_gradient.mapValues(lambda v: encrypted_calculator.encrypt(v))
        en_capped_fore_gradient = encrypted_calculator[batch_index].encrypt(capped_fore_gradient)
        unilateral_gradient = compute_gradient(data_instances,
                                               en_capped_fore_gradient,
                                               agg_vx_mul_fg,
                                               x_square_mul_fg,
                                               model_weights)

        if optimizer is not None:
            unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, model_weights)

        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        return optimized_gradient

    def remote_host_forward(self, host_forward, suffix=tuple()):
        self.host_forward_transfer.remote(obj=host_forward, role=consts.GUEST, idx=0, suffix=suffix)

    def get_agg_vx_mul_fg(self, suffix=tuple()):
        agg_vx_mul_fg = self.agg_vx_mul_fg_transfer.get(idx=0, suffix=suffix)
        return agg_vx_mul_fg

    def get_capped_fore_gradient(self, suffix=tuple()):
        capped_fore_gradient = self.capped_fore_gradient_transfer.get(idx=0, suffix=suffix)
        LOGGER.info("Get capped fore gradient from arbiter.")
        return capped_fore_gradient

    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient

    def compute_forwards(self, data_instances, model_weights, encrypted_calculator):
        # w*x + <vi,vj>xi*xj + b
        fm_forward = self.compute_fm(data_instances, model_weights, fit_intercept=False)
        en_fm_forward = encrypted_calculator.encrypt(fm_forward)

        vx = self.compute_vx(data_instances, model_weights.embed_)
        en_vx = encrypted_calculator.encrypt(vx)

        self.raw_forwards = fm_forward
        host_forward = en_fm_forward.join(en_vx, lambda wx_p_fm, vx: (wx_p_fm, vx))
        return host_forward, vx

    def compute_loss(self, fm_weights, optimizer, n_iter_, batch_index):
        """
        Compute hetero-fm loss for:
        loss = (1/N)*∑(log2 - 1/2*y*f(x) + 1/8*f(x)^2), where y is label, f(x) is the fm model's predict.
        """
        current_suffix = (n_iter_, batch_index)
        loss_regular = optimizer.loss_norm(fm_weights)
        self.remote_loss_regular(loss_regular, suffix=current_suffix)


class Arbiter(loss_sync.Arbiter):
    def __init__(self):
        self.has_multiple_hosts = False

    def _register_gradient_sync(self, guest_gradient_transfer, host_gradient_transfer,
                                guest_optim_gradient_transfer, host_optim_gradient_transfer,
                                aggregated_forwards_transfer, capped_fore_gradient_transfer,
                                fore_gradient_transfer):
        self.guest_gradient_transfer = guest_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.guest_optim_gradient_transfer = guest_optim_gradient_transfer
        self.host_optim_gradient_transfer = host_optim_gradient_transfer
        self.aggregated_forwards_transfer = aggregated_forwards_transfer
        self.capped_fore_gradient_transfer = capped_fore_gradient_transfer
        self.fore_gradient_transfer = fore_gradient_transfer

    def register_gradient_procedure(self, transfer_variables):
        self._register_gradient_sync(transfer_variables.guest_gradient,
                                     transfer_variables.host_gradient,
                                     transfer_variables.guest_optim_gradient,
                                     transfer_variables.host_optim_gradient,
                                     transfer_variables.aggregated_forwards,
                                     transfer_variables.capped_fore_gradient,
                                     transfer_variables.fore_gradient)
        self._register_loss_sync(transfer_variables.loss)

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

        fore_gradient = self.get_fore_gradient(current_suffix)

        def decrypt_and_capped(v):
            v = cipher_operator.decrypt(v)
            v = max(v, -1.)
            v = min(v, 1.)
            return v

        capped_fore_gradient = fore_gradient.mapValues(lambda v: decrypt_and_capped(v))
        LOGGER.debug(f"Capped fore gradient count {capped_fore_gradient.count()}.")
        self.remote_capped_fore_gradient(capped_fore_gradient, current_suffix)

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

        LOGGER.debug("In arbiter compute_gradient_procedure, before apply grad: {}, size_list: {}".format(
            grad, size_list
        ))

        delta_grad = optimizer.apply_gradients(grad)

        LOGGER.debug("In arbiter compute_gradient_procedure, delta_grad: {}".format(
            delta_grad
        ))
        separate_optim_gradient = self.separate(delta_grad, size_list)
        LOGGER.debug("In arbiter compute_gradient_procedure, separated gradient: {}".format(
            separate_optim_gradient
        ))
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

    def get_aggregated_forwards(self, suffix=tuple()):
        aggregated_forwards = self.aggregated_forwards_transfer.get(idx=-1, suffix=suffix)
        LOGGER.info("Get aggregated_forwards from Guest")
        return aggregated_forwards

    def get_fore_gradient(self, suffix=tuple()):
        fore_gradient = self.fore_gradient_transfer.get(idx=0, suffix=suffix)
        LOGGER.info("Get fore_gradient from Guest")
        return fore_gradient

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

    def remote_capped_fore_gradient(self, capped_fore_gradient, suffix=tuple()):
        self.capped_fore_gradient_transfer.remote(obj=capped_fore_gradient,
                                                  role=consts.HOST,
                                                  idx=-1,
                                                  suffix=suffix)
        self.capped_fore_gradient_transfer.remote(obj=capped_fore_gradient,
                                                  role=consts.GUEST,
                                                  idx=-1,
                                                  suffix=suffix)

    def compute_loss(self, cipher, n_iter_, batch_index):
        """
        Compute hetero-fm loss for:
        loss = (1/N)*∑(log2 - 1/2*y*f(x) + 1/8*f(x)^2), where y is label, f(x) is fm model's prediction.
        """
        if self.has_multiple_hosts:
            LOGGER.info("Has more than one host, loss is not available")
            return []

        suffix = (n_iter_, batch_index)
        aggregated_forwards = self.get_aggregated_forwards(suffix=suffix)

        # compute "Loss part 2: 1/8*f(x)^2"
        def _decrypted_and_square(x):
            v = cipher.decrypt(x)
            return v * v
        loss_part2 = aggregated_forwards[0].mapValues(_decrypted_and_square).reduce(reduce_add) * (1. / 8.)
        loss_part2 /= aggregated_forwards[0].count()

        LOGGER.debug(f"Arbiter receive {aggregated_forwards}")

        current_suffix = (n_iter_, batch_index)
        loss_list = self.sync_loss_info(suffix=current_suffix)
        de_loss_list = cipher.decrypt_list(loss_list)
        de_loss_list = [loss_part1 + loss_part2 for loss_part1 in de_loss_list]

        return de_loss_list
