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
#
################################################################################
#
#
################################################################################

# =============================================================================
# 
# =============================================================================

from arch.api import session
from arch.api.utils import log_utils
import warnings
from federatedml.tree import XgboostCriterion
from federatedml.util import consts
from arch.api.utils.splitable import segment_transfer_enabled

LOGGER = log_utils.getLogger()

class SplitInfo(object):
    def __init__(self, sitename=consts.GUEST, best_fid=None, best_bid=None,
                 sum_grad=0, sum_hess=0, gain=None, missing_dir=1, left_sample_count=0):
        self.sitename = sitename
        self.best_fid = best_fid
        self.best_bid = best_bid
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.gain = gain
        self.missing_dir = missing_dir

    def __str__(self):
        return 'best_fid:{},best_bid{},sum_grad{},sum_hess{},gain{}'.format(self.best_fid, self.best_bid,
                                                                            self.sum_grad, self.sum_hess, self.gain)


class Splitter(object):
    def __init__(self, criterion_method, criterion_params=[0, 1], min_impurity_split=1e-2, min_sample_split=2,
                 min_leaf_node=1):
        LOGGER.info("splitter init!")
        if not isinstance(criterion_method, str):
            raise TypeError("criterion_method type should be str, but %s find" % (type(criterion_method).__name__))

        if criterion_method == "xgboost":
            if not criterion_params:
                self.criterion = XgboostCriterion()
            else:
                try:
                    reg_lambda = float(criterion_params[0])
                    self.criterion = XgboostCriterion(reg_lambda)
                except:
                    warnings.warn("criterion_params' first criterion_params should be numeric")
                    self.criterion = XgboostCriterion()

        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node

    def find_split_single_histogram_guest(self, histogram, valid_features, sitename, use_missing, zero_as_missing):
        best_fid = None
        best_gain = self.min_impurity_split - consts.FLOAT_ZERO
        best_bid = None
        best_sum_grad_l = None
        best_sum_hess_l = None
        missing_bin = 0
        if use_missing:
            missing_bin = 1
        
        # in default, missing value going to right
        missing_dir = 1

        for fid in range(len(histogram)):
            if valid_features[fid] is False:
                continue
            bin_num = len(histogram[fid])
            if bin_num == 0 + missing_bin:
                continue
            sum_grad = histogram[fid][bin_num - 1][0]
            sum_hess = histogram[fid][bin_num - 1][1]
            node_cnt = histogram[fid][bin_num - 1][2]

            if node_cnt < self.min_sample_split:
                break

            for bid in range(bin_num - missing_bin - 1):
                sum_grad_l = histogram[fid][bid][0]
                sum_hess_l = histogram[fid][bid][1]
                node_cnt_l = histogram[fid][bid][2]

                sum_grad_r = sum_grad - sum_grad_l
                sum_hess_r = sum_hess - sum_hess_l
                node_cnt_r = node_cnt - node_cnt_l

                if node_cnt_l >= self.min_leaf_node and node_cnt_r >= self.min_leaf_node:
                    gain = self.criterion.split_gain([sum_grad, sum_hess],
                                                     [sum_grad_l, sum_hess_l], [sum_grad_r, sum_hess_r])

                    if gain > self.min_impurity_split and gain > best_gain:

                        best_gain = gain
                        best_fid = fid
                        best_bid = bid
                        best_sum_grad_l = sum_grad_l
                        best_sum_hess_l = sum_hess_l
                        missing_dir = 1

                """ missing value handle: dispatch to left child"""
                if use_missing:
                    sum_grad_l += histogram[fid][-1][0] - histogram[fid][-2][0]
                    sum_hess_l += histogram[fid][-1][1] - histogram[fid][-2][1]
                    node_cnt_l += histogram[fid][-1][2] - histogram[fid][-2][2]

                    sum_grad_r -= histogram[fid][-1][0] - histogram[fid][-2][0]
                    sum_hess_r -= histogram[fid][-1][1] - histogram[fid][-2][1]
                    node_cnt_r -= histogram[fid][-1][2] - histogram[fid][-2][2]

                    if node_cnt_l >= self.min_leaf_node and node_cnt_r >= self.min_leaf_node:
                        gain = self.criterion.split_gain([sum_grad, sum_hess],
                                                         [sum_grad_l, sum_hess_l], [sum_grad_r, sum_hess_r])


                        if gain > self.min_impurity_split and gain > best_gain:
                            best_gain = gain
                            best_fid = fid
                            best_bid = bid
                            best_sum_grad_l = sum_grad_l
                            best_sum_hess_l = sum_hess_l
                            missing_dir = -1

        splitinfo = SplitInfo(sitename=sitename, best_fid=best_fid, best_bid=best_bid,
                              gain=best_gain, sum_grad=best_sum_grad_l, sum_hess=best_sum_hess_l,
                              missing_dir=missing_dir)

        return splitinfo

    def find_split(self, histograms, valid_features, partitions=1, sitename=consts.GUEST,
                   use_missing=False, zero_as_missing=False):
        LOGGER.info("splitter find split of raw data")
        histogram_table = session.parallelize(histograms, include_key=False, partition=partitions)
        splitinfo_table = histogram_table.mapValues(lambda sub_hist:
                                                    self.find_split_single_histogram_guest(sub_hist,
                                                                                           valid_features,
                                                                                           sitename,
                                                                                           use_missing,
                                                                                           zero_as_missing))

        tree_node_splitinfo = [None for i in range(len(histograms))]
        for id, splitinfo in splitinfo_table.collect():
            tree_node_splitinfo[id] = splitinfo

        # tree_node_splitinfo = [splitinfo[1] for splitinfo in splitinfo_table.collect()]

        return tree_node_splitinfo

    def find_split_single_histogram_host(self, fid_with_histogram, valid_features, sitename, use_missing=False,
                                         zero_as_missing=False):
        node_splitinfo = []
        node_grad_hess = []

        missing_bin = 0
        if use_missing:
            missing_bin = 1

        fid, histogram = fid_with_histogram
        if valid_features[fid] is False:
            return [], []
        bin_num = len(histogram)
        if bin_num == 0:
            return [], []

        node_cnt = histogram[bin_num - 1][2]

        if node_cnt < self.min_sample_split:
            return [], []

        for bid in range(bin_num - missing_bin - 1):
            sum_grad_l = histogram[bid][0]
            sum_hess_l = histogram[bid][1]
            node_cnt_l = histogram[bid][2]

            node_cnt_r = node_cnt - node_cnt_l

            if node_cnt_l >= self.min_leaf_node and node_cnt_r >= self.min_leaf_node:
                splitinfo = SplitInfo(sitename=sitename, best_fid=fid,
                                      best_bid=bid, sum_grad=sum_grad_l, sum_hess=sum_hess_l,
                                      missing_dir=1)

                node_splitinfo.append(splitinfo)
                node_grad_hess.append((sum_grad_l, sum_hess_l))

            if use_missing:
                sum_grad_l += histogram[-1][0] - histogram[-2][0]
                sum_hess_l += histogram[-1][1] - histogram[-2][1]
                node_cnt_l += histogram[-1][2] - histogram[-2][2]

                splitinfo = SplitInfo(sitename=sitename, best_fid=fid,
                                      best_bid=bid, sum_grad=sum_grad_l, sum_hess=sum_hess_l,
                                      missing_dir=-1)

                node_splitinfo.append(splitinfo)
                node_grad_hess.append((sum_grad_l, sum_hess_l))

        return node_splitinfo, node_grad_hess

    def find_split_host(self, histograms, valid_features, node_map, sitename=consts.HOST,
                        use_missing=False, zero_as_missing=False):
        LOGGER.info("splitter find split of host")
        tree_node_splitinfo = [[] for i in range(len(node_map))]
        encrypted_node_grad_hess = [[] for i in range(len(node_map))]
        host_splitinfo_table = histograms.mapValues(lambda fid_with_hist:
                                                    self.find_split_single_histogram_host(fid_with_hist, valid_features,
                                                                                          sitename,
                                                                                          use_missing,
                                                                                          zero_as_missing))

        for (nid, fid), splitinfo in host_splitinfo_table.collect():
            tree_node_splitinfo[nid].extend(splitinfo[0])
            encrypted_node_grad_hess[nid].extend(splitinfo[1])

        return tree_node_splitinfo, BigObjectTransfer(encrypted_node_grad_hess)

    def node_gain(self, grad, hess):
        return self.criterion.node_gain(grad, hess)

    def node_weight(self, grad, hess):
        return self.criterion.node_weight(grad, hess)

    def split_gain(self, sum_grad, sum_hess, sum_grad_l, sum_hess_l, sum_grad_r, sum_hess_r):
        gain = self.criterion.split_gain([sum_grad, sum_hess], \
                                         [sum_grad_l, sum_hess_l], [sum_grad_r, sum_hess_r])
        return gain


class BigObjectTransfer(metaclass=segment_transfer_enabled()):
    def __init__(self, data):
        self._obj = data

    def get_data(self):
        return self._obj
