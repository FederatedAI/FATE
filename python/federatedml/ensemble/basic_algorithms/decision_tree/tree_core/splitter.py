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
import numpy as np
import warnings
import functools
import random
from fate_arch.session import computing_session as session
from fate_arch.common import log
from fate_arch.federation import segment_transfer_enabled
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.criterion import XgboostCriterion
from federatedml.util import consts

LOGGER = log.getLogger()


class SplitInfo(object):
    def __init__(self, sitename=consts.GUEST, best_fid=None, best_bid=None,
                 sum_grad=0, sum_hess=0, gain=None, missing_dir=1, mask_id=None, sample_count=-1):
        self.sitename = sitename
        self.best_fid = best_fid
        self.best_bid = best_bid
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.gain = gain
        self.missing_dir = missing_dir
        self.mask_id = mask_id
        self.sample_count = sample_count

    def __str__(self):
        return '(fid {} bid {}, sum_grad {}, sum_hess {}, gain {}, sitename {}, missing dir {}, mask_id {}, ' \
               'sample_count {})\n'.format(
                   self.best_fid, self.best_bid, self.sum_grad, self.sum_hess, self.gain, self.sitename, self.missing_dir,
                   self.mask_id, self.sample_count)

    def __repr__(self):
        return self.__str__()


class Splitter(object):

    def __init__(self, criterion_method, criterion_params=[0, 0], min_impurity_split=1e-2, min_sample_split=2,
                 min_leaf_node=1, min_child_weight=1):

        LOGGER.info("splitter init!")
        if not isinstance(criterion_method, str):
            raise TypeError("criterion_method type should be str, but %s find" % (type(criterion_method).__name__))

        if criterion_method == "xgboost":
            if not criterion_params:
                self.criterion = XgboostCriterion()
            else:
                try:
                    reg_lambda, reg_alpha = 0, 0
                    if isinstance(criterion_params, list):
                        reg_lambda = float(criterion_params[0])
                        reg_alpha = float(criterion_params[1])
                    self.criterion = XgboostCriterion(reg_lambda=reg_lambda, reg_alpha=reg_alpha)
                except BaseException:
                    warnings.warn("criterion_params' first criterion_params should be numeric")
                    self.criterion = XgboostCriterion()

        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node
        self.min_child_weight = min_child_weight

    def _check_min_child_weight(self, l_h, r_h):

        if isinstance(l_h, np.ndarray):
            l_h, r_h = np.sum(l_h), np.sum(r_h)
        rs = l_h >= self.min_child_weight and r_h >= self.min_child_weight
        return rs

    def _check_sample_num(self, l_cnt, r_cnt):
        return l_cnt >= self.min_leaf_node and r_cnt >= self.min_leaf_node

    def find_split_single_histogram_guest(self, histogram, valid_features, sitename, use_missing, zero_as_missing,
                                          reshape_tuple=None):

        if reshape_tuple:
            histogram = histogram.reshape(reshape_tuple)

        # default values
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

            # last bin contains sum values (cumsum from left)
            sum_grad = histogram[fid][bin_num - 1][0]
            sum_hess = histogram[fid][bin_num - 1][1]
            node_cnt = histogram[fid][bin_num - 1][2]

            if node_cnt < self.min_sample_split:
                break

            if node_cnt < 1:  # avoid float error
                break

            # last bin will not participate in split find, so bin_num - 1
            for bid in range(bin_num - missing_bin - 1):

                # left gh
                sum_grad_l = histogram[fid][bid][0]
                sum_hess_l = histogram[fid][bid][1]
                node_cnt_l = histogram[fid][bid][2]
                # right gh
                sum_grad_r = sum_grad - sum_grad_l
                sum_hess_r = sum_hess - sum_hess_l
                node_cnt_r = node_cnt - node_cnt_l

                if self._check_min_child_weight(sum_hess_l, sum_hess_r) and self._check_sample_num(node_cnt_l,
                                                                                                   node_cnt_r):
                    gain = self.criterion.split_gain([sum_grad, sum_hess],
                                                     [sum_grad_l, sum_hess_l], [sum_grad_r, sum_hess_r])

                    if gain > self.min_impurity_split and gain > best_gain + consts.FLOAT_ZERO:
                        best_gain = gain
                        best_fid = fid
                        best_bid = bid
                        best_sum_grad_l = sum_grad_l
                        best_sum_hess_l = sum_hess_l
                        missing_dir = 1

                """ missing value handle: dispatch to left child"""
                if use_missing:

                    # add sum of samples with missing features to left
                    sum_grad_l += histogram[fid][-1][0] - histogram[fid][-2][0]
                    sum_hess_l += histogram[fid][-1][1] - histogram[fid][-2][1]
                    node_cnt_l += histogram[fid][-1][2] - histogram[fid][-2][2]

                    sum_grad_r -= histogram[fid][-1][0] - histogram[fid][-2][0]
                    sum_hess_r -= histogram[fid][-1][1] - histogram[fid][-2][1]
                    node_cnt_r -= histogram[fid][-1][2] - histogram[fid][-2][2]

                    # if have a better gain value, missing dir is left
                    if self._check_sample_num(node_cnt_l, node_cnt_r) and self._check_min_child_weight(sum_hess_l,
                                                                                                       sum_hess_r):

                        gain = self.criterion.split_gain([sum_grad, sum_hess],
                                                         [sum_grad_l, sum_hess_l], [sum_grad_r, sum_hess_r])

                        if gain > self.min_impurity_split and gain > best_gain + consts.FLOAT_ZERO:
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

    def construct_feature_split_points(self, fid_with_histogram, valid_features, sitename, use_missing,
                                       left_missing_dir, right_missing_dir, mask_id_mapping):

        feature_split_info = []
        missing_bin = 0
        if use_missing:
            missing_bin = 1

        fid, histogram = fid_with_histogram
        if valid_features[fid] is False:
            return [], None
        bin_num = len(histogram)
        if bin_num == 0:
            return [], None

        node_cnt = histogram[bin_num - 1][2]

        if node_cnt < self.min_sample_split:
            return [], None

        for bid in range(bin_num - missing_bin - 1):
            sum_grad_l = histogram[bid][0]
            sum_hess_l = histogram[bid][1]
            node_cnt_l = histogram[bid][2]

            node_cnt_r = node_cnt - node_cnt_l
            mask_id = mask_id_mapping[(fid, bid)]
            if self._check_sample_num(node_cnt_l, node_cnt_r):

                missing_dir = np.random.choice(right_missing_dir)
                splitinfo = SplitInfo(sitename=sitename, sum_grad=sum_grad_l, sum_hess=sum_hess_l,
                                      missing_dir=missing_dir, mask_id=mask_id, sample_count=node_cnt_l)  # 1
                feature_split_info.append(splitinfo)

                if use_missing:
                    sum_grad_l += histogram[-1][0] - histogram[-2][0]
                    sum_hess_l += histogram[-1][1] - histogram[-2][1]
                    node_cnt_l += histogram[-1][2] - histogram[-2][2]
                    missing_dir = np.random.choice(left_missing_dir)
                    splitinfo = SplitInfo(sitename=sitename, sum_grad=sum_grad_l, sum_hess=sum_hess_l,
                                          missing_dir=missing_dir, mask_id=mask_id, sample_count=node_cnt_l)  # -1
                    feature_split_info.append(splitinfo)

        # split info contains g/h sum and node cnt
        g_sum, h_sum = histogram[-1][0], histogram[-1][1]
        g_h_sum_info = SplitInfo(sum_grad=g_sum, sum_hess=h_sum, sample_count=node_cnt)

        return feature_split_info, g_h_sum_info

    def construct_feature_split_points_batches(self, kv_iter, valid_features, sitename,
                                               use_missing, mask_id_mapping, left_missing_dir,
                                               right_missing_dir, batch_size,
                                               cipher_compressor=None,
                                               shuffle_random_seed=None):

        result_list = []
        split_info_dict = {}
        g_h_sum_dict = {}
        partition_key = None
        for key, value in kv_iter:

            nid, fid = key
            if partition_key is None:
                partition_key = str((nid, fid))

            split_info_list, g_h_sum_info = self.construct_feature_split_points(value, valid_features, sitename,
                                                                                use_missing,
                                                                                left_missing_dir, right_missing_dir,
                                                                                mask_id_mapping)
            # collect all splitinfo of a node
            if nid not in split_info_dict:
                split_info_dict[nid] = []
            split_info_dict[nid] += split_info_list

            if nid not in g_h_sum_dict:
                if g_h_sum_info is not None:
                    g_h_sum_dict[nid] = g_h_sum_info

        # cut split info into batches
        for nid in split_info_dict:

            split_info_list = split_info_dict[nid]
            if len(split_info_list) == 0:
                result_list.append(
                    ((nid, partition_key + '-empty'), []))  # add an empty split info list if no split info available
                continue

            if shuffle_random_seed:
                random.seed(shuffle_random_seed)
                random.shuffle(split_info_list)
                # LOGGER.debug('nid {} mask id list {}'.format(nid, shuffle_list))

            LOGGER.debug('split info len is {}'.format(len(split_info_list)))

            batch_start_idx = range(0, len(split_info_list), batch_size)
            batch_idx = 0
            for i in batch_start_idx:
                key = (nid, (partition_key + '-{}'.format(batch_idx)))  # nid, batch_id
                batch_idx += 1
                g_h_sum_info = g_h_sum_dict[nid]
                batch_split_info_list = split_info_list[i: i + batch_size]
                # compress ciphers
                if cipher_compressor is not None:
                    compressed_packages = cipher_compressor.compress_split_info(batch_split_info_list, g_h_sum_info)
                    result_list.append((key, (nid, compressed_packages)))
                else:
                    result_list.append((key, (batch_split_info_list, g_h_sum_info)))

        return result_list

    def _find_host_best_splits_map_func(self, value, decrypter, gh_packer=None,
                                        host_sitename=consts.HOST):

        # find best split points in a node for every host feature, mapValues function
        best_gain = self.min_impurity_split - consts.FLOAT_ZERO
        best_idx = -1
        best_split_info = SplitInfo(sitename=host_sitename, best_fid=-1, best_bid=-1, gain=best_gain,
                                    mask_id=-1)

        if len(value) == 0:  # this node can not be further split, because split info list is empty
            return best_idx, best_split_info

        if gh_packer is None:
            split_info_list, g_h_info = value
            for split_info in split_info_list:
                split_info.sum_grad, split_info.sum_hess = decrypter.decrypt(split_info.sum_grad), decrypter.decrypt(
                    split_info.sum_hess)
            g_sum, h_sum = decrypter.decrypt(g_h_info.sum_grad), decrypter.decrypt(g_h_info.sum_hess)
        else:
            nid, package = value
            split_info_list = gh_packer.decompress_and_unpack(package)
            g_sum, h_sum = split_info_list[-1].sum_grad, split_info_list[-1].sum_hess  # g/h sum is at last index
            split_info_list = split_info_list[:-1]

        for idx, split_info in enumerate(split_info_list):

            l_g, l_h = split_info.sum_grad, split_info.sum_hess
            r_g, r_h = g_sum - l_g, h_sum - l_h
            gain = self.split_gain(g_sum, h_sum, l_g, l_h, r_g, r_h)

            if self._check_min_child_weight(l_h, r_h) and \
                    gain > self.min_impurity_split and gain > best_gain + consts.FLOAT_ZERO:
                new_split_info = SplitInfo(sitename=host_sitename, best_fid=split_info.best_fid,
                                           best_bid=split_info.best_bid, gain=gain,
                                           sum_grad=l_g, sum_hess=l_h, missing_dir=split_info.missing_dir,
                                           mask_id=split_info.mask_id)
                best_gain = gain
                best_idx = idx
                best_split_info = new_split_info

        best_split_info.gain = best_gain

        return best_idx, best_split_info

    @staticmethod
    def key_sort_func(a, b):
        key_1, key_2 = a[0], b[0]
        if key_1[0] == key_2[0]:
            if key_1[1] > key_2[1]:
                return 1
            else:
                return -1
        else:
            if key_1[0] > key_2[0]:
                return 1
            else:
                return -1

    def find_host_best_split_info(self, host_split_info_table, host_sitename, decrypter, gh_packer=None):

        map_func = functools.partial(self._find_host_best_splits_map_func,
                                     decrypter=decrypter,
                                     host_sitename=host_sitename,
                                     gh_packer=gh_packer
                                     )

        host_feature_best_split_table = host_split_info_table.mapValues(map_func)
        feature_best_splits = list(host_feature_best_split_table.collect())
        sorted_list = sorted(feature_best_splits, key=functools.cmp_to_key(self.key_sort_func))

        node_best_splits = {}
        for key, result in sorted_list:
            node_id, fid = key
            best_idx, split_info = result
            if node_id not in node_best_splits:
                node_best_splits[node_id] = SplitInfo(sitename=host_sitename, best_bid=-1, best_fid=-1,
                                                      gain=self.min_impurity_split - consts.FLOAT_ZERO)
            if best_idx == -1:
                continue
            elif split_info.gain > self.min_impurity_split and split_info.gain > node_best_splits[node_id].gain \
                    + consts.FLOAT_ZERO:
                node_best_splits[node_id] = split_info

        return node_best_splits

    def host_prepare_split_points(self, histograms, valid_features, mask_id_mapping, use_missing, left_missing_dir,
                                  right_missing_dir, sitename=consts.HOST, batch_size=consts.MAX_SPLITINFO_TO_COMPUTE,
                                  cipher_compressor=None, shuffle_random_seed=None):

        LOGGER.info("splitter find split of host")
        LOGGER.debug('missing dir mask dict {}, {}'.format(left_missing_dir, right_missing_dir))

        map_partition_func = functools.partial(self.construct_feature_split_points_batches,
                                               valid_features=valid_features,
                                               sitename=sitename,
                                               use_missing=use_missing,
                                               left_missing_dir=left_missing_dir,
                                               right_missing_dir=right_missing_dir,
                                               mask_id_mapping=mask_id_mapping,
                                               batch_size=batch_size,
                                               cipher_compressor=cipher_compressor,
                                               shuffle_random_seed=shuffle_random_seed
                                               )

        host_splitinfo_table = histograms.mapPartitions(map_partition_func, use_previous_behavior=False)

        return host_splitinfo_table

    def find_split_host(self, histograms, valid_features, node_map, sitename=consts.HOST,
                        use_missing=False, zero_as_missing=False):
        LOGGER.info("splitter find split of host")
        LOGGER.debug('node map len is {}'.format(len(node_map)))
        tree_node_splitinfo = [[] for i in range(len(node_map))]
        encrypted_node_grad_hess = [[] for i in range(len(node_map))]
        host_splitinfo_table = histograms.mapValues(lambda fid_with_hist:
                                                    self.find_split_single_histogram_host(fid_with_hist, valid_features,
                                                                                          sitename,
                                                                                          use_missing,
                                                                                          zero_as_missing))

        # node_id, map it to node index
        for (idx, fid), splitinfo in host_splitinfo_table.collect():
            idx = node_map[idx]
            tree_node_splitinfo[idx].extend(splitinfo[0])
            encrypted_node_grad_hess[idx].extend(splitinfo[1])

        return tree_node_splitinfo, BigObjectTransfer(encrypted_node_grad_hess)

    def node_gain(self, grad, hess):
        return self.criterion.node_gain(grad, hess)

    def node_weight(self, grad, hess):
        return self.criterion.node_weight(grad, hess)

    def split_gain(self, sum_grad, sum_hess, sum_grad_l, sum_hess_l, sum_grad_r, sum_hess_r):
        gain = self.criterion.split_gain([sum_grad, sum_hess],
                                         [sum_grad_l, sum_hess_l], [sum_grad_r, sum_hess_r])
        return gain


class BigObjectTransfer(metaclass=segment_transfer_enabled()):
    def __init__(self, data):
        self._obj = data

    def get_data(self):
        return self._obj
