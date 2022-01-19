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
# FeatureHistogram
# =============================================================================
import copy
import functools
import numpy as np
from operator import add, sub
import scipy.sparse as sp
import uuid
from typing import List
from fate_arch.session import computing_session as session
from fate_arch.common import log
from federatedml.feature.fate_element_type import NoneType
from federatedml.framework.weights import Weights

LOGGER = log.getLogger()

# ret type
TENSOR = 'tensor'
TABLE = 'tb'


class HistogramBag(object):
    """
    holds histograms
    """

    def __init__(self, tensor: list, hid: int = -1, p_hid: int = -1):
        """
        :param tensor: list returned by calculate_histogram
        :param hid: histogram id
        :param p_hid: parent node histogram id
        :param tensor_type: 'list' or 'array'
        """

        self.hid = hid
        self.p_hid = p_hid
        self.bag = tensor
        self.tensor_type = type(self.bag)

    def binary_op(self, other, func, inplace=False):

        assert isinstance(other, HistogramBag)
        assert len(self.bag) == len(other)

        bag = self.bag
        newbag = None
        if not inplace:
            newbag = copy.deepcopy(other)
            bag = newbag.bag

        for bag_idx in range(len(self.bag)):
            for hist_idx in range(len(self.bag[bag_idx])):
                bag[bag_idx][hist_idx][0] = func(self.bag[bag_idx][hist_idx][0], other[bag_idx][hist_idx][0])
                bag[bag_idx][hist_idx][1] = func(self.bag[bag_idx][hist_idx][1], other[bag_idx][hist_idx][1])
                bag[bag_idx][hist_idx][2] = func(self.bag[bag_idx][hist_idx][2], other[bag_idx][hist_idx][2])

        return self if inplace else newbag

    def __add__(self, other):
        if self.tensor_type == 'list':
            return self.binary_op(other, add, inplace=False)
        elif self.tensor_type == 'array':
            self.bag += other.bag
            return self
        else:
            raise ValueError('unknown tensor type')

    def __sub__(self, other):
        if self.tensor_type == list:
            return self.binary_op(other, sub, inplace=False)
        elif self.tensor_type == np.ndarray:
            self.bag -= other.bag
            return self
        else:
            raise ValueError('unknown tensor type')

    def __len__(self):
        return len(self.bag)

    def __getitem__(self, item):
        return self.bag[item]

    def __str__(self):
        return str(self.bag)

    def __repr__(self):
        return str(self.bag)


class FeatureHistogramWeights(Weights):

    def __init__(self, list_of_histogram_bags: List[HistogramBag]):

        self.hists = list_of_histogram_bags
        super(FeatureHistogramWeights, self).__init__(l=list_of_histogram_bags)

    def map_values(self, func, inplace):

        if inplace:
            hists = self.hists
        else:
            hists = copy.deepcopy(self.hists)

        for histbag in hists:
            bag = histbag.bag
            for component_idx in range(len(bag)):
                for hist_idx in range(len(bag[component_idx])):
                    bag[component_idx][hist_idx][0] = func(bag[component_idx][hist_idx][0])
                    bag[component_idx][hist_idx][1] = func(bag[component_idx][hist_idx][1])
                    bag[component_idx][hist_idx][2] = func(bag[component_idx][hist_idx][2])

        if inplace:
            return self
        else:
            return FeatureHistogramWeights(list_of_histogram_bags=hists)

    def binary_op(self, other: 'FeatureHistogramWeights', func, inplace: bool):

        new_weights = []
        hists, other_hists = self.hists, other.hists
        for h1, h2 in zip(hists, other_hists):
            rnt = h1.binary_op(h2, func, inplace=inplace)
            if not inplace:
                new_weights.append(rnt)

        if inplace:
            return self
        else:
            return FeatureHistogramWeights(new_weights)

    def axpy(self, a, y: 'FeatureHistogramWeights'):

        def func(x1, x2): return x1 + a * x2
        self.binary_op(y, func, inplace=True)

        return self

    def __iter__(self):
        pass

    def __str__(self):
        return str([str(hist) for hist in self.hists])

    def __repr__(self):
        return str(self.hists)


class FeatureHistogram(object):

    def __init__(self):

        self._cur_to_split_node_info = {}
        self._prev_layer_cached_histograms = {}
        self._cur_layer_cached_histograms = {}
        self._cur_dep = -1
        self._prev_layer_dtable = None
        self._cur_layer_dtables = [None]
        self.stable_reduce = False

    """
    Public Interface for Histogram Computation
    """

    def compute_histogram(self, dep, data_bin, grad_and_hess, bin_split_points, bin_sparse_points,
                          valid_features, node_map,
                          node_sample_count,
                          use_missing=False,
                          zero_as_missing=False,
                          ret="tensor",
                          hist_sub=True,
                          cur_to_split_nodes=None
                          ):
        """
        This the new interface for histogram computation
        """

        if hist_sub:

            # if run histogram subtraction, need to trim node map, and get parent/sibling node info for computation
            LOGGER.info('get histogram using histogram subtraction')
            self._update_node_info(cur_to_split_nodes)
            to_compute_node_map, sibling_node_id_map = self._trim_node_map(node_map, node_sample_count)
            parent_node_id_map = self._get_parent_nid_map()
            LOGGER.debug('histogram subtraction at dep {}, new node map is {}, sibling node map is {}, '
                         'cur to split node info is {}, parent node id map is {}'.
                         format(dep, to_compute_node_map, sibling_node_id_map, self._cur_to_split_node_info,
                                parent_node_id_map))
        else:
            # else use original node map
            to_compute_node_map = node_map
            sibling_node_id_map = None
            parent_node_id_map = None

        if ret == TENSOR:

            histograms = self.calculate_histogram(data_bin, grad_and_hess,
                                                  bin_split_points, bin_sparse_points,
                                                  valid_features, to_compute_node_map,
                                                  use_missing, zero_as_missing, ret=ret)

            if not hist_sub:
                return histograms

            # running hist sub
            self._update_cached_histograms(dep, ret=ret)
            if self._is_root_node(node_map):  # root node need no hist sub
                self._cur_layer_cached_histograms[0] = histograms[0]
                result = histograms
            else:
                node_id_list, result = self._tensor_subtraction(histograms, to_compute_node_map)
                self._cached_histograms((node_id_list, result), ret=ret)
            return result

        elif ret == 'tb':

            LOGGER.debug('maps are {} {}'.format(parent_node_id_map, sibling_node_id_map))

            LOGGER.info('computing histogram table using normal mode')
            histogram_table = self.calculate_histogram(data_bin, grad_and_hess,
                                                       bin_split_points, bin_sparse_points,
                                                       valid_features, to_compute_node_map,
                                                       use_missing, zero_as_missing,
                                                       ret=ret,
                                                       parent_node_id_map=parent_node_id_map,
                                                       sibling_node_id_map=sibling_node_id_map)

            if not hist_sub:
                return histogram_table

            # running hist sub
            self._update_cached_histograms(dep, ret=ret)
            if self._is_root_node(node_map):  # root node need not hist sub
                self._cur_layer_dtables.append(histogram_table)
                result = histogram_table
            else:
                result = self._table_subtraction(histogram_table)
                self._cached_histograms(result, ret=ret)
            return result

    def calculate_histogram(self, data_bin, grad_and_hess,
                            bin_split_points, bin_sparse_points,
                            valid_features=None,
                            node_map=None,
                            use_missing=False,
                            zero_as_missing=False,
                            parent_node_id_map=None,
                            sibling_node_id_map=None,
                            ret=TENSOR):
        """
        This is the old interface for histogram computation

        data_bin: data after binning with node positions
        grad_and_hess: g/h for each sample
        bin_split_points: split points
        bin_sparse_points: sparse points
        node_map: node id to node index
        use_missing: enable use missing
        zero_as_missing: enable zero as missing
        parent_node_id_map: map current node_id to its parent id, this para is for hist sub
        sibling_node_id_map: map current node_id to its sibling id, this para is for hist sub
        ret: return type, if 'tb', return histograms stored in Table
        """

        LOGGER.debug("bin_shape is {}, node num is {}".format(bin_split_points.shape, len(node_map)))

        if grad_and_hess.count() == 0:
            raise ValueError('input grad and hess is empty')

        # histogram template will be adjusted when running mo tree
        mo_dim = None
        g_h_example = grad_and_hess.take(1)
        if isinstance(g_h_example[0][1][0], np.ndarray) and len(g_h_example[0][1][0]) > 1:
            mo_dim = len(g_h_example[0][1][0])

        # reformat, now format is: key, ((data_instance, node position), (g, h))
        batch_histogram_intermediate_rs = data_bin.join(grad_and_hess, lambda data_inst, g_h: (data_inst, g_h))

        if batch_histogram_intermediate_rs.count() == 0:  # if input sample number is 0, return empty histograms

            node_histograms = FeatureHistogram._generate_histogram_template(node_map, bin_split_points, valid_features,
                                                                            1 if use_missing else 0, mo_dim=mo_dim)
            hist_list = FeatureHistogram._generate_histogram_key_value_list(node_histograms, node_map, bin_split_points,
                                                                            parent_node_id_map=parent_node_id_map,
                                                                            sibling_node_id_map=sibling_node_id_map)

            if ret == TENSOR:
                feature_num = bin_split_points.shape[0]
                return FeatureHistogram._recombine_histograms(hist_list, node_map, feature_num)
            else:
                histograms_table = session.parallelize(hist_list, partition=data_bin.partitions, include_key=True)
                return FeatureHistogram._construct_table(histograms_table)

        else:  # compute histograms

            batch_histogram_cal = functools.partial(
                FeatureHistogram._batch_calculate_histogram,
                bin_split_points=bin_split_points, bin_sparse_points=bin_sparse_points,
                valid_features=valid_features, node_map=node_map,
                use_missing=use_missing, zero_as_missing=zero_as_missing,
                parent_nid_map=parent_node_id_map,
                sibling_node_id_map=sibling_node_id_map,
                stable_reduce=self.stable_reduce,
                mo_dim=mo_dim
            )

            agg_func = self._stable_hist_aggregate if self.stable_reduce else self._hist_aggregate
            histograms_table = batch_histogram_intermediate_rs.mapReducePartitions(batch_histogram_cal, agg_func)
            if self.stable_reduce:
                histograms_table = histograms_table.mapValues(self._stable_hist_reduce)

            if ret == "tensor":
                feature_num = bin_split_points.shape[0]
                histogram_list = list(histograms_table.collect())
                rs = FeatureHistogram._recombine_histograms(histogram_list, node_map, feature_num)
                return rs
            else:
                return FeatureHistogram._construct_table(histograms_table)

    """
    Histogram computation functions
    """

    @staticmethod
    def _tensor_histogram_cumsum(histograms):
        # histogram cumsum, from left to right
        for i in range(1, len(histograms)):
            for j in range(len(histograms[i])):
                histograms[i][j] += histograms[i - 1][j]

        return histograms

    @staticmethod
    def _dtable_histogram_cumsum(histograms):

        # histogram cumsum, from left to right
        if len(histograms) == 0:
            return histograms

        new_hist = [[0, 0, 0] for i in range(len(histograms))]
        new_hist[0][0] = copy.deepcopy(histograms[0][0])
        new_hist[0][1] = copy.deepcopy(histograms[0][1])
        new_hist[0][2] = copy.deepcopy(histograms[0][2])

        for i in range(1, len(histograms)):
            # ciphertext cumsum skipping
            if histograms[i][2] == 0:
                new_hist[i] = new_hist[i - 1]
                continue

            for j in range(len(histograms[i])):
                new_hist[i][j] = new_hist[i - 1][j] + histograms[i][j]

        return new_hist

    @staticmethod
    def _host_histogram_cumsum_map_func(v):

        fid, histograms = v
        new_value = (fid, FeatureHistogram._dtable_histogram_cumsum(histograms))
        return new_value

    @staticmethod
    def _hist_aggregate(fid_histogram1, fid_histogram2):
        # add histograms with same key((node id, feature id)) together
        fid_1, histogram1 = fid_histogram1
        fid_2, histogram2 = fid_histogram2
        aggregated_res = [[] for i in range(len(histogram1))]
        for i in range(len(histogram1)):
            for j in range(len(histogram1[i])):
                aggregated_res[i].append(histogram1[i][j] + histogram2[i][j])

        return fid_1, aggregated_res

    @staticmethod
    def _stable_hist_aggregate(fid_histogram1, fid_histogram2):

        partition_id_list_1, hist_val_list_1 = fid_histogram1
        partition_id_list_2, hist_val_list_2 = fid_histogram2
        value = [partition_id_list_1 + partition_id_list_2, hist_val_list_1 + hist_val_list_2]
        return value

    @staticmethod
    def _stable_hist_reduce(value):

        # [partition1, partition2, ...], [(fid, hist), (fid, hist) .... ]
        partition_id_list, hist_list = value
        order = np.argsort(partition_id_list)
        aggregated_hist = None
        for idx in order:  # make sure reduce in order to avoid float error
            hist = hist_list[idx]
            if aggregated_hist is None:
                aggregated_hist = hist
                continue
            aggregated_hist = FeatureHistogram._hist_aggregate(aggregated_hist, hist)
        return aggregated_hist

    @staticmethod
    def _generate_histogram_template(node_map: dict, bin_split_points: np.ndarray, valid_features: dict,
                                     missing_bin, mo_dim=None):

        # for every feature, generate histograms containers (initialized val are 0s)
        node_num = len(node_map)
        node_histograms = []
        for k in range(node_num):
            feature_histogram_template = []
            for fid in range(bin_split_points.shape[0]):
                # if is not valid features, skip generating
                if valid_features is not None and valid_features[fid] is False:
                    feature_histogram_template.append([])
                    continue
                else:
                    # 0, 0, 0 -> grad, hess, sample count
                    if mo_dim:
                        feature_histogram_template.append([[np.zeros(mo_dim), np.zeros(mo_dim), 0]
                                                           for j in
                                                           range(bin_split_points[fid].shape[0] + missing_bin)])
                    else:
                        feature_histogram_template.append([[0, 0, 0]
                                                           for j in
                                                           range(bin_split_points[fid].shape[0] + missing_bin)])

            node_histograms.append(feature_histogram_template)
            # check feature num
            assert len(feature_histogram_template) == bin_split_points.shape[0]

        return node_histograms

    @staticmethod
    def _generate_histogram_key_value_list(node_histograms, node_map, bin_split_points, parent_node_id_map,
                                           sibling_node_id_map, partition_key=None):

        # generate key_value hist list for Table parallelization
        ret = []
        inverse_map = FeatureHistogram._inverse_node_map(node_map)
        for node_idx in range(len(node_map)):
            for fid in range(bin_split_points.shape[0]):
                # key: (nid, fid), value: (fid, hist)
                # if parent_nid is offered, map nid to its parent nid for histogram subtraction
                node_id = inverse_map[node_idx]
                key = (parent_node_id_map[node_id], fid) if parent_node_id_map is not None else (node_id, fid)
                # if sibling_node_id_map is offered, recorded its sibling ids for histogram subtraction
                value = (fid, node_histograms[node_idx][fid]) if sibling_node_id_map is None else \
                    ((fid, node_id, sibling_node_id_map[node_id]), node_histograms[node_idx][fid])
                if partition_key is not None:
                    value = [[partition_key], [value]]
                ret.append((key, value))

        return ret

    @staticmethod
    def _batch_calculate_histogram(kv_iterator, bin_split_points=None,
                                   bin_sparse_points=None, valid_features=None,
                                   node_map=None, use_missing=False, zero_as_missing=False,
                                   parent_nid_map=None, sibling_node_id_map=None, stable_reduce=False,
                                   mo_dim=None):
        data_bins = []
        node_ids = []
        grad = []
        hess = []

        data_record = 0  # total instance number of this partition

        partition_key = None  # this var is for stable reduce

        # go through iterator to collect g/h feature instances/ node positions
        for data_id, value in kv_iterator:

            if partition_key is None and stable_reduce:  # first key of data is used as partition key
                partition_key = data_id

            data_bin, nodeid_state = value[0]
            unleaf_state, nodeid = nodeid_state
            if unleaf_state == 0 or nodeid not in node_map:
                continue
            g, h = value[1]  # encrypted text in host, plaintext in guest
            data_bins.append(data_bin)  # features
            node_ids.append(nodeid)  # current node position
            grad.append(g)
            hess.append(h)
            data_record += 1

        LOGGER.debug("begin batch calculate histogram, data count is {}".format(data_record))
        node_num = len(node_map)

        missing_bin = 1 if use_missing else 0

        # if the value of a feature is 0, the corresponding bin index will not appear in the sample sparse vector
        # need to compute correct sparse point g_sum and s_sum by:
        # (node total sum value) - (node feature total sum value) + (non 0 sparse point sum)
        # [0, 0, 0] -> g, h, sample count
        zero_optim = [[[0 for i in range(3)]
                       for j in range(bin_split_points.shape[0])]
                      for k in range(node_num)]
        zero_opt_node_sum = [[0 for i in range(3)]
                             for j in range(node_num)]

        node_histograms = FeatureHistogram._generate_histogram_template(node_map, bin_split_points, valid_features,
                                                                        missing_bin, mo_dim=mo_dim)

        for rid in range(data_record):

            # node index is the position in the histogram list of a certain node
            node_idx = node_map.get(node_ids[rid])
            # node total sum value
            zero_opt_node_sum[node_idx][0] += grad[rid]
            zero_opt_node_sum[node_idx][1] += hess[rid]
            zero_opt_node_sum[node_idx][2] += 1

            for fid, value in data_bins[rid].features.get_all_data():
                if valid_features is not None and valid_features[fid] is False:
                    continue

                if use_missing and value == NoneType():
                    # missing value is set as -1
                    value = -1

                node_histograms[node_idx][fid][value][0] += grad[rid]
                node_histograms[node_idx][fid][value][1] += hess[rid]
                node_histograms[node_idx][fid][value][2] += 1

        for nid in range(node_num):
            # cal feature level g_h incrementally
            for fid in range(bin_split_points.shape[0]):
                if valid_features is not None and valid_features[fid] is False:
                    continue
                for bin_index in range(len(node_histograms[nid][fid])):
                    zero_optim[nid][fid][0] += node_histograms[nid][fid][bin_index][0]
                    zero_optim[nid][fid][1] += node_histograms[nid][fid][bin_index][1]
                    zero_optim[nid][fid][2] += node_histograms[nid][fid][bin_index][2]

        for node_idx in range(node_num):
            for fid in range(bin_split_points.shape[0]):
                if valid_features is not None and valid_features[fid] is True:
                    if not use_missing or (use_missing and not zero_as_missing):
                        # add 0 g/h sum to sparse point
                        sparse_point = bin_sparse_points[fid]
                        node_histograms[node_idx][fid][sparse_point][0] += zero_opt_node_sum[node_idx][0] - \
                            zero_optim[node_idx][fid][
                            0]
                        node_histograms[node_idx][fid][sparse_point][1] += zero_opt_node_sum[node_idx][1] - \
                            zero_optim[node_idx][fid][
                            1]
                        node_histograms[node_idx][fid][sparse_point][2] += zero_opt_node_sum[node_idx][2] - \
                            zero_optim[node_idx][fid][
                            2]
                    else:
                        # if 0 is regarded as missing value, add to missing bin
                        node_histograms[node_idx][fid][-1][0] += zero_opt_node_sum[node_idx][0] - \
                            zero_optim[node_idx][fid][0]
                        node_histograms[node_idx][fid][-1][1] += zero_opt_node_sum[node_idx][1] - \
                            zero_optim[node_idx][fid][1]
                        node_histograms[node_idx][fid][-1][2] += zero_opt_node_sum[node_idx][2] - \
                            zero_optim[node_idx][fid][2]

        ret = FeatureHistogram._generate_histogram_key_value_list(node_histograms, node_map, bin_split_points,
                                                                  parent_nid_map, sibling_node_id_map,
                                                                  partition_key=partition_key)
        return ret

    @staticmethod
    def _recombine_histograms(histograms_list: list, node_map, feature_num):

        histograms = [[[] for j in range(feature_num)] for k in range(len(node_map))]
        for tuple_ in histograms_list:
            node_id, fid = tuple_[0]
            node_idx = node_map[node_id]
            histograms[int(node_idx)][int(fid)] = FeatureHistogram._tensor_histogram_cumsum(tuple_[1][1])
        return histograms

    @staticmethod
    def _construct_table(histograms_table):
        histograms_table = histograms_table.mapValues(FeatureHistogram._host_histogram_cumsum_map_func)
        return histograms_table

    """
    Histogram subtraction functions
    """

    def _update_node_info(self, nodes):
        """
        generate node summaries for hist subtraction
        """
        if nodes is None:
            raise ValueError('node list should not be None if histogram subtraction is enabled')
        self._cur_to_split_node_info = {}
        for node in nodes:
            node_id = node.id
            self._cur_to_split_node_info[node_id] = {'pid': node.parent_nodeid, 'is_left_node': node.is_left_node}

    @staticmethod
    def _is_root_node(node_map):
        """
        check if current to split is root node
        """
        return 0 in node_map

    def _update_cached_histograms(self, dep, ret='tensor'):
        """
        update cached parent histograms
        """

        if dep != self._cur_dep and ret == 'tensor':
            del self._prev_layer_cached_histograms  # delete previous cached histograms
            self._prev_layer_cached_histograms = self._cur_layer_cached_histograms  # update cached histograms
            self._cur_layer_cached_histograms = {}  # for caching new histograms
            self._cur_dep = dep

        elif dep != self._cur_dep and ret == 'tb':
            del self._prev_layer_dtable
            self._prev_layer_dtable = self._cur_layer_dtables[0]
            for table in self._cur_layer_dtables[1:]:
                self._prev_layer_dtable = self._prev_layer_dtable.union(table)
            self._cur_layer_dtables = []
            self._cur_dep = dep

        LOGGER.info('hist subtraction dep is updated to {}'.format(self._cur_dep))

    def _cached_histograms(self, histograms, ret='tensor'):
        """
        cached cur layer histograms
        """
        if ret == 'tb':
            self._cur_layer_dtables.append(histograms)
        elif ret == 'tensor':
            result_nid, result = histograms
            for node_id, result_hist in zip(result_nid, result):
                self._cur_layer_cached_histograms[node_id] = result_hist

    @staticmethod
    def _inverse_node_map(node_map):
        return {v: k for k, v in node_map.items()}

    def _is_left(self, node_id):
        """
        check if it is left node
        """
        return self._cur_to_split_node_info[node_id]['is_left_node']

    def _get_parent_nid_map(self, ):
        """
        get a map that can map a node to its parent node
        """

        rs = {}
        for nid in self._cur_to_split_node_info:
            if nid == 0:
                return None
            rs[nid] = self._cur_to_split_node_info[nid]['pid']
        return rs

    @staticmethod
    def _trim_node_map(node_map, leaf_sample_counts):
        """
        Only keep the nodes with fewer sample and remove their siblings, for accelerating hist computation
        """

        inverse_node_map = {v: k for k, v in node_map.items()}
        sibling_node_map = {}
        # if is root node, return directly
        if 0 in node_map:
            return node_map, None

        kept_node_id = []

        idx = 0
        for left_count, right_count in zip(leaf_sample_counts[0::2], leaf_sample_counts[1::2]):
            if left_count < right_count:
                kept_node_id.append(inverse_node_map[idx])
                sibling_node_map[inverse_node_map[idx]] = inverse_node_map[idx + 1]
            else:
                kept_node_id.append(inverse_node_map[idx + 1])
                sibling_node_map[inverse_node_map[idx + 1]] = inverse_node_map[idx]
            idx += 2

        new_node_map = {node_id: idx for idx, node_id in enumerate(kept_node_id)}

        return new_node_map, sibling_node_map

    @staticmethod
    def _g_h_count_sub(hist_a, hist_b):
        return hist_a[0] - hist_b[0], hist_a[1] - hist_b[1], hist_a[2] - hist_b[2]

    @staticmethod
    def _hist_sub(tensor_hist_a, tensor_hist_b):

        new_hist = copy.deepcopy(tensor_hist_b)
        assert len(tensor_hist_a) == len(tensor_hist_b)
        for fid in range(len(tensor_hist_a)):
            for bid in range(len(tensor_hist_a[fid])):  # if is not a valid feature, bin_num is 0
                new_hist[fid][bid][0], new_hist[fid][bid][1], new_hist[fid][bid][2] = FeatureHistogram._g_h_count_sub(
                    tensor_hist_a[fid][bid], tensor_hist_b[fid][bid])

        return new_hist

    @staticmethod
    def _table_hist_sub(kv):

        res = []
        for k, v in kv:
            parent_hist, son_hist = v
            fid, p_hist = parent_hist
            (fid, node_id, sib_node_id), s_hist = son_hist
            assert len(p_hist) == len(s_hist), 'bin num not equal'
            bin_num = len(p_hist)
            new_hist = [[0, 0, 0] for i in range(bin_num)]
            for bid in range(bin_num):
                # get sibling histograms by hist subtraction, if is not a valid feature, bin_num is 0
                new_hist[bid][0], new_hist[bid][1], new_hist[bid][2] = FeatureHistogram._g_h_count_sub(p_hist[bid],
                                                                                                       s_hist[bid])
            # key, value
            res.append(((sib_node_id, fid), (fid, new_hist)))
            res.append(((node_id, fid), (fid, s_hist)))

        return res

    def _tensor_subtraction(self, histograms, node_map):
        """
        histogram subtraction for tensor format
        """

        inverse_node_map = self._inverse_node_map(node_map)  # get inverse node map
        node_ids = []
        p_node_ids = []

        for idx in range(len(histograms)):
            node_id = inverse_node_map[idx]
            node_ids.append(node_id)
            p_node_ids.append(self._cur_to_split_node_info[node_id]['pid'])  # get parent histograms id

        result = []
        result_nid = []

        for node_id, pid, hist in zip(node_ids, p_node_ids, histograms):

            # get sibling histograms by histogram subtraction
            parent_hist = self._prev_layer_cached_histograms[pid]
            sibling_hist = self._hist_sub(parent_hist, hist)
            # is right sibling or left sibling ?
            if self._is_left(node_id):
                result.append(hist)
                result.append(sibling_hist)
                result_nid.append(node_id)
                result_nid.append(node_id + 1)
            else:
                result.append(sibling_hist)
                result.append(hist)
                result_nid.append(node_id - 1)
                result_nid.append(node_id)

        return result_nid, result

    def _table_subtraction(self, histograms):
        """
        histogram subtraction for table format
        """

        LOGGER.debug('joining parent and son histogram tables')
        parent_and_son_hist_table = self._prev_layer_dtable.join(histograms, lambda v1, v2: (v1, v2))
        result = parent_and_son_hist_table.mapPartitions(FeatureHistogram._table_hist_sub, use_previous_behavior=False)
        return result
