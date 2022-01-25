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

# =============================================================================
# DecisionTree Base Class
# =============================================================================
import abc
from abc import ABC
import numpy as np
import functools
from federatedml.ensemble.basic_algorithms.algorithm_prototype import BasicAlgorithms
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.feature.fate_element_type import NoneType
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitter import \
    SplitInfo, Splitter
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.node import Node
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.feature_histogram import \
    HistogramBag, FeatureHistogram
from typing import List
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.feature_importance import FeatureImportance


class DecisionTree(BasicAlgorithms, ABC):

    def __init__(self, tree_param):

        # input parameters
        self.criterion_method = tree_param.criterion_method
        self.criterion_params = tree_param.criterion_params
        self.max_depth = tree_param.max_depth
        self.min_sample_split = tree_param.min_sample_split
        self.min_impurity_split = tree_param.min_impurity_split
        self.min_leaf_node = tree_param.min_leaf_node
        self.max_split_nodes = tree_param.max_split_nodes
        self.feature_importance_type = tree_param.feature_importance_type
        self.n_iter_no_change = tree_param.n_iter_no_change
        self.tol = tree_param.tol
        self.use_missing = tree_param.use_missing
        self.zero_as_missing = tree_param.zero_as_missing
        self.min_child_weight = tree_param.min_child_weight
        self.sitename = ''

        # transfer var
        self.transfer_inst = None

        # runtime variable
        self.feature_importance = {}
        self.tree_node = []
        self.cur_layer_nodes = []
        self.cur_to_split_nodes = []
        self.tree_node_num = 0
        self.runtime_idx = None
        self.valid_features = None
        self.splitter = Splitter(
            self.criterion_method,
            self.criterion_params,
            self.min_impurity_split,
            self.min_sample_split,
            self.min_leaf_node,
            self.min_child_weight)  # splitter for finding splits
        self.inst2node_idx = None  # record the internal node id an instance belongs to
        self.sample_leaf_pos = None  # record the final leaf id of samples
        self.sample_weights = None  # leaf weights of samples
        self.leaf_count = None  # num of samples a leaf covers

        # data
        self.data_bin = None  # data after binning
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.data_with_node_assignments = None
        self.cur_layer_sample_count = None

        # g_h
        self.grad_and_hess = None

        # for data protection
        self.split_maskdict = {}
        self.missing_dir_maskdict = {}

        # histogram
        self.deterministic = tree_param.deterministic
        self.hist_computer = FeatureHistogram()
        if self.deterministic:
            self.hist_computer.stable_reduce = True

    """
    Common functions
    """

    def get_feature_importance(self):
        return self.feature_importance

    @staticmethod
    def get_grad_hess_sum(grad_and_hess_table):
        LOGGER.info("calculate the sum of grad and hess")
        grad, hess = grad_and_hess_table.reduce(
            lambda value1, value2: (value1[0] + value2[0], value1[1] + value2[1]))
        return grad, hess

    def init_data_and_variable(self, flowid, runtime_idx, data_bin, bin_split_points, bin_sparse_points, valid_features,
                               grad_and_hess):

        self.set_flowid(flowid)
        self.set_runtime_idx(runtime_idx)

        LOGGER.info("set valid features")
        self.valid_features = valid_features
        self.grad_and_hess = grad_and_hess

        self.data_bin = data_bin
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_points

    def check_max_split_nodes(self):
        # check max_split_nodes
        if self.max_split_nodes != 0 and self.max_split_nodes % 2 == 1:
            self.max_split_nodes += 1
            LOGGER.warning('an even max_split_nodes value is suggested '
                           'when using histogram-subtraction, max_split_nodes reset to {}'.format(self.max_split_nodes))

    def set_flowid(self, flowid=0):
        LOGGER.info("set flowid, flowid is {}".format(flowid))
        self.transfer_inst.set_flowid(flowid)

    def set_runtime_idx(self, runtime_idx):
        self.runtime_idx = runtime_idx
        self.sitename = ":".join([self.sitename, str(self.runtime_idx)])

    """
    Histogram interface
    """

    def get_local_histograms(self, dep, data_with_pos, g_h, node_sample_count, cur_to_split_nodes, node_map,
                             ret='tensor', hist_sub=True):

        LOGGER.info("start to compute node histograms")
        acc_histograms = self.hist_computer.compute_histogram(dep,
                                                              data_with_pos,
                                                              g_h,
                                                              self.bin_split_points,
                                                              self.bin_sparse_points,
                                                              self.valid_features,
                                                              node_map, node_sample_count,
                                                              use_missing=self.use_missing,
                                                              zero_as_missing=self.zero_as_missing,
                                                              ret=ret,
                                                              hist_sub=hist_sub,
                                                              cur_to_split_nodes=cur_to_split_nodes)
        LOGGER.info("compute node histograms done")

        return acc_histograms

    """
    Node map functions
    """

    @staticmethod
    def get_node_map(nodes: List[Node], left_node_only=False):
        node_map = {}
        idx = 0
        for node in nodes:
            if node.id != 0 and (not node.is_left_node and left_node_only):
                continue
            node_map[node.id] = idx
            idx += 1
        return node_map

    @staticmethod
    def get_leaf_node_map(nodes: List[Node]):
        leaf_nodes = []
        for n in nodes:
            if n.is_leaf:
                leaf_nodes.append(n)
        return DecisionTree.get_node_map(leaf_nodes)

    """
    Sample count functions
    """

    @staticmethod
    def sample_count_map_func(kv, node_map):

        # record node sample number in count_arr
        count_arr = np.zeros(len(node_map))
        for k, v in kv:
            if isinstance(v, int):  # leaf node format: (leaf_node_id)
                key = v
            else:  # internal node format: (1, node_id)
                key = v[1]
            if key not in node_map:
                continue
            node_idx = node_map[key]
            count_arr[node_idx] += 1
        return count_arr

    @staticmethod
    def sample_count_reduce_func(v1, v2):
        return v1 + v2

    def count_node_sample_num(self, inst2node_idx, node_map):
        """
        count sample number in internal nodes during training
        """
        count_func = functools.partial(self.sample_count_map_func, node_map=node_map)
        rs = inst2node_idx.applyPartitions(count_func).reduce(self.sample_count_reduce_func)
        return rs

    """
    Sample weight functions
    """

    def get_sample_weights(self):
        # return sample weights to boosting class
        return self.sample_weights

    @staticmethod
    def assign_instance_to_root_node(data_bin, root_node_id):
        return data_bin.mapValues(lambda inst: (1, root_node_id))

    @staticmethod
    def float_round(num):
        """
        prevent float error
        """
        return np.round(num, consts.TREE_DECIMAL_ROUND)

    def update_feature_importance(self, splitinfo, record_site_name=True):

        inc_split, inc_gain = 1, splitinfo.gain

        sitename = splitinfo.sitename
        fid = splitinfo.best_fid

        if record_site_name:
            key = (sitename, fid)
        else:
            key = fid

        if key not in self.feature_importance:
            self.feature_importance[key] = FeatureImportance(0, 0, self.feature_importance_type)

        self.feature_importance[key].add_split(inc_split)
        if inc_gain is not None:
            self.feature_importance[key].add_gain(inc_gain)

    @staticmethod
    def get_node_weights(node_id, tree_nodes):
        return tree_nodes[node_id].weight

    def extract_sample_weights_from_node(self, sample_leaf_pos):
        """
        Given a dtable contains leaf positions of samples, return leaf weights
        """
        func = functools.partial(self.get_node_weights, tree_nodes=self.tree_node)
        sample_weights = sample_leaf_pos.mapValues(func)
        return sample_weights

    def sample_weights_post_process(self):

        self.sample_weights = self.extract_sample_weights_from_node(self.sample_leaf_pos)
        leaf_node_map = self.get_leaf_node_map(self.tree_node)
        leaf_count = self.count_node_sample_num(self.sample_leaf_pos, leaf_node_map)
        rs = {}
        for k, v in leaf_node_map.items():
            rs[k] = int(leaf_count[v])
        self.leaf_count = rs
        LOGGER.debug('final leaf count is {}'.format(self.leaf_count))

    @staticmethod
    def make_decision(data_inst, fid, bid, missing_dir, use_missing, zero_as_missing, zero_val=0):

        left, right = True, False
        missing_dir = left if missing_dir == -1 else right

        # use missing and zero as missing
        if use_missing and zero_as_missing:
            # missing or zero
            if data_inst.features.get_data(fid) == NoneType() or data_inst.features.get_data(fid, None) is None:
                return missing_dir

        # is missing feat
        if data_inst.features.get_data(fid) == NoneType():
            return missing_dir

        # no missing val
        feat_val = data_inst.features.get_data(fid, zero_val)
        direction = left if feat_val <= bid + consts.FLOAT_ZERO else right
        return direction

    @staticmethod
    def go_next_layer(node, data_inst, use_missing, zero_as_missing, bin_sparse_point=None,
                      split_maskdict=None,
                      missing_dir_maskdict=None,
                      decoder=None,
                      return_node_id=True):

        if missing_dir_maskdict is not None and split_maskdict is not None:
            fid = decoder("feature_idx", node.fid, split_maskdict=split_maskdict)
            bid = decoder("feature_val", node.bid, node.id, split_maskdict=split_maskdict)
            missing_dir = decoder("missing_dir", node.missing_dir, node.id, missing_dir_maskdict=missing_dir_maskdict)
        else:
            fid, bid = node.fid, node.bid
            missing_dir = node.missing_dir
        zero_val = 0 if bin_sparse_point is None else bin_sparse_point[fid]
        go_left = DecisionTree.make_decision(data_inst, fid, bid, missing_dir, use_missing, zero_as_missing, zero_val)

        if not return_node_id:
            return go_left

        if go_left:
            return node.left_nodeid
        else:
            return node.right_nodeid

    def round_leaf_val(self):
        # process predict weight to prevent float error
        for node in self.tree_node:
            if node.is_leaf:
                node.weight = self.float_round(node.weight)

    @staticmethod
    def mo_weight_extract(node):

        mo_weight = None
        weight = node.weight
        if isinstance(node.weight, np.ndarray) and len(node.weight) > 1:
            weight = -1
            mo_weight = list(node.weight)  # use multi output

        return weight, mo_weight

    @staticmethod
    def mo_weight_load(node_param):

        weight = node_param.weight
        mo_weight = list(node_param.mo_weight)
        if len(mo_weight) != 0:
            weight = np.array(list(node_param.mo_weight))

        return weight

    """
    To implement
    """

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def predict(self, data_inst):
        pass

    @abc.abstractmethod
    def initialize_root_node(self, *args):
        pass

    @abc.abstractmethod
    def compute_best_splits(self, *args):
        pass

    @abc.abstractmethod
    def update_instances_node_positions(self, *args):
        pass

    @abc.abstractmethod
    def assign_an_instance(self, *args):
        pass

    @abc.abstractmethod
    def assign_instances_to_new_node(self, *args):
        pass

    @abc.abstractmethod
    def update_tree(self, *args):
        pass

    @abc.abstractmethod
    def convert_bin_to_real(self, *args):
        pass

    @abc.abstractmethod
    def get_model_meta(self):
        raise NotImplementedError("method should overload")

    @abc.abstractmethod
    def get_model_param(self):
        raise NotImplementedError("method should overload")

    @abc.abstractmethod
    def set_model_param(self, model_param):
        pass

    @abc.abstractmethod
    def set_model_meta(self, model_meta):
        pass

    @abc.abstractmethod
    def traverse_tree(self, *args):
        pass

    """
    Model I/O
    """

    def get_model(self):

        model_meta = self.get_model_meta()
        model_param = self.get_model_param()
        return model_meta, model_param

    def load_model(self, model_meta=None, model_param=None):
        LOGGER.info("load tree model")
        self.set_model_meta(model_meta)
        self.set_model_param(model_param)

    """
    For debug
    """

    def print_leafs(self):
        LOGGER.debug('printing tree')
        if len(self.tree_node) == 0:
            LOGGER.debug('this tree is empty')
        else:
            for node in self.tree_node:
                LOGGER.debug(node)

    @staticmethod
    def print_split(split_infos: [SplitInfo]):
        LOGGER.debug('printing split info')
        for info in split_infos:
            LOGGER.debug(info)

    @staticmethod
    def print_hist(hist_list: [HistogramBag]):
        LOGGER.debug('printing histogramBag')
        for bag in hist_list:
            LOGGER.debug(bag)
