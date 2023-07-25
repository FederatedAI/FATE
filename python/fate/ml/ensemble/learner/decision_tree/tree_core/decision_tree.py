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
import numpy as np
import pandas as pd
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import SplitInfo
from typing import List
import logging


FLOAT_ZERO = 1e-8
LEAF_IDX = -1


logger = logging.getLogger(__name__)

class FeatureImportance(object):

    def __init__(self, gain=0, split=0):

        self.gain = gain
        self.split = split

    def add_gain(self, val):
        self.gain += val

    def add_split(self, val):
        self.split += val

    def __repr__(self):
        return 'gain: {}, split {}'.format(self.importance, self.importance_2)

    def __add__(self, other):
        new_importance = FeatureImportance(gain=self.gain + other.gain, split=self.split + other.split)
        return new_importance


class Node(object):

    """
    Parameters:
        -----------
        nid : int, optional
            ID of the node.
        sitename : str, optional
            Name of the site that the node belongs to.
        fid : int, optional
            ID of the feature that the node splits on.
        bid : float or int, optional
            Feature value that the node splits on.
        weight : float, optional
            Weight of the node.
        is_leaf : bool, optional
            Boolean indicating whether the node is a leaf node.
        grad : float, optional
            Gradient value of the node.
        hess : float, optional
            Hessian value of the node.
        l : int, optional
            ID of the left child node.
        r : int, optional
            ID of the right child node.
        missing_dir : int, optional
            Direction for missing values (1 for left, -1 for right).
        sample_num : int, optional
            Number of samples in the node.
        is_left_node : bool, optional
            Boolean indicating whether the node is a left child node.
        sibling_nodeid : int, optional
            ID of the sibling node.
    """

    def __init__(self, nid=None, sitename=None, fid=None,
                 bid=None, weight=0, is_leaf=False, grad=None,
                 hess=None, l=-1, r=-1,
                 missing_dir=1, sample_num=0, is_left_node=False, sibling_nodeid=None, parent_nodeid=None, inst_indices=None):
        
        self.nid = nid
        self.sitename = sitename
        self.fid = fid
        self.bid = bid
        self.weight = weight
        self.is_leaf = is_leaf
        self.grad = grad
        self.hess = hess
        self.l = l
        self.r = r
        self.missing_dir = missing_dir
        self.sample_num = sample_num
        self.is_left_node = is_left_node
        self.sibling_nodeid = sibling_nodeid
        self.parent_nodeid = parent_nodeid
        self.inst_indices = inst_indices

    def set_inst_indices(self, inst_indices):
        self.inst_indices = inst_indices
        self.inst_indices = self.inst_indices.astype(np.uint32)

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return "(node_id {}: left {}, right {}, is_leaf {}, sample_count {},  g {}, h {}, weight {}, sitename {})".format(self.nid, \
                 self.l, self.r, self.is_leaf, self.sample_num, self.grad, self.hess, self.weight, self.sitename)



def _make_decision(feat_val, bid, missing_dir=None, use_missing=None, zero_as_missing=None, zero_val=0):

    # no missing val
    left, right = True, False
    direction = left if feat_val <= bid + FLOAT_ZERO else right
    return direction


def _update_sample_pos(s: pd.Series, cur_layer_node: List[Node], node_map: dict, sitename=None):

    node_id = s[-1]
    node = cur_layer_node[node_map[node_id]]
    if node.is_leaf:
        return -1  # reach leaf
    feat_val = s[node.fid]
    bid = node.bid
    
    dir_ = _make_decision(feat_val, bid)
    if dir_:  # go left
        ret_node = node.l
    else:
        ret_node = node.r
    return ret_node


def _get_sample_on_local_nodes(s: pd.Series, cur_layer_node: List[Node], node_map: dict, sitename):

    node_id = s[-1]
    node = cur_layer_node[node_map[node_id]]
    on_local_node = (node.sitename == sitename)
    return on_local_node


class DecisionTree(object):
    

    def __init__(self, max_depth=3, use_missing=False, zero_as_missing=False, feature_importance_type='split', valid_features=None):
        """
        Initialize a DecisionTree instance.

        Parameters:
        -----------
        max_depth : int
            The maximum depth of the tree.
        use_missing : bool, optional
            Whether or not to use missing values (default is False).
        zero_as_missing : bool, optional
            Whether to treat zero as a missing value (default is False).
        feature_importance_type : str, optional
            if is 'split', feature_importances calculate by feature split times,
            if is 'gain', feature_importances calculate by feature split gain.
            default: 'split'
            Due to the safety concern, we adjust training strategy of Hetero-SBT in FATE-1.8,
            When running Hetero-SBT, this parameter is now abandoned, guest side will compute split, gain of local features,
            and receive anonymous feature importance results from hosts. Hosts will compute split importance of local features.
        valid_features: list of boolean, optional
            Valid features for training, default is None, which means all features are valid.
        """
        self.max_depth = max_depth
        self.use_missing = use_missing
        self.zero_as_missing = zero_as_missing
        self.feature_importance_type = feature_importance_type

        # runtime variables
        self._nodes = []
        self._cur_layer_node = []
        self._cur_leaf_idx = -1
        self._feature_importance = {}
        self._predict_weights = None
        self._g_tensor, self._h_tensor = None, None
        self._sample_pos = None
        self._leaf_node_map = {}
        self._valid_feature = valid_features

    def _init_sample_pos(self, train_data: DataFrame):
        sample_pos = train_data.create_frame()
        sample_pos['node_idx'] = 0  # position of current sample
        return sample_pos

    def _get_leaf_node_map(self):
        if len(self._nodes) >= len(self._leaf_node_map):
            for n in self._nodes:
                self._leaf_node_map[n.nid] = n.is_leaf

    def _assign_sample_position(self, ):
        pass

    def _convert_bin_idx_to_split_val(self):
        pass

    def _compute_best_splits(self):
        pass

    def _initialize_root_node(self, gh: DataFrame, sitename):

        sum_g = float(gh['g'].sum())
        sum_h = float(gh['h'].sum())
        root_node = Node(nid=0, grad=sum_g, hess=sum_h, sitename=sitename, sample_num=len(gh))

        return root_node
    
    def update_tree(self, cur_to_split: List[Node], split_info: List[SplitInfo]):
        
        assert len(cur_to_split) == len(split_info), 'node num not match split info num, got {} node vs {} split info'.format(len(cur_to_split), len(split_info))

        next_layer_node = []

        for idx in range(len(split_info)):

            node: Node = cur_to_split[idx]

            if split_info[idx] is None:
                node.is_leaf = True
                self._nodes.append(node)
                continue

            sum_grad = node.grad
            sum_hess = node.hess
            node.fid = split_info[idx].best_fid
            node.bid = split_info[idx].best_bid
            node.missing_dir = split_info[idx].missing_dir

            p_id = node.nid
            l_id, r_id = self._tree_node_num + 1, self._tree_node_num + 2
            self._tree_node_num += 2
            node.l, node.r = l_id, r_id

            l_g, l_h = split_info[idx].sum_grad, split_info[idx].sum_hess

            # create new left node and new right node
            left_node = Node(nid=l_id,
                             sitename=self.sitename,
                             grad=float(l_g),
                             hess=float(l_h),
                             weight=float(self.splitter.node_weight(l_g, l_h)),
                             parent_nodeid=p_id,
                             sibling_nodeid=r_id,
                             is_left_node=True
                             )
            right_node = Node(nid=r_id,
                              sitename=self.sitename,
                              grad=float(sum_grad - l_g),
                              hess=float(sum_hess - l_h),
                              weight=float(self.splitter.node_weight(sum_grad - l_g, sum_hess - l_h)),
                              parent_nodeid=p_id,
                              sibling_nodeid=l_id,
                              is_left_node=False)
            next_layer_node.append(left_node)
            next_layer_node.append(right_node)
            self._nodes.append(node)
 
        return next_layer_node
    
    def drop_leaf_samples(self, new_sample_pos: DataFrame, data: DataFrame):
        assert len(new_sample_pos) == len(data), 'sample pos num not match data num, got {} sample pos vs {} data'.format(len(new_sample_pos), len(data))

        x = (new_sample_pos != LEAF_IDX)
        indexer = x.get_indexer('sample_id')
        update_pos = new_sample_pos.loc(indexer, preserve_order=True)[x.as_tensor()]
        new_data = data.loc(indexer, preserve_order=True)[x.as_tensor()]
        logger.info('drop leaf samples, new sample count is {}, {} samples dropped'.format(len(new_sample_pos), len(data) - len(new_data)))
        return new_data, update_pos

    def get_column_max_bin(self, result_dict):
        bin_len = {}
        for column, values in result_dict.items():
            bin_num = len(values)
            bin_len[column] = bin_num 
        max_max_value = max(bin_len.values())
        return bin_len, max_max_value

    def fit(self, ctx: Context, train_data: DataFrame, grad_and_hess: DataFrame, encryptor):
        raise NotImplementedError("This method should be implemented by subclass")

    def get_feature_importance(self):
        return self._feature_importance
    
    def get_sample_predict_weights(self):
        pass

    def get_nodes(self):
        return self._nodes

    def print_tree(self):
        from anytree import Node as AnyNode, RenderTree
        nodes = self._nodes
        anytree_nodes = {}
        for node in nodes:
            if not node.is_leaf:
                anytree_nodes[node.nid] = AnyNode(name=f'{node.nid}: fid {node.fid}, bid {node.bid}')
            else:
                anytree_nodes[node.nid] = AnyNode(name=f'{node.nid}: weight {node.weight}, leaf')
        for node in nodes:
            if node.l != -1:
                anytree_nodes[node.l].parent = anytree_nodes[node.nid]
            if node.r != -1:
                anytree_nodes[node.r].parent = anytree_nodes[node.nid]
        
        for pre, _, node in RenderTree(anytree_nodes[0]):
            print("%s%s" % (pre, node.name))

    def from_model(self):
        pass

    def get_model(self):
        pass