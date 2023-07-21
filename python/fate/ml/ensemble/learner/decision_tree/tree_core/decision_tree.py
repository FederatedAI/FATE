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
import functools
from fate.arch import Context
from fate.arch.dataframe import DataFrame


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
        fval : float or int, optional
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
                 fval=None, weight=0, is_leaf=False, grad=None,
                 hess=None, l=-1, r=-1,
                 missing_dir=1, sample_num=0, is_left_node=False, sibling_nodeid=None, parent_nodeid=None, inst_indices=None):
        
        self.nid = nid
        self.sitename = sitename
        self.fid = fid
        self.fval = fval
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
        return "(node_id {}: left {}, right {}, is_leaf {}, sample_count {},  g {}, h {})".format(self.nid, \
                 self.l, self.r, self.is_leaf, self.sample_num, self.grad, self.hess)


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
            When running Hetero-SBT, this parameter is now abandoned， guest side will compute split, gain of local features,
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

    def _tree_to_array(self):
        # convert tree node to array
        pass

    def _init_sample_pos(self, train_data: DataFrame):
        sample_pos = train_data.create_frame()
        sample_pos['node_idx'] = 0  # position of current sample
        sample_pos['dir'] = True  # direction to next layer, use True to initalize all
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

        sum_g = gh['g'].sum()
        sum_h = gh['h'].sum()
        root_node = Node(nid=0, grad=sum_g, hess=sum_h, sitename=sitename, sample_num=len(gh))

        return root_node

    def fit(self, ctx: Context, train_data: DataFrame, grad_and_hess: DataFrame, encryptor):
        raise NotImplementedError("This method should be implemented by subclass")

    def get_feature_importance(self):
        return self._feature_importance
    
    def get_sample_predict_weights(self):
        pass

    def get_nodes(self):
        return self._nodes

    def from_model(self):
        pass

    def get_model(self):
        pass