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


class DecisionTree(object):
    def __init__(self, tree_param):
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
        self.sitename = ''

    def fit(self):
        raise NotImplementedError("fit method should overload")

    def predict(self, data_inst):
        raise NotImplementedError("fit method should overload")
    
    def set_runtime_idx(self, runtime_idx):
        self.runtime_idx = runtime_idx
        self.sitename = ":".join([self.sitename, str(self.runtime_idx)])

