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
from typing import List
import functools
from fate.arch.dataframe import DataFrame
from fate.ml.abc.module import HeteroModule, Model
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import FeatureImportance, Node
from typing import Dict
import numpy as np


class HeteroBoostingTree(HeteroModule):
    def __init__(self) -> None:
        super().__init__()
        self._global_feature_importance = {}
        self._trees = []
        self._saved_tree = []
        self._fid_name_mapping = {}

    def _update_feature_importance(self, fi_dict: Dict[int, FeatureImportance]):
        for fid, fi in fi_dict.items():
            if fid not in self._global_feature_importance:
                self._global_feature_importance[fid] = fi
            else:
                self._global_feature_importance[fid] = self._global_feature_importance[fid] + fi

    def _sum_leaf_weights(self, leaf_pos: DataFrame, trees, learing_rate: float, num_dim=1):
        def _compute_score(leaf_pos_: np.array, trees_: List[List[Node]], learning_rate: float, num_dim_=1):
            score = np.zeros(num_dim_)
            leaf_pos_ = leaf_pos_["sample_pos"]
            tree_idx = 0
            for node_idx, tree in zip(leaf_pos_, trees_):
                recovered_idx = -(node_idx + 1)
                score[tree_idx % num_dim_] += tree[recovered_idx].weight * learning_rate
                tree_idx += 1

            return float(score[0]) if num_dim_ == 1 else [score]

        tree_list = [tree.get_nodes() for tree in trees]
        apply_func = functools.partial(_compute_score, trees_=tree_list, learning_rate=learing_rate, num_dim_=num_dim)
        predict_score = leaf_pos.create_frame()
        predict_score["score"] = leaf_pos.apply_row(apply_func)
        return predict_score

    def _get_fid_name_mapping(self, data_instances: DataFrame):
        columns = data_instances.schema.columns
        for idx, col in enumerate(columns):
            self._fid_name_mapping[idx] = col

    def get_trees(self):
        return self._trees

    def get_feature_importance(self):
        return self._global_feature_importance

    def print_forest(self):
        idx = 0
        for tree in self._trees:
            print("tree {}: ".format(idx))
            idx += 1
            tree.print_tree()
            print()

    def _get_hyper_param(self) -> dict:
        pass

    def _load_feature_importance(self, feature_importance: dict):
        self._global_feature_importance = {k: FeatureImportance.from_dict(v) for k, v in feature_importance.items()}

    def get_model(self) -> dict:
        import copy

        hyper_param = self._get_hyper_param()
        result = {}
        result["hyper_param"] = hyper_param
        result["trees"] = copy.deepcopy(self._saved_tree)
        result["fid_name_mapping"] = self._fid_name_mapping
        result["feature_importance"] = {k: v.to_dict() for k, v in self._global_feature_importance.items()}
        return result
