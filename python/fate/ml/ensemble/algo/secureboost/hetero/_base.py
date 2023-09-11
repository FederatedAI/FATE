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

    def _update_feature_importance(self, fi_dict: Dict[int, FeatureImportance]):
        for fid, fi in fi_dict.items():
            if fid not in self._global_feature_importance:
                self._global_feature_importance[fid] = fi
            else:
                self._global_feature_importance[fid] = self._global_feature_importance[fid] + fi

    def _sum_leaf_weights(self, leaf_pos: DataFrame, trees, learing_rate: float, loss_func):
        def _compute_score(leaf_pos: np.array, trees: List[List[Node]], learning_rate: float):
            score = 0
            leaf_pos = leaf_pos["sample_pos"]
            for node_idx, tree in zip(leaf_pos, trees):
                recovered_idx = -(node_idx + 1)
                score += tree[recovered_idx].weight * learning_rate
            return score

        tree_list = [tree.get_nodes() for tree in trees]
        apply_func = functools.partial(_compute_score, trees=tree_list, learning_rate=learing_rate)
        predict_score = leaf_pos.create_frame()
        predict_score["score"] = leaf_pos.apply_row(apply_func)
        return loss_func.predict(predict_score)

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

    def get_model(self) -> dict:
        import copy

        hyper_param = self._get_hyper_param()
        result = {}
        result["hyper_param"] = hyper_param
        result["trees"] = copy.deepcopy(self._saved_tree)
        return result
