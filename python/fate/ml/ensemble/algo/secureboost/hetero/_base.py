from typing import Optional
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.abc.module import HeteroModule
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import FeatureImportance
from typing import Dict


class HeteroBoostingTree(HeteroModule):

    def __init__(self) -> None:
        super().__init__()
        self._global_feature_importance = {}
        self._trees = []
        self._saved_tree = []
    
    def get_tree(self, idx):
        return self._trees[idx]
    
    def _update_feature_importance(self, fi_dict: Dict[int, FeatureImportance]):
        
        for fid, fi in fi_dict.items():
            if fid not in self._global_feature_importance:
                self._global_feature_importance[fid] = fi
            else:
                self._global_feature_importance[fid] = self._global_feature_importance[fid] + fi

    def get_trees(self):
        return self._trees

    def get_feature_importance(self):
        return self._global_feature_importance
    