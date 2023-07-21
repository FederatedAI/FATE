from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import DecisionTree
from fate.arch import Context
from fate.arch.dataframe import DataFrame


class HeteroDecisionTreeGuest(DecisionTree):

    def __init__(self, max_depth=3, feature_importance_type='split', valid_features=None):
        super().__init__(max_depth, use_missing=False, zero_as_missing=False, feature_importance_type=feature_importance_type, valid_features=valid_features)

    def fit(self, ctx: Context, train_data: DataFrame, grad_and_hess: DataFrame, encryptor):
        pass

    def predict(self, ctx: Context, data_inst: DataFrame):
        pass
    
