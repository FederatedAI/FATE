from typing import Optional
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.abc.module import HeteroModule
from fate.ml.ensemble.learner.decision_tree.hetero.guest import HeteroDecisionTreeGuest
from fate.ml.ensemble.learner.decision_tree.hetero.host import HeteroDecisionTreeHost



class HeteroSecureBoostGuest(HeteroModule):

    def __init__(self, num_trees=3, learning_rate=0.3, max_depth=3, feature_importance_type='split', objective='binary', max_split_nodes=1024) -> None:
        super().__init__()

    def fit(self, ctx: Context, train_data: DataFrame, validate_data: DataFrame = None) -> None:
        return super().fit(ctx, train_data, validate_data)
    
    def predict(self, ctx: Context, predict_data: DataFrame) -> DataFrame:
        return super().predict(ctx, predict_data)