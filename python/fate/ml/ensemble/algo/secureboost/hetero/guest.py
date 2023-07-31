from typing import Optional
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.abc.module import HeteroModule
from fate.ml.ensemble.learner.decision_tree.hetero.guest import HeteroDecisionTreeGuest
from fate.ml.ensemble.learner.decision_tree.hetero.host import HeteroDecisionTreeHost
from fate.ml.ensemble.utils.binning import binning
from fate.ml.ensemble.learner.decision_tree.tree_core.loss import BCELoss, CELoss, L2Loss


OBJECTIVE = {
    "binary": BCELoss,
    "multiclass": CELoss,
    "regression": L2Loss
}


class HeteroSecureBoostGuest(HeteroModule):

    def __init__(self, num_trees=3, learning_rate=0.3, max_depth=3, feature_importance_type='split', objective='binary', num_class=3, max_split_nodes=1024, 
                 max_bin=32, l2=0.1, l1=0, colsample=1.0) -> None:
        super().__init__()
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_importance_type = feature_importance_type
        self.objective = objective
        self.max_split_nodes = max_split_nodes
        self.max_bin = max_bin
        self.l2 = l2
        self.num_class = num_class

        self._accumulate_scores = None

    def _get_loss_func(self, objective: str) -> Optional[object]:
        # to lowercase
        objective = objective.lower()
        assert objective in OBJECTIVE, f"objective {objective} not found, supported objective: {list(OBJECTIVE.keys())}"
        loss_func = OBJECTIVE[objective]()
        return loss_func
    
    def _compute_gh(self, data: DataFrame, scores: DataFrame, loss_func):

        label = data.label
        predict = loss_func.predict(scores)
        gh = data.create_frame()
        loss_func.compute_grad(gh, label, predict)
        loss_func.compute_hess(gh, label, predict)

        return gh

    def fit(self, ctx: Context, train_data: DataFrame, validate_data: DataFrame = None) -> None:
        
        # data binning
        bin_info = binning(train_data, max_bin=self.max_bin)
        bin_data: DataFrame = train_data.bucketize(boundaries=bin_info)

        # init loss func & scores
        loss_func = self._get_loss_func(self.objective)
        label = bin_data.label
        self._accumulate_scores = loss_func.initialize(label)

        

    
    def predict(self, ctx: Context, predict_data: DataFrame) -> DataFrame:
        pass