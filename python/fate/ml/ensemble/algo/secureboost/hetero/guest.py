from typing import Optional
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.ensemble.algo.secureboost.hetero._base import HeteroBoostingTree
from fate.ml.ensemble.learner.decision_tree.hetero.guest import HeteroDecisionTreeGuest
from fate.ml.ensemble.utils.binning import binning
from fate.ml.ensemble.learner.decision_tree.tree_core.loss import BCELoss, CELoss, L2Loss
import logging


logger = logging.getLogger(__name__)


OBJECTIVE = {
    "binary": BCELoss,
    "multiclass": CELoss,
    "regression": L2Loss
}


class HeteroSecureBoostGuest(HeteroBoostingTree):

    def __init__(self, num_trees=3, learning_rate=0.3, max_depth=3, objective='binary', num_class=3, max_split_nodes=1024, 
                 max_bin=32, l2=0.1, l1=0, colsample=1.0) -> None:
        super().__init__()
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.objective = objective
        self.max_split_nodes = max_split_nodes
        self.max_bin = max_bin
        self.l2 = l2
        self.l1 = l1
        self.colsample = colsample
        self.num_class = num_class
        self._accumulate_scores = None
        self._tree_dim = 1  # tree dimension, if is multilcass task, tree dim > 1
        self._loss_func = None

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
    
    def get_cache_predict_score(self):
        return self._loss_func.predict(self._accumulate_scores)
    
    def get_tree(self, idx):
        return self._trees[idx]

    def fit(self, ctx: Context, train_data: DataFrame, validate_data: DataFrame = None) -> None:
        
        # data binning
        bin_info = binning(train_data, max_bin=self.max_bin)
        bin_data: DataFrame = train_data.bucketize(boundaries=bin_info)

        # init loss func & scores
        self._loss_func = self._get_loss_func(self.objective)
        label = bin_data.label
        self._accumulate_scores = self._loss_func.initialize(label)

        # start tree fitting
        for tree_idx, tree_ctx in ctx.on_iterations.ctxs_range(len(self._trees), len(self._trees)+self.num_trees):
            # compute gh of current iter
            logger.info('start to fit a host tree')
            gh = self._compute_gh(bin_data, self._accumulate_scores, self._loss_func)
            tree = HeteroDecisionTreeGuest(max_depth=self.max_depth, max_split_nodes=self.max_split_nodes, l2=self.l2)
            tree.booster_fit(tree_ctx, bin_data, gh, bin_info)
            # accumulate scores of cur boosting round
            scores = tree.get_sample_predict_weights()
            assert len(scores) == len(self._accumulate_scores), f"tree predict scores length {len(scores)} not equal to accumulate scores length {len(self._accumulate_scores)}."
            scores =  scores.loc(self._accumulate_scores.get_indexer(target="sample_id"), preserve_order=True)
            self._accumulate_scores = self._accumulate_scores + scores * self.learning_rate
            self._trees.append(tree)
            self._saved_tree.append(tree.get_model())
            self._update_feature_importance(tree.get_feature_importance())
            logger.info('fitting guest decision tree {} done'.format(tree_idx))

    def predict(self, ctx: Context, predict_data: DataFrame) -> DataFrame:
        pass

    def _get_hyper_param(self) -> dict:
        return {
            "num_trees": self.num_trees,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "objective": self.objective,
            "max_split_nodes": self.max_split_nodes,
            "max_bin": self.max_bin,
            "l2": self.l2,
            "num_class": self.num_class,
            "colsample": self.colsample
        }
    
    def from_model(self, model: dict):
        
        trees = model['trees']
        self._saved_tree = trees
        self._trees = [HeteroDecisionTreeGuest.from_model(tree) for tree in trees]
        return self