from typing import Union
from fate.ml.abc.module import Model
from fate.ml.ensemble.learner.decision_tree.hetero.host import HeteroDecisionTreeHost
from fate.ml.ensemble.algo.secureboost.hetero._base import HeteroBoostingTree
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.ensemble.utils.binning import binning
import logging


logger = logging.getLogger(__name__) 


class HeteroSecureBoostHost(HeteroBoostingTree):

    def __init__(self, num_trees=3, learning_rate=0.3, max_depth=3,  max_split_nodes=1024, max_bin=32, colsample=1.0) -> None:
        super().__init__()
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_split_nodes = max_split_nodes
        self.max_bin = max_bin
        self.colsample = colsample

    def get_tree(self, idx):
        return self._trees[idx]

    def fit(self, ctx: Context, train_data: DataFrame, validate_data: DataFrame = None) -> None:

        # data binning
        bin_info = binning(train_data, max_bin=self.max_bin)
        bin_data: DataFrame = train_data.bucketize(boundaries=bin_info)
        logger.info('data binning done')
        for tree_idx, tree_ctx in ctx.on_iterations.ctxs_range(self.num_trees):
            logger.info('start to fit a host tree')
            tree = HeteroDecisionTreeHost(max_depth=self.max_depth, max_split_nodes=self.max_split_nodes)
            tree.booster_fit(tree_ctx, bin_data, bin_info)
            self._trees.append(tree)
            self._saved_tree.append(tree.get_model())
            self._update_feature_importance(tree.get_feature_importance())
            logger.info('fitting host decision tree {} done'.format(tree_idx))

    def predict(self, ctx: Context, predict_data: DataFrame) -> DataFrame:
        pass

    def get_model(self) -> dict:
        pass

    def from_model(cls, model: dict):
        pass