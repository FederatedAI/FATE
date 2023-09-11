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
import random
import os
from fate.ml.ensemble.learner.decision_tree.hetero.host import HeteroDecisionTreeHost
from fate.ml.ensemble.algo.secureboost.hetero._base import HeteroBoostingTree
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.ensemble.utils.binning import binning
from fate.ml.ensemble.algo.secureboost.common.predict import predict_leaf_host
import logging

logger = logging.getLogger(__name__)


class HeteroSecureBoostHost(HeteroBoostingTree):
    def __init__(self, num_trees=3, learning_rate=0.3, max_depth=3, max_bin=32, hist_sub=True) -> None:
        super().__init__()
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bin = max_bin
        self._model_loaded = False
        self._hist_sub = hist_sub

    def get_tree(self, idx):
        return self._trees[idx]

    def _get_seeds(self, ctx: Context):
        if ctx.cipher.allow_custom_random_seed:
            seed = ctx.cipher.get_custom_random_seed()
            random_state = random.Random(seed)
            yield random_state.getrandbits(64)
            while True:
                yield random_state.getrandbits(64)
        else:
            while True:
                random_seed = os.urandom(8)
                yield int.from_bytes(random_seed, byteorder="big")

    def fit(self, ctx: Context, train_data: DataFrame, validate_data: DataFrame = None) -> None:
        # data binning
        bin_info = binning(train_data, max_bin=self.max_bin)
        bin_data: DataFrame = train_data.bucketize(boundaries=bin_info)
        logger.info("data binning done")
        # predict to help guest to get the warmstart scores
        if self._model_loaded:
            pred_ctx = ctx.sub_ctx("warmstart_predict")
            self.predict(pred_ctx, train_data)

        random_seeds = self._get_seeds(ctx)
        global_random_seed = next(random_seeds)
        for tree_idx, tree_ctx in ctx.on_iterations.ctxs_range(len(self._trees), len(self._trees) + self.num_trees):
            logger.info("start to fit a host tree")
            tree = HeteroDecisionTreeHost(
                max_depth=self.max_depth,
                hist_sub=self._hist_sub,
                global_random_seed=global_random_seed,
                random_seed=next(random_seeds),
            )
            tree.booster_fit(tree_ctx, bin_data, bin_info)
            self._trees.append(tree)
            self._saved_tree.append(tree.get_model())
            self._update_feature_importance(tree.get_feature_importance())
            logger.info("fitting host decision tree {} done".format(tree_idx))

    def predict(self, ctx: Context, predict_data: DataFrame) -> None:
        predict_leaf_host(ctx, self._trees, predict_data)

    def _get_hyper_param(self) -> dict:
        return {
            "num_trees": self.num_trees,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_bin": self.max_bin,
        }

    def from_model(self, model: dict):
        trees = model["trees"]
        self._saved_tree = trees
        self._trees = [HeteroDecisionTreeHost.from_model(tree) for tree in trees]
        self._model_loaded = True
        return self
