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
import numpy as np
from typing import Optional
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.ensemble.algo.secureboost.hetero._base import HeteroBoostingTree
from fate.ml.ensemble.learner.decision_tree.hetero.guest import HeteroDecisionTreeGuest
from fate.ml.ensemble.utils.binning import binning
from fate.ml.ensemble.learner.decision_tree.tree_core.loss import OBJECTIVE, get_task_info, MULTI_CE
from fate.ml.ensemble.algo.secureboost.common.predict import predict_leaf_guest
from fate.ml.utils.predict_tools import compute_predict_details, PREDICT_SCORE, LABEL, BINARY, MULTI, REGRESSION
import logging


logger = logging.getLogger(__name__)


class HeteroSecureBoostGuest(HeteroBoostingTree):
    def __init__(
        self,
        num_trees=3,
        learning_rate=0.3,
        max_depth=3,
        objective="binary:bce",
        num_class=3,
        max_bin=32,
        l2=0.1,
        l1=0,
        min_impurity_split=1e-2,
        min_sample_split=2,
        min_leaf_node=1,
        min_child_weight=1,
        gh_pack=True,
        split_info_pack=True,
        hist_sub=True,
    ):
        super().__init__()
        self.num_trees = num_trees
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.objective = objective
        self.max_bin = max_bin

        # regularization
        self.l2 = l2
        self.l1 = l1
        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node
        self.min_child_weight = min_child_weight

        # running var
        self.num_class = num_class
        self._accumulate_scores = None
        self._tree_dim = 1  # tree dimension, if is multilcass task, tree dim > 1
        self._loss_func = None
        self._train_predict = None
        self._hist_sub = hist_sub

        # encryption
        self._encrypt_kit = None
        self._gh_pack = gh_pack
        self._split_info_pack = split_info_pack

        # reg score
        self._init_score = None

        # model loaded
        self._model_loaded = False

    def _prepare_parameter(self):
        self._tree_dim = self.num_class if self.objective == "multiclass:ce" else 1

    def _get_loss_func(self, objective: str) -> Optional[object]:
        # to lowercase
        objective = objective.lower()
        if objective == MULTI_CE:
            raise ValueError(
                "multi:ce objective is not supported in the beta version, will be added in the next version"
            )
        assert (
            objective in OBJECTIVE
        ), f"objective {objective} not found, supported objective: {list(OBJECTIVE.keys())}"
        obj_class = OBJECTIVE[objective]
        loss_func = obj_class()
        return loss_func

    def _compute_gh(self, data: DataFrame, scores: DataFrame, loss_func):
        label = data.label
        predict = loss_func.predict(scores)
        gh = data.create_frame()
        loss_func.compute_grad(gh, label, predict)
        loss_func.compute_hess(gh, label, predict)
        return gh

    def _check_encrypt_kit(self, ctx: Context):
        if self._encrypt_kit is None:
            # make sure cipher is initialized
            kit = ctx.cipher.phe.setup()
            self._encrypt_kit = kit

        if not self._encrypt_kit.can_support_negative_number:
            self._gh_pack = True
            logger.info("current encrypt method cannot support neg num, gh pack is forced to be True")
        if not self._encrypt_kit.can_support_squeeze:
            self._split_info_pack = False
            logger.info("current encrypt method cannot support compress, split info pack is forced to be False")
        if not self._encrypt_kit.can_support_pack:
            self._gh_pack = False
            self._split_info_pack = False
            logger.info("current encrypt method cannot support pack, gh pack is forced to be False")
        return kit

    def get_train_predict(self):
        return self._train_predict

    def get_tree(self, idx):
        return self._trees[idx]

    def _init_sample_scores(self, ctx: Context, label, train_data: DataFrame):
        task_type = self.objective.split(":")[0]
        pred_ctx = ctx.sub_ctx("warmstart_predict")
        if self._model_loaded:
            logger.info("prepare warmstarting score")
            self._accumulate_scores = self.predict(pred_ctx, train_data, ret_std_format=False)
            self._accumulate_scores = self._accumulate_scores.loc(
                train_data.get_indexer(target="sample_id"), preserve_order=True
            )
        else:
            if task_type == REGRESSION:
                self._accumulate_scores, avg_score = self._loss_func.initialize(label)
                if self._init_score is None:
                    self._init_score = avg_score
            elif task_type == MULTI:
                self._accumulate_scores = self._loss_func.initialize(label, self.num_class)
            else:
                self._accumulate_scores = self._loss_func.initialize(label)

    def _check_label(self, label: DataFrame):
        label_df = label.as_pd_df()[label.schema.label_name]
        if self.objective == "multi:ce":
            if self.num_class is None or self.num_class <= 2:
                raise ValueError(
                    f"num_class should be set and greater than 2 for multi:ce objective, but got {self.num_class}"
                )
            label_set = set(np.unique(label_df))
            if len(label_set) > self.num_class:
                raise ValueError(
                    f"num_class should be greater than or equal to the number of unique label in provided train data, but got {self.num_class} and {len(label_set)}"
                )
            if max(label_set) - 1 > self.num_class:
                raise ValueError(
                    f"the max label index in the provided train data should be less than or equal to num_class - 1, but got index {max(label_set)} which is > {self.num_class}"
                )

        elif self.objective == "binary:bce":
            label_set = set(np.unique(label_df))
            assert len(label_set) == 2, f"binary classification task should have 2 unique label, but got {label_set}"
            assert (
                0 in label_set and 1 in label_set
            ), f"binary classification task should have label 0 and 1, but got {label_set}"
            self.num_class = 2
        else:
            self.num_class = None

    def get_task_info(self):
        task_type = get_task_info(self.objective)
        if task_type == BINARY:
            classes = [0, 1]
        elif task_type == REGRESSION:
            classes = None
        return task_type, classes

    def fit(self, ctx: Context, train_data: DataFrame, validate_data: DataFrame = None) -> None:
        """
        Train model with train data and validate data.

        Parameters
        ----------
        ctx: Context
            FATE Context object
        train_data: DataFrame
            Train data used to fit model.
        validate_data: DataFrame, optional
            Validate data used to evaluate model performance during training process.
        """

        # data binning
        bin_info = binning(train_data, max_bin=self.max_bin)
        bin_data: DataFrame = train_data.bucketize(boundaries=bin_info)
        self._check_label(bin_data.label)

        # init loss func & scores
        self._loss_func = self._get_loss_func(self.objective)
        label = bin_data.label
        self._init_sample_scores(ctx, label, train_data)

        # init encryption kit
        self._encrypt_kit = self._check_encrypt_kit(ctx)

        # start tree fittingf
        for tree_idx, tree_ctx in ctx.on_iterations.ctxs_range(len(self._trees), len(self._trees) + self.num_trees):
            # compute gh of current iter
            logger.info("start to fit a guest tree")
            gh = self._compute_gh(bin_data, self._accumulate_scores, self._loss_func)
            tree = HeteroDecisionTreeGuest(
                max_depth=self.max_depth,
                l2=self.l2,
                l1=self.l1,
                min_impurity_split=self.min_impurity_split,
                min_sample_split=self.min_sample_split,
                min_leaf_node=self.min_leaf_node,
                min_child_weight=self.min_child_weight,
                objective=self.objective,
                gh_pack=self._gh_pack,
                split_info_pack=self._split_info_pack,
                hist_sub=self._hist_sub,
            )
            tree.set_encrypt_kit(self._encrypt_kit)
            tree.booster_fit(tree_ctx, bin_data, gh, bin_info)
            # accumulate scores of cur boosting round
            scores = tree.get_sample_predict_weights()
            assert len(scores) == len(
                self._accumulate_scores
            ), f"tree predict scores length {len(scores)} not equal to accumulate scores length {len(self._accumulate_scores)}."
            scores = scores.loc(self._accumulate_scores.get_indexer(target="sample_id"), preserve_order=True)
            self._accumulate_scores = self._accumulate_scores + scores * self.learning_rate
            self._trees.append(tree)
            self._saved_tree.append(tree.get_model())
            self._update_feature_importance(tree.get_feature_importance())
            logger.info("fitting guest decision tree {} done".format(tree_idx))

        # compute train predict using cache scores
        train_predict: DataFrame = self._loss_func.predict(self._accumulate_scores)
        train_predict = train_predict.loc(train_data.get_indexer(target="sample_id"), preserve_order=True)
        train_predict.label = train_data.label
        task_type, classes = self.get_task_info()
        train_predict.rename(columns={"score": PREDICT_SCORE})
        self._train_predict = compute_predict_details(train_predict, task_type, classes)

    def predict(self, ctx: Context, predict_data: DataFrame, predict_leaf=False, ret_std_format=True) -> DataFrame:
        """
        predict function

        Parameters
        ----------
        ctx: Context
            FATE Context object
        predict_data: DataFrame
            Data used to predict.
        predict_leaf: bool, optional
            Whether to predict and return leaf index.
        ret_std_format: bool, optional
            Whether to return result in a FATE standard format which contains more details.
        """

        task_type, classes = self.get_task_info()
        leaf_pos = predict_leaf_guest(ctx, self._trees, predict_data)
        if predict_leaf:
            return leaf_pos
        result = self._sum_leaf_weights(leaf_pos, self._trees, self.learning_rate, self._loss_func)

        if task_type == REGRESSION:
            logger.debug("regression task, add init score")
            result = result + self._init_score

        if ret_std_format:
            # align table
            result: DataFrame = result.loc(predict_data.get_indexer(target="sample_id"), preserve_order=True)
            ret_frame = result.create_frame()
            if predict_data.schema.label_name is not None:
                ret_frame.label = predict_data.label
            ret_frame[PREDICT_SCORE] = result["score"]

            return compute_predict_details(ret_frame, task_type, classes)
        else:
            return result

    def _get_hyper_param(self) -> dict:
        return {
            "num_trees": self.num_trees,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "objective": self.objective,
            "max_bin": self.max_bin,
            "l2": self.l2,
            "num_class": self.num_class,
        }

    def get_model(self) -> dict:
        ret_dict = super().get_model()
        ret_dict["init_score"] = self._init_score
        return ret_dict

    def from_model(self, model: dict):
        trees = model["trees"]
        self._saved_tree = trees
        self._trees = [HeteroDecisionTreeGuest.from_model(tree) for tree in trees]
        hyper_parameter = model["hyper_param"]

        # these parameter are related to predict
        self.learning_rate = hyper_parameter["learning_rate"]
        self.num_class = hyper_parameter["num_class"]
        self.objective = hyper_parameter["objective"]
        self._init_score = float(model["init_score"]) if model["init_score"] is not None else None
        # initialize
        self._prepare_parameter()
        self._loss_func = self._get_loss_func(self.objective)
        # for warmstart
        self._model_loaded = True

        return self
