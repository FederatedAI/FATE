from typing import Optional
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.ensemble.algo.secureboost.hetero._base import HeteroBoostingTree
from fate.ml.ensemble.learner.decision_tree.hetero.guest import HeteroDecisionTreeGuest
from fate.ml.ensemble.utils.binning import binning
from fate.ml.ensemble.learner.decision_tree.tree_core.loss import BCELoss, CELoss, L2Loss
from fate.ml.ensemble.algo.secureboost.common.predict import predict_leaf_guest
from fate.ml.utils.predict_tools import compute_predict_details, PREDICT_SCORE, LABEL, BINARY, MULTI, REGRESSION
import logging


logger = logging.getLogger(__name__)


OBJECTIVE = {
    "binary:bce": BCELoss,
    "multi:ce": CELoss,
    "regression:l2": L2Loss
}


class HeteroSecureBoostGuest(HeteroBoostingTree):

    def __init__(self, num_trees=3, learning_rate=0.3, max_depth=3, objective='binary:bce', num_class=3,
                 max_bin=32, encrypt_key_length=2048, l2=0.1, l1=0, min_impurity_split=1e-2, min_sample_split=2, min_leaf_node=1, min_child_weight=1) -> None:
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

        # encryption
        self._encrypt_kit = None
        self._encrypt_key_length = encrypt_key_length

        # reg score
        self._init_score = None

        # model loaded
        self._model_loaded = False

    def _prepare_parameter(self):
        self._tree_dim = self.num_class if self.objective == 'multiclass:ce' else 1

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
    
    def _init_encrypt_kit(self, ctx):
        kit = ctx.cipher.phe.setup(options={"kind": "paillier", "key_length": self._encrypt_key_length})
        return kit
    
    def get_cache_predict_score(self):
        return self._loss_func.predict(self._accumulate_scores)
    
    def get_tree(self, idx):
        return self._trees[idx]
    
    def _init_sample_scores(self, label):
        task_type = self.objective.split(":")[0]
        if task_type == REGRESSION:
            self._accumulate_scores, avg_score = self._loss_func.initialize(label)
            if self._init_score is None:
                self._init_score = avg_score
        else:
            self._accumulate_scores = self._loss_func.initialize(label)

    def fit(self, ctx: Context, train_data: DataFrame, validate_data: DataFrame = None) -> None:
        
        # data binning
        bin_info = binning(train_data, max_bin=self.max_bin)
        bin_data: DataFrame = train_data.bucketize(boundaries=bin_info)

        # init loss func & scores
        self._loss_func = self._get_loss_func(self.objective)
        label = bin_data.label
        self._init_sample_scores(label)

        # init encryption kit
        self._encrypt_kit= self._init_encrypt_kit(ctx)

        # start tree fitting
        for tree_idx, tree_ctx in ctx.on_iterations.ctxs_range(len(self._trees), len(self._trees)+self.num_trees):
            # compute gh of current iter
            logger.info('start to fit a host tree')
            gh = self._compute_gh(bin_data, self._accumulate_scores, self._loss_func)
            tree = HeteroDecisionTreeGuest(max_depth=self.max_depth, l2=self.l2, l1=self.l1, 
                                           min_impurity_split=self.min_impurity_split, min_sample_split=self.min_sample_split, 
                                           min_leaf_node=self.min_leaf_node, min_child_weight=self.min_child_weight)
            tree.set_encrypt_kit(self._encrypt_kit)
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

    def predict(self, ctx: Context, predict_data: DataFrame, predict_leaf=False, ret_std_format=True) -> DataFrame:
        
        leaf_pos = predict_leaf_guest(ctx, self._trees, predict_data)
        if predict_leaf:
            return leaf_pos
        result = self._sum_leaf_weights(leaf_pos, self._trees, self.learning_rate, self._loss_func)

        if ret_std_format:
            task_type = self.objective.split(':')[0]
            if task_type == BINARY:
                classes = [0, 1]
            elif task_type == REGRESSION:
                classes = None
                result = result + self._init_score
            # align table
            result: DataFrame = result.loc(predict_data.get_indexer(target="sample_id"), preserve_order=True)
            ret_frame = result.create_frame()
            if predict_data.schema.label_name is not None:
                ret_frame[LABEL] = predict_data.label
            ret_frame[PREDICT_SCORE] = result['score']

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
        ret_dict['init_score'] = self._init_score
        return ret_dict
    
    def from_model(self, model: dict):
        
        trees = model['trees']
        self._saved_tree = trees
        self._trees = [HeteroDecisionTreeGuest.from_model(tree) for tree in trees]
        hyper_parameter = model['hyper_param']

        # these parameter are related to predict
        self.learning_rate = hyper_parameter['learning_rate']
        self.num_class = hyper_parameter['num_class']
        self.objective = hyper_parameter['objective']
        self._init_score = float(model['init_score']) if model['init_score'] is not None else None
        # initialize
        self._prepare_parameter()
        self._loss_func = self._get_loss_func(self.objective)
        # for warmstart
        self._model_loaded = True

        return self