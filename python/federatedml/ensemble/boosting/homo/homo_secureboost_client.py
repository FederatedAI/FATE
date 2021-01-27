import numpy as np
import functools
from typing import List
from operator import itemgetter
from federatedml.ensemble.boosting.boosting_core.homo_boosting import HomoBoostingClient
from federatedml.param.boosting_param import HomoSecureBoostParam
from federatedml.ensemble.basic_algorithms.decision_tree.homo.homo_decision_tree_client import HomoDecisionTreeClient
from federatedml.util import consts
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import ObjectiveMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import FeatureImportanceInfo
from federatedml.ensemble import HeteroSecureBoostingTreeGuest
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.util import LOGGER


make_readable_feature_importance = HeteroSecureBoostingTreeGuest.make_readable_feature_importance


class HomoSecureBoostingTreeClient(HomoBoostingClient):

    def __init__(self):
        super(HomoSecureBoostingTreeClient, self).__init__()
        self.model_name = 'HomoSecureBoost'
        self.tree_param = None  # decision tree param
        self.use_missing = False
        self.zero_as_missing = False
        self.cur_epoch_idx = -1
        self.grad_and_hess = None
        self.feature_importance = {}
        self.model_param = HomoSecureBoostParam()

    def _init_model(self, boosting_param: HomoSecureBoostParam):

        super(HomoSecureBoostingTreeClient, self)._init_model(boosting_param)
        self.use_missing = boosting_param.use_missing
        self.zero_as_missing = boosting_param.zero_as_missing
        self.tree_param = boosting_param.tree_param

        if self.use_missing:
            self.tree_param.use_missing = self.use_missing
            self.tree_param.zero_as_missing = self.zero_as_missing

    def get_valid_features(self, epoch_idx, b_idx):
        valid_feature = self.transfer_inst.valid_features.get(idx=0, suffix=('valid_features', epoch_idx, b_idx))
        return valid_feature

    def compute_local_grad_and_hess(self, y_hat):

        loss_method = self.loss
        if self.task_type == consts.CLASSIFICATION:
            grad_and_hess = self.y.join(y_hat, lambda y,  f_val:\
                 (loss_method.compute_grad(y,  loss_method.predict(f_val)),\
                 loss_method.compute_hess(y,  loss_method.predict(f_val))))
        else:
            grad_and_hess = self.y.join(y_hat, lambda y,  f_val:
            (loss_method.compute_grad(y,  f_val),
             loss_method.compute_hess(y,  f_val)))

        return grad_and_hess

    @staticmethod
    def get_subtree_grad_and_hess(g_h, t_idx: int):
        """
        grad and hess of sub tree
        """
        LOGGER.info("get grad and hess of tree {}".format(t_idx))
        grad_and_hess_subtree = g_h.mapValues(
            lambda grad_and_hess: (grad_and_hess[0][t_idx], grad_and_hess[1][t_idx]))
        return grad_and_hess_subtree

    def update_feature_importance(self, tree_feature_importance):

        for fid in tree_feature_importance:
            if fid not in self.feature_importance:
                self.feature_importance[fid] = tree_feature_importance[fid]
            else:
                self.feature_importance[fid] += tree_feature_importance[fid]

    def fit_a_booster(self, epoch_idx: int, booster_dim: int):

        valid_features = self.get_valid_features(epoch_idx, booster_dim)
        LOGGER.debug('valid features are {}'.format(valid_features))

        if self.cur_epoch_idx != epoch_idx:
            # update g/h every epoch
            self.grad_and_hess = self.compute_local_grad_and_hess(self.y_hat)
            self.cur_epoch_idx = epoch_idx

        subtree_g_h = self.get_subtree_grad_and_hess(self.grad_and_hess, booster_dim)
        flow_id = self.generate_flowid(epoch_idx, booster_dim)
        new_tree = HomoDecisionTreeClient(self.tree_param, self.data_bin, self.bin_split_points,
                                          self.bin_sparse_points, subtree_g_h, valid_feature=valid_features
                                          , epoch_idx=epoch_idx, role=self.role, flow_id=flow_id, tree_idx= \
                                          booster_dim, mode='train')
        new_tree.fit()
        self.update_feature_importance(new_tree.get_feature_importance())

        return new_tree

    @staticmethod
    def predict_helper(data, tree_list: List[HomoDecisionTreeClient], init_score, zero_as_missing, use_missing,
                       learning_rate, class_num=1):

        weight_list = []
        for tree in tree_list:
            weight = tree.traverse_tree(data, tree.tree_node, use_missing=use_missing, zero_as_missing=zero_as_missing)
            weight_list.append(weight)

        weights = np.array(weight_list)

        if class_num > 2:
            weights = weights.reshape((-1, class_num))
            return np.sum(weights * learning_rate, axis=0) + init_score
        else:
            return float(np.sum(weights * learning_rate, axis=0) + init_score)

    def fast_homo_tree_predict(self, data_inst):

        LOGGER.info('running fast homo tree predict')
        to_predict_data = self.data_and_header_alignment(data_inst)
        tree_list = []
        rounds = len(self.boosting_model_list) // self.booster_dim
        for idx in range(0, rounds):
            for booster_idx in range(self.booster_dim):
                model = self.load_booster(self.booster_meta,
                                          self.boosting_model_list[idx * self.booster_dim + booster_idx],
                                          idx, booster_idx)
                tree_list.append(model)

        func = functools.partial(self.predict_helper, tree_list=tree_list, init_score=self.init_score,
                                 zero_as_missing=self.zero_as_missing, use_missing=self.use_missing,
                                 learning_rate=self.learning_rate, class_num=self.booster_dim)
        predict_rs = to_predict_data.mapValues(func)
        return self.score_to_predict_result(data_inst, predict_rs)

    @assert_io_num_rows_equal
    def predict(self, data_inst):
        return self.fast_homo_tree_predict(data_inst)

    def generate_summary(self) -> dict:

        summary = {'feature_importance': make_readable_feature_importance(self.feature_name_fid_mapping,
                                                                          self.feature_importance),
                   'validation_metrics': None if not self.validation_strategy else self.validation_strategy.summary()}

        return summary

    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):
        tree_inst = HomoDecisionTreeClient(tree_param=self.tree_param, mode='predict')
        tree_inst.load_model(model_meta=model_meta, model_param=model_param)
        return tree_inst

    def set_model_param(self, model_param):
        self.boosting_model_list = list(model_param.trees_)
        self.init_score = np.array(list(model_param.init_score))
        self.classes_ = list(model_param.classes_)
        self.booster_dim = model_param.tree_dim
        self.num_classes = model_param.num_classes
        self.feature_name_fid_mapping.update(model_param.feature_name_fid_mapping)

    def set_model_meta(self, model_meta):

        self.booster_meta = model_meta.tree_meta
        self.learning_rate = model_meta.learning_rate
        self.boosting_round = model_meta.num_trees
        self.bin_num = model_meta.quantile_meta.bin_num
        self.objective_param.objective = model_meta.objective_meta.objective
        self.objective_param.params = list(model_meta.objective_meta.param)
        self.task_type = model_meta.task_type
        self.n_iter_no_change = model_meta.n_iter_no_change
        self.tol = model_meta.tol

    def get_model_param(self):
        model_param = BoostingTreeModelParam()
        model_param.tree_num = len(list(self.boosting_model_list))
        model_param.tree_dim = self.booster_dim
        model_param.trees_.extend(self.boosting_model_list)
        model_param.init_score.extend(self.init_score)
        model_param.classes_.extend(map(str, self.classes_))
        model_param.num_classes = self.num_classes
        model_param.best_iteration = -1
        model_param.model_name = consts.HOMO_SBT

        feature_importance = list(self.feature_importance.items())
        feature_importance = sorted(feature_importance, key=itemgetter(1), reverse=True)
        feature_importance_param = []
        for fid, importance in feature_importance:
            feature_importance_param.append(FeatureImportanceInfo(fid=fid,
                                                                  importance=importance.importance,
                                                                  importance2=importance.importance_2,
                                                                  main=importance.main_type
                                                                  ))

        model_param.feature_importances.extend(feature_importance_param)

        model_param.feature_name_fid_mapping.update(self.feature_name_fid_mapping)

        param_name = "HomoSecureBoostingTreeGuestParam"

        return param_name, model_param

    def get_model_meta(self):
        model_meta = BoostingTreeModelMeta()
        model_meta.tree_meta.CopyFrom(self.booster_meta)
        model_meta.learning_rate = self.learning_rate
        model_meta.num_trees = self.boosting_round
        model_meta.quantile_meta.CopyFrom(QuantileMeta(bin_num=self.bin_num))
        model_meta.objective_meta.CopyFrom(ObjectiveMeta(objective=self.objective_param.objective,
                                                         param=self.objective_param.params))
        model_meta.task_type = self.task_type
        model_meta.n_iter_no_change = self.n_iter_no_change
        model_meta.tol = self.tol

        meta_name = "HomoSecureBoostingTreeGuestMeta"

        return meta_name, model_meta

