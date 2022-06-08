import copy
import functools
import numpy as np
from typing import List
from operator import itemgetter
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.feature.sparse_vector import SparseVector
from federatedml.feature.fate_element_type import NoneType
from federatedml.ensemble import HeteroSecureBoostingTreeGuest
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.param.boosting_param import HomoSecureBoostParam
from federatedml.ensemble.boosting.homo_boosting import HomoBoostingClient
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import ObjectiveMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import FeatureImportanceInfo
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.feature_importance import FeatureImportance
from federatedml.ensemble.basic_algorithms.decision_tree.homo.homo_decision_tree_client import HomoDecisionTreeClient

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
        self.feature_importance_ = {}
        self.model_param = HomoSecureBoostParam()

        # memory back end
        self.backend = consts.DISTRIBUTED_BACKEND
        self.bin_arr, self.sample_id_arr = None, None

        # mo tree
        self.multi_mode = consts.SINGLE_OUTPUT

    def _init_model(self, boosting_param: HomoSecureBoostParam):

        super(HomoSecureBoostingTreeClient, self)._init_model(boosting_param)
        self.use_missing = boosting_param.use_missing
        self.zero_as_missing = boosting_param.zero_as_missing
        self.tree_param = boosting_param.tree_param
        self.backend = boosting_param.backend
        self.multi_mode = boosting_param.multi_mode

        if self.use_missing:
            self.tree_param.use_missing = self.use_missing
            self.tree_param.zero_as_missing = self.zero_as_missing

    def get_valid_features(self, epoch_idx, b_idx):
        valid_feature = self.transfer_inst.valid_features.get(idx=0, suffix=('valid_features', epoch_idx, b_idx))
        return valid_feature

    def compute_local_grad_and_hess(self, y_hat):

        loss_method = self.loss
        if self.task_type == consts.CLASSIFICATION:
            grad_and_hess = self.y.join(y_hat, lambda y, f_val:
                                        (loss_method.compute_grad(y, loss_method.predict(f_val)),
                                         loss_method.compute_hess(y, loss_method.predict(f_val))))
        else:
            grad_and_hess = self.y.join(y_hat, lambda y, f_val:
                                        (loss_method.compute_grad(y, f_val),
                                         loss_method.compute_hess(y, f_val)))

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
            if fid not in self.feature_importance_:
                self.feature_importance_[fid] = tree_feature_importance[fid]
            else:
                self.feature_importance_[fid] += tree_feature_importance[fid]

    """
    Functions for memory backends
    """

    @staticmethod
    def _handle_zero_as_missing(inst, feat_num, missing_bin_idx):
        """
        This for use_missing + zero_as_missing case
        """

        sparse_vec = inst.features.sparse_vec
        arr = np.zeros(feat_num, dtype=np.uint8) + missing_bin_idx
        for k, v in sparse_vec.items():
            if v != NoneType():
                arr[k] = v
        inst.features = arr
        return inst

    @staticmethod
    def _map_missing_bin(inst, bin_index):

        arr_bin = copy.deepcopy(inst.features)
        arr_bin[arr_bin == NoneType()] = bin_index
        inst.features = arr_bin
        return inst

    @staticmethod
    def _fill_nan(inst):
        arr = copy.deepcopy(inst.features)
        nan_index = np.isnan(arr)
        arr = arr.astype(np.object)
        arr[nan_index] = NoneType()
        inst.features = arr
        return inst

    @staticmethod
    def _sparse_recover(inst, feat_num):

        arr = np.zeros(feat_num)
        for k, v in inst.features.sparse_vec.items():
            arr[k] = v
        inst.features = arr
        return inst

    def data_preporcess(self, data_inst):
        """
        override parent function
        """
        need_transform_to_sparse = self.backend == consts.DISTRIBUTED_BACKEND or \
            (self.backend == consts.MEMORY_BACKEND and self.use_missing and self.zero_as_missing)

        backup_schema = copy.deepcopy(data_inst.schema)
        if self.backend == consts.MEMORY_BACKEND:
            # memory backend only support dense format input
            data_example = data_inst.take(1)[0][1]
            if isinstance(data_example.features, SparseVector):
                recover_func = functools.partial(self._sparse_recover, feat_num=len(data_inst.schema['header']))
                data_inst = data_inst.mapValues(recover_func)
                data_inst.schema = backup_schema

        if need_transform_to_sparse:
            data_inst = self.data_alignment(data_inst)
        elif self.use_missing:
            # fill nan
            data_inst = data_inst.mapValues(self._fill_nan)
            data_inst.schema = backup_schema

        self.data_bin, self.bin_split_points, self.bin_sparse_points = self.federated_binning(data_inst)

        if self.backend == consts.MEMORY_BACKEND:

            if self.use_missing and self.zero_as_missing:
                feat_num = len(self.bin_split_points)
                func = functools.partial(self._handle_zero_as_missing, feat_num=feat_num, missing_bin_idx=self.bin_num)
                self.data_bin = self.data_bin.mapValues(func)
            elif self.use_missing:  # use missing only
                missing_bin_index = self.bin_num
                func = functools.partial(self._map_missing_bin, bin_index=missing_bin_index)
                self.data_bin = self.data_bin.mapValues(func)

            self._collect_data_arr(self.data_bin)

    def _collect_data_arr(self, bin_arr_table):

        bin_arr = []
        id_list = []
        for id_, inst in bin_arr_table.collect():
            bin_arr.append(inst.features)
            id_list.append(id_)
        self.bin_arr = np.asfortranarray(np.stack(bin_arr, axis=0).astype(np.uint8))
        self.sample_id_arr = np.array(id_list)

    def preprocess(self):

        if self.multi_mode == consts.MULTI_OUTPUT:
            self.booster_dim = 1
            LOGGER.debug('multi mode tree dim reset to 1')

    def fit_a_learner(self, epoch_idx: int, booster_dim: int):

        valid_features = self.get_valid_features(epoch_idx, booster_dim)
        LOGGER.debug('valid features are {}'.format(valid_features))

        if self.cur_epoch_idx != epoch_idx:
            # update g/h every epoch
            self.grad_and_hess = self.compute_local_grad_and_hess(self.y_hat)
            self.cur_epoch_idx = epoch_idx

        if self.multi_mode == consts.MULTI_OUTPUT:
            g_h = self.grad_and_hess
        else:
            g_h = self.get_subtree_grad_and_hess(self.grad_and_hess, booster_dim)

        flow_id = self.generate_flowid(epoch_idx, booster_dim)
        new_tree = HomoDecisionTreeClient(
            self.tree_param,
            self.data_bin,
            self.bin_split_points,
            self.bin_sparse_points,
            g_h,
            valid_feature=valid_features,
            epoch_idx=epoch_idx,
            role=self.role,
            flow_id=flow_id,
            tree_idx=booster_dim,
            mode='train')

        if self.backend == consts.DISTRIBUTED_BACKEND:
            new_tree.fit()
        elif self.backend == consts.MEMORY_BACKEND:
            # memory backend needed variable
            LOGGER.debug('running memory fit')
            new_tree.arr_bin_data = self.bin_arr
            new_tree.bin_num = self.bin_num
            new_tree.sample_id_arr = self.sample_id_arr
            new_tree.memory_fit()

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
            return np.sum(weights * learning_rate, axis=0) + init_score

    def fast_homo_tree_predict(self, data_inst, ret_format='std'):

        assert ret_format in ['std', 'raw'], 'illegal ret format'

        LOGGER.info('running fast homo tree predict')
        to_predict_data = self.data_and_header_alignment(data_inst)
        tree_list = []
        rounds = len(self.boosting_model_list) // self.booster_dim
        for idx in range(0, rounds):
            for booster_idx in range(self.booster_dim):
                model = self.load_learner(self.booster_meta,
                                          self.boosting_model_list[idx * self.booster_dim + booster_idx],
                                          idx, booster_idx)
                tree_list.append(model)

        func = functools.partial(self.predict_helper, tree_list=tree_list, init_score=self.init_score,
                                 zero_as_missing=self.zero_as_missing, use_missing=self.use_missing,
                                 learning_rate=self.learning_rate, class_num=self.booster_dim)
        predict_rs = to_predict_data.mapValues(func)

        if ret_format == 'std':
            return self.score_to_predict_result(data_inst, predict_rs)
        elif ret_format == 'raw':
            return predict_rs
        else:
            raise ValueError('illegal ret format')

    @assert_io_num_rows_equal
    def predict(self, data_inst, ret_format='std'):
        return self.fast_homo_tree_predict(data_inst, ret_format=ret_format)

    def generate_summary(self) -> dict:

        summary = {'feature_importance': make_readable_feature_importance(self.feature_name_fid_mapping,
                                                                          self.feature_importance_),
                   'validation_metrics': self.callback_variables.validation_summary}

        return summary

    def load_learner(self, model_meta, model_param, epoch_idx, booster_idx):
        tree_inst = HomoDecisionTreeClient(tree_param=self.tree_param, mode='predict')
        tree_inst.load_model(model_meta=model_meta, model_param=model_param)
        return tree_inst

    def load_feature_importance(self, feat_importance_param):

        param = list(feat_importance_param)
        rs_dict = {}
        for fp in param:
            key = fp.fid
            importance = FeatureImportance()
            importance.from_protobuf(fp)
            rs_dict[key] = importance
        self.feature_importance_ = rs_dict
        LOGGER.debug('load feature importance": {}'.format(self.feature_importance_))

    def set_model_param(self, model_param):

        self.boosting_model_list = list(model_param.trees_)
        self.init_score = np.array(list(model_param.init_score))
        self.classes_ = list(map(int, model_param.classes_))
        self.booster_dim = model_param.tree_dim
        self.num_classes = model_param.num_classes
        self.feature_name_fid_mapping.update(model_param.feature_name_fid_mapping)
        self.load_feature_importance(model_param.feature_importances)
        # initialize loss function
        self.loss = self.get_loss_function()

    def set_model_meta(self, model_meta):

        if not self.is_warm_start:
            self.boosting_round = model_meta.num_trees
            self.n_iter_no_change = model_meta.n_iter_no_change
            self.tol = model_meta.tol
            self.bin_num = model_meta.quantile_meta.bin_num

        self.learning_rate = model_meta.learning_rate
        self.booster_meta = model_meta.tree_meta
        self.objective_param.objective = model_meta.objective_meta.objective
        self.objective_param.params = list(model_meta.objective_meta.param)
        self.task_type = model_meta.task_type

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

        feature_importance = list(self.feature_importance_.items())
        feature_importance = sorted(feature_importance, key=itemgetter(1), reverse=True)
        feature_importance_param = []
        for fid, importance in feature_importance:
            feature_importance_param.append(FeatureImportanceInfo(fid=fid,
                                                                  fullname=self.feature_name_fid_mapping[fid],
                                                                  sitename=self.role,
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
        model_meta.use_missing = self.use_missing
        model_meta.zero_as_missing = self.zero_as_missing
        model_meta.module = 'HomoSecureBoost'

        meta_name = "HomoSecureBoostingTreeGuestMeta"

        return meta_name, model_meta
