from operator import itemgetter
import numpy as np
from arch.api.utils import log_utils
from typing import List
import functools
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import ObjectiveMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import FeatureImportanceInfo
from federatedml.ensemble.boosting.boosting_core import HeteroBoostingGuest
from federatedml.param.boosting_param import HeteroSecureBoostParam
from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeGuest
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.hetero_secure_boosting_predict_transfer_variable import \
    HeteroSecureBoostTransferVariable
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.util.fate_operator import generate_anonymous

LOGGER = log_utils.getLogger()


class HeteroSecureBoostGuest(HeteroBoostingGuest):

    def __init__(self):
        super(HeteroSecureBoostGuest, self).__init__()
        self.tree_param = None  # decision tree param
        self.use_missing = False
        self.zero_as_missing = False
        self.cur_epoch_idx = -1
        self.grad_and_hess = None
        self.feature_importances_ = {}
        self.model_param = HeteroSecureBoostParam()
        self.complete_secure = False

        self.data_alignment_map = {}

        self.predict_transfer_inst = HeteroSecureBoostTransferVariable()

    def _init_model(self, param: HeteroSecureBoostParam):
        super(HeteroSecureBoostGuest, self)._init_model(param)
        self.tree_param = param.tree_param
        self.use_missing = param.use_missing
        self.zero_as_missing = param.zero_as_missing
        self.complete_secure = param.complete_secure

        if self.use_missing:
            self.tree_param.use_missing = self.use_missing
            self.tree_param.zero_as_missing = self.zero_as_missing

    def compute_grad_and_hess(self, y_hat, y):
        LOGGER.info("compute grad and hess")
        loss_method = self.loss
        if self.task_type == consts.CLASSIFICATION:
            grad_and_hess = y.join(y_hat, lambda y, f_val: \
                (loss_method.compute_grad(y, loss_method.predict(f_val)), \
                 loss_method.compute_hess(y, loss_method.predict(f_val))))
        else:
            grad_and_hess = y.join(y_hat, lambda y, f_val:
            (loss_method.compute_grad(y, f_val),
             loss_method.compute_hess(y, f_val)))

        return grad_and_hess

    @staticmethod
    def get_grad_and_hess(g_h, dim=0):
        LOGGER.info("get grad and hess of tree {}".format(dim))
        grad_and_hess_subtree = g_h.mapValues(
            lambda grad_and_hess: (grad_and_hess[0][dim], grad_and_hess[1][dim]))
        return grad_and_hess_subtree

    def update_feature_importance(self, tree_feature_importance):
        for fid in tree_feature_importance:
            if fid not in self.feature_importances_:
                self.feature_importances_[fid] = 0

            self.feature_importances_[fid] += tree_feature_importance[fid]

    def fit_a_booster(self, epoch_idx: int, booster_dim: int):

        if self.cur_epoch_idx != epoch_idx:
            self.grad_and_hess = self.compute_grad_and_hess(self.y_hat, self.y)
            self.cur_epoch_idx = epoch_idx

        g_h = self.get_grad_and_hess(self.grad_and_hess, booster_dim)

        tree = HeteroDecisionTreeGuest(tree_param=self.tree_param)
        tree.set_input_data(self.data_bin, self.bin_split_points, self.bin_sparse_points)
        tree.set_grad_and_hess(g_h)
        tree.set_encrypter(self.encrypter)
        tree.set_encrypted_mode_calculator(self.encrypted_calculator)
        tree.set_valid_features(self.sample_valid_features())
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_dim))
        tree.set_host_party_idlist(self.component_properties.host_party_idlist)
        tree.set_runtime_idx(self.component_properties.local_partyid)

        if self.cur_epoch_idx == 0 and self.complete_secure:
            tree.set_as_complete_secure_tree()

        tree.fit()

        self.update_feature_importance(tree.get_feature_importance())

        return tree

    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):
        tree = HeteroDecisionTreeGuest(self.tree_param)
        tree.load_model(model_meta, model_param)
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_idx))
        tree.set_runtime_idx(self.component_properties.local_partyid)
        tree.set_host_party_idlist(self.component_properties.host_party_idlist)
        return tree

    def generate_summary(self) -> dict:

        summary = {'loss_history': self.history_loss, 'best_iteration': self.validation_strategy.best_iteration,
                   'feature_importance': self.make_readable_feature_importance(self.feature_name_fid_mapping,
                                                                               self.feature_importances_),
                   'validation_metrics': self.validation_strategy.summary(),
                   'is_converged': self.is_converged}

        return summary

    @staticmethod
    def generate_leaf_pos_dict(x, tree_num):
        """
        x: just occupy the first parameter position
        return: a numpy array record sample pos, and a counter counting how many trees reach a leaf node
        """
        node_pos = np.zeros(tree_num, dtype=np.int64) + 0
        reach_leaf_node = np.zeros(tree_num, dtype=np.bool)
        return {'node_pos': node_pos, 'reach_leaf_node': reach_leaf_node}

    @staticmethod
    def traverse_a_tree(tree: HeteroDecisionTreeGuest, sample, cur_node_idx):

        reach_leaf = False
                                                      # only need nid here, predict state is not needed
        rs = tree.traverse_tree(tree_=tree.tree_node, data_inst=sample, predict_state=(cur_node_idx, -1),
                                decoder=tree.decode, sitename=tree.sitename, use_missing=tree.use_missing,
                                split_maskdict=tree.split_maskdict, missing_dir_maskdict=tree.missing_dir_maskdict,
                                return_leaf_id=True)

        if not isinstance(rs, tuple):
            reach_leaf = True
            leaf_id = rs
            return leaf_id, reach_leaf
        else:
            cur_node_idx = rs[0]
            return cur_node_idx, reach_leaf

    @staticmethod
    def make_readable_feature_importance(fid_mapping, feature_importances):
        """
        replace feature id by real feature name
        """
        new_fi = {}
        for id_ in feature_importances:

            if type(id_) == tuple:
                if 'guest' in id_[0]:
                    new_fi[fid_mapping[id_[1]]] = feature_importances[id_]
                else:
                    role, party_id = id_[0].split(':')
                    new_fi[generate_anonymous(role=role, fid=id_[1], party_id=party_id)] = feature_importances[id_]
            else:
                new_fi[fid_mapping[id_]] = feature_importances[id_]

        return new_fi

    @staticmethod
    def traverse_trees(node_pos, sample, trees: List[HeteroDecisionTreeGuest]):

        if node_pos['reach_leaf_node'].all():
            return node_pos

        for t_idx, tree in enumerate(trees):

            cur_node_idx = node_pos['node_pos'][t_idx]

            # reach leaf
            if cur_node_idx == -1:
                continue

            rs, reach_leaf = HeteroSecureBoostGuest.traverse_a_tree(tree, sample, cur_node_idx)

            if reach_leaf:
                node_pos['reach_leaf_node'][t_idx] = True

            node_pos['node_pos'][t_idx] = rs

        return node_pos

    @staticmethod
    def merge_predict_pos(node_pos1, node_pos2):

        pos_arr1 = node_pos1['node_pos']
        pos_arr2 = node_pos2['node_pos']
        stack_arr = np.stack([pos_arr1, pos_arr2])
        node_pos1['node_pos'] = np.max(stack_arr, axis=0)
        return node_pos1

    @staticmethod
    def add_y_hat(leaf_pos, init_score, learning_rate, trees: List[HeteroDecisionTreeGuest], multi_class_num=None):

        # finally node pos will hold weights
        weights = []
        for leaf_idx, tree in zip(leaf_pos, trees):
            weights.append(tree.tree_node[leaf_idx].weight)
        weights = np.array(weights)
        if multi_class_num is not None:
            weights = weights.reshape((-1, multi_class_num))
        return np.sum(weights * learning_rate, axis=0) + init_score

    @staticmethod
    def get_predict_scores(leaf_pos, learning_rate, init_score, trees: List[HeteroDecisionTreeGuest]
                           , multi_class_num=-1, predict_cache=None):

        if predict_cache:
            init_score = 0  # prevent init_score re-add

        predict_func = functools.partial(HeteroSecureBoostGuest.add_y_hat,
                                         learning_rate=learning_rate, init_score=init_score, trees=trees,
                                         multi_class_num=multi_class_num)
        predict_result = leaf_pos.mapValues(predict_func)

        if predict_cache:
            predict_result = predict_result.join(predict_cache, lambda v1, v2: v1+v2)

        return predict_result

    @staticmethod
    def save_leaf_pos_helper(v1, v2):

        reach_leaf_idx = v2['reach_leaf_node']
        select_idx = reach_leaf_idx & (v2['node_pos'] != -1)  # reach leaf and are not recorded( if recorded idx is -1)
        v1[select_idx] = v2['node_pos'][select_idx]
        return v1

    @staticmethod
    def mask_leaf_pos(v):

        reach_leaf_idx = v['reach_leaf_node']
        v['node_pos'][reach_leaf_idx] = -1
        return v

    def save_leaf_pos_and_mask_leaf_pos(self, node_pos_tb, final_leaf_pos):

        # save leaf pos
        saved_leaf_pos = final_leaf_pos.join(node_pos_tb, self.save_leaf_pos_helper)
        rest_part = final_leaf_pos.subtractByKey(saved_leaf_pos)
        final_leaf_pos = saved_leaf_pos.union(rest_part)
        # mask leaf pos
        node_pos_tb = node_pos_tb.mapValues(self.mask_leaf_pos)

        return node_pos_tb, final_leaf_pos

    def boosting_fast_predict(self, data_inst, trees: List[HeteroDecisionTreeGuest], predict_cache=None):

        tree_num = len(trees)
        generate_func = functools.partial(self.generate_leaf_pos_dict, tree_num=tree_num)
        node_pos_tb = data_inst.mapValues(generate_func)  # record node pos
        final_leaf_pos = data_inst.mapValues(lambda x: np.zeros(tree_num, dtype=np.int64) - 1)  # record final leaf pos
        traverse_func = functools.partial(self.traverse_trees, trees=trees)
        comm_round = 0

        while True:

            LOGGER.debug('cur predict round is {}'.format(comm_round))

            node_pos_tb = node_pos_tb.join(data_inst, traverse_func)
            node_pos_tb, final_leaf_pos = self.save_leaf_pos_and_mask_leaf_pos(node_pos_tb, final_leaf_pos)

            # remove sample that reaches leaves of all trees
            reach_leaf_samples = node_pos_tb.filter(lambda key, value: value['reach_leaf_node'].all())
            node_pos_tb = node_pos_tb.subtractByKey(reach_leaf_samples)

            if node_pos_tb.count() == 0:
                self.predict_transfer_inst.predict_stop_flag.remote(True, idx=-1, suffix=(comm_round, ))
                break

            self.predict_transfer_inst.predict_stop_flag.remote(False, idx=-1, suffix=(comm_round, ))
            self.predict_transfer_inst.guest_predict_data.remote(node_pos_tb, idx=-1, suffix=(comm_round, ))

            host_pos_tbs = self.predict_transfer_inst.host_predict_data.get(idx=-1, suffix=(comm_round, ))

            for host_pos_tb in host_pos_tbs:
                node_pos_tb = node_pos_tb.join(host_pos_tb, self.merge_predict_pos)

            comm_round += 1

        LOGGER.debug('federation process done')

        predict_result = self.get_predict_scores(leaf_pos=final_leaf_pos, learning_rate=self.learning_rate,
                                                 init_score=self.init_score, trees=trees,
                                                 multi_class_num=self.booster_dim, predict_cache=predict_cache)

        return predict_result

    @assert_io_num_rows_equal
    def predict(self, data_inst):

        LOGGER.info('running prediction')
        cache_dataset_key = self.predict_data_cache.get_data_key(data_inst)

        processed_data = self.data_and_header_alignment(data_inst)

        last_round = self.predict_data_cache.predict_data_last_round(cache_dataset_key)

        self.sync_predict_round(last_round + 1)

        rounds = len(self.boosting_model_list) // self.booster_dim
        trees = []
        for idx in range(last_round + 1, rounds):
            for booster_idx in range(self.booster_dim):
                tree = self.load_booster(self.booster_meta,
                                         self.boosting_model_list[idx * self.booster_dim + booster_idx],
                                         idx, booster_idx)
                trees.append(tree)

        predict_cache = None
        if last_round != -1:
            predict_cache = self.predict_data_cache.predict_data_at(cache_dataset_key, last_round)
            LOGGER.debug('load predict cache of round {}'.format(last_round))

        predict_rs = self.boosting_fast_predict(processed_data, trees=trees, predict_cache=predict_cache)
        self.predict_data_cache.add_data(cache_dataset_key, predict_rs)

        return self.score_to_predict_result(data_inst, predict_rs)

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
        meta_name = "HeteroSecureBoostingTreeGuestMeta"

        return meta_name, model_meta

    def get_model_param(self):

        model_param = BoostingTreeModelParam()
        model_param.tree_num = len(self.boosting_model_list)
        model_param.tree_dim = self.booster_dim
        model_param.trees_.extend(self.boosting_model_list)
        model_param.init_score.extend(self.init_score)
        model_param.losses.extend(self.history_loss)
        model_param.classes_.extend(map(str, self.classes_))
        model_param.num_classes = self.num_classes
        model_param.model_name = consts.HETERO_SBT
        model_param.best_iteration = -1 if self.validation_strategy is None else self.validation_strategy.best_iteration

        feature_importances = list(self.feature_importances_.items())
        feature_importances = sorted(feature_importances, key=itemgetter(1), reverse=True)
        feature_importance_param = []
        for (sitename, fid), _importance in feature_importances:
            if consts.GUEST in sitename:
                fullname = self.feature_name_fid_mapping[fid]
            else:
                role_name, party_id = sitename.split(':')
                fullname = generate_anonymous(fid=fid, party_id=party_id, role=role_name)
            feature_importance_param.append(FeatureImportanceInfo(sitename=sitename,
                                                                  fid=fid,
                                                                  importance=_importance,
                                                                  fullname=fullname))
        model_param.feature_importances.extend(feature_importance_param)

        model_param.feature_name_fid_mapping.update(self.feature_name_fid_mapping)

        param_name = "HeteroSecureBoostingTreeGuestParam"

        return param_name, model_param

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

    def set_model_param(self, model_param):
        self.boosting_model_list = list(model_param.trees_)
        self.init_score = np.array(list(model_param.init_score))
        self.history_loss = list(model_param.losses)
        self.classes_ = list(model_param.classes_)
        self.booster_dim = model_param.tree_dim
        self.num_classes = model_param.num_classes
        self.feature_name_fid_mapping.update(model_param.feature_name_fid_mapping)