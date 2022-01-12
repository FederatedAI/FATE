from typing import List
import numpy as np
import functools
from federatedml.ensemble.boosting.hetero.hetero_secureboost_guest import HeteroSecureBoostingTreeGuest
from federatedml.param.boosting_param import HeteroFastSecureBoostParam
from federatedml.ensemble.basic_algorithms import HeteroFastDecisionTreeGuest
from federatedml.ensemble.boosting.hetero import hetero_fast_secureboost_plan as plan
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroFastSecureBoostingTreeGuest(HeteroSecureBoostingTreeGuest):

    def __init__(self):
        super(HeteroFastSecureBoostingTreeGuest, self).__init__()

        self.tree_num_per_party = 1
        self.guest_depth = 0
        self.host_depth = 0
        self.work_mode = consts.MIX_TREE
        self.init_tree_plan = False
        self.tree_plan = []
        self.model_param = HeteroFastSecureBoostParam()
        self.model_name = 'HeteroFastSecureBoost'

    def _init_model(self, param: HeteroFastSecureBoostParam):
        super(HeteroFastSecureBoostingTreeGuest, self)._init_model(param)
        self.tree_num_per_party = param.tree_num_per_party
        self.work_mode = param.work_mode
        self.guest_depth = param.guest_depth
        self.host_depth = param.host_depth

    def get_tree_plan(self, idx):

        if not self.init_tree_plan:
            tree_plan = plan.create_tree_plan(self.work_mode, k=self.tree_num_per_party, tree_num=self.boosting_round,
                                              host_list=self.component_properties.host_party_idlist,
                                              complete_secure=self.complete_secure)
            self.tree_plan += tree_plan
            self.init_tree_plan = True

        LOGGER.info('tree plan is {}'.format(self.tree_plan))
        return self.tree_plan[idx]

    def check_host_number(self, tree_type):
        host_num = len(self.component_properties.host_party_idlist)
        LOGGER.info('host number is {}'.format(host_num))
        if tree_type == plan.tree_type_dict['layered_tree']:
            assert host_num == 1, 'only 1 host party is allowed in layered mode'

    def fit_a_booster(self, epoch_idx: int, booster_dim: int):

        # prepare tree plan
        tree_type, target_host_id = self.get_tree_plan(epoch_idx)
        LOGGER.info('tree work mode is {}'.format(tree_type))
        self.check_host_number(tree_type)

        if self.cur_epoch_idx != epoch_idx:
            # update g/h every epoch
            self.grad_and_hess = self.compute_grad_and_hess(self.y_hat, self.y, self.data_inst)
            self.cur_epoch_idx = epoch_idx

        g_h = self.get_grad_and_hess(self.grad_and_hess, booster_dim)

        tree = HeteroFastDecisionTreeGuest(tree_param=self.tree_param)
        tree.init(flowid=self.generate_flowid(epoch_idx, booster_dim),
                  data_bin=self.data_bin, bin_split_points=self.bin_split_points, bin_sparse_points=self.bin_sparse_points,
                  grad_and_hess=g_h,
                  encrypter=self.encrypter, encrypted_mode_calculator=self.encrypted_calculator,
                  valid_features=self.sample_valid_features(),
                  host_party_list=self.component_properties.host_party_idlist,
                  runtime_idx=self.component_properties.local_partyid,
                  goss_subsample=self.enable_goss,
                  top_rate=self.top_rate, other_rate=self.other_rate,
                  task_type=self.task_type,
                  complete_secure=True if (self.cur_epoch_idx == 0 and self.complete_secure) else False,
                  cipher_compressing=self.cipher_compressing,
                  max_sample_weight=self.max_sample_weight,
                  new_ver=self.new_ver
                  )
        tree.set_tree_work_mode(tree_type, target_host_id)
        tree.set_layered_depth(self.guest_depth, self.host_depth)
        tree.fit()
        self.update_feature_importance(tree.get_feature_importance())
        return tree

    @staticmethod
    def traverse_guest_local_trees(node_pos, sample, trees: List[HeteroFastDecisionTreeGuest]):

        """
        in mix mode, a sample can reach leaf directly
        """
        new_node_pos = node_pos + 0  # avoid inplace manipulate
        for t_idx, tree in enumerate(trees):

            cur_node_idx = new_node_pos[t_idx]

            if not tree.use_guest_feat_only_predict_mode:
                continue

            rs, reach_leaf = HeteroSecureBoostingTreeGuest.traverse_a_tree(tree, sample, cur_node_idx)
            new_node_pos[t_idx] = rs

        return new_node_pos

    @staticmethod
    def merge_leaf_pos(pos1, pos2):
        return pos1 + pos2

    # this func will be called by super class's predict()
    def boosting_fast_predict(self, data_inst, trees: List[HeteroFastDecisionTreeGuest], predict_cache=None,
                              pred_leaf=False):

        LOGGER.info('fast sbt running predict')

        if self.work_mode == consts.MIX_TREE:

            LOGGER.info('running mix mode predict')

            tree_num = len(trees)
            node_pos = data_inst.mapValues(lambda x: np.zeros(tree_num, dtype=np.int64))

            # traverse local trees
            traverse_func = functools.partial(self.traverse_guest_local_trees, trees=trees)
            guest_leaf_pos = node_pos.join(data_inst, traverse_func)

            # get leaf node from other host parties
            host_leaf_pos_list = self.predict_transfer_inst.host_predict_data.get(idx=-1)

            for host_leaf_pos in host_leaf_pos_list:
                guest_leaf_pos = guest_leaf_pos.join(host_leaf_pos, self.merge_leaf_pos)

            if pred_leaf:  # predict leaf, return leaf position only
                return guest_leaf_pos
            else:
                predict_result = self.get_predict_scores(leaf_pos=guest_leaf_pos, learning_rate=self.learning_rate,
                                                         init_score=self.init_score, trees=trees,
                                                         multi_class_num=self.booster_dim, predict_cache=predict_cache)
                return predict_result
        else:
            LOGGER.debug('running layered mode predict')
            return super(HeteroFastSecureBoostingTreeGuest, self).boosting_fast_predict(data_inst, trees, predict_cache,
                                                                                        pred_leaf=pred_leaf)

    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):

        tree = HeteroFastDecisionTreeGuest(self.tree_param)
        tree.load_model(model_meta, model_param)
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_idx))
        tree.set_runtime_idx(self.component_properties.local_partyid)
        tree.set_host_party_idlist(self.component_properties.host_party_idlist)

        tree_type, target_host_id = self.get_tree_plan(epoch_idx)
        tree.set_tree_work_mode(tree_type, target_host_id)

        if self.tree_plan[epoch_idx][0] == plan.tree_type_dict['guest_feat_only']:
            LOGGER.debug('tree of epoch {} is guest only'.format(epoch_idx))
            tree.use_guest_feat_only_predict_mode()

        return tree

    def get_model_meta(self):

        _, model_meta = super(HeteroFastSecureBoostingTreeGuest, self).get_model_meta()
        meta_name = consts.HETERO_FAST_SBT_GUEST_MODEL + "Meta"
        model_meta.work_mode = self.work_mode

        return meta_name, model_meta

    def get_model_param(self):

        _, model_param = super(HeteroFastSecureBoostingTreeGuest, self).get_model_param()
        param_name = consts.HETERO_FAST_SBT_GUEST_MODEL + 'Param'
        model_param.tree_plan.extend(plan.encode_plan(self.tree_plan))
        model_param.model_name = consts.HETERO_FAST_SBT_MIX if self.work_mode == consts.MIX_TREE else \
                                 consts.HETERO_FAST_SBT_LAYERED

        return param_name, model_param

    def set_model_meta(self, model_meta):
        super(HeteroFastSecureBoostingTreeGuest, self).set_model_meta(model_meta)
        self.work_mode = model_meta.work_mode

    def set_model_param(self, model_param):
        super(HeteroFastSecureBoostingTreeGuest, self).set_model_param(model_param)
        self.tree_plan = plan.decode_plan(model_param.tree_plan)