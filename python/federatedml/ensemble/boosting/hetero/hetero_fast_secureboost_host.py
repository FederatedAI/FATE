from typing import List
import numpy as np
import functools
from operator import itemgetter
from federatedml.ensemble.boosting.hetero.hetero_secureboost_host import HeteroSecureBoostingTreeHost
from federatedml.param.boosting_param import HeteroFastSecureBoostParam
from federatedml.ensemble.basic_algorithms import HeteroFastDecisionTreeHost
from federatedml.ensemble.boosting.hetero import hetero_fast_secureboost_plan as plan
from federatedml.ensemble import HeteroSecureBoostingTreeGuest
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import FeatureImportanceInfo
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.feature_importance import FeatureImportance
from federatedml.util import LOGGER
from federatedml.util import consts

make_readable_feature_importance = HeteroSecureBoostingTreeGuest.make_readable_feature_importance


class HeteroFastSecureBoostingTreeHost(HeteroSecureBoostingTreeHost):

    def __init__(self):
        super(HeteroFastSecureBoostingTreeHost, self).__init__()

        self.tree_num_per_party = 1
        self.guest_depth = 0
        self.host_depth = 0
        self.work_mode = consts.MIX_TREE
        self.init_tree_plan = False
        self.tree_plan = []
        self.model_param = HeteroFastSecureBoostParam()
        self.model_name = 'HeteroFastSecureBoost'

        self.feature_importances_ = {}

    def _init_model(self, param: HeteroFastSecureBoostParam):
        super(HeteroFastSecureBoostingTreeHost, self)._init_model(param)
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

    def update_feature_importance(self, tree_feature_importance):
        for fid in tree_feature_importance:
            if fid not in self.feature_importances_:
                self.feature_importances_[fid] = tree_feature_importance[fid]
            else:
                self.feature_importances_[fid] += tree_feature_importance[fid]

    def load_feature_importance(self, feat_importance_param):
        param = list(feat_importance_param)
        rs_dict = {}
        for fp in param:
            key = fp.fid
            importance = FeatureImportance()
            importance.from_protobuf(fp)
            rs_dict[key] = importance

        self.feature_importances_ = rs_dict

    def check_host_number(self, tree_type):
        host_num = len(self.component_properties.host_party_idlist)
        LOGGER.info('host number is {}'.format(host_num))
        if tree_type == plan.tree_type_dict['layered_tree']:
            assert host_num == 1, 'only 1 host party is allowed in layered mode'

    def fit_a_booster(self, epoch_idx: int, booster_dim: int):

        tree_type, target_host_id = self.get_tree_plan(epoch_idx)
        self.check_host_number(tree_type)
        tree = HeteroFastDecisionTreeHost(tree_param=self.tree_param)
        tree.init(flowid=self.generate_flowid(epoch_idx, booster_dim),
                  valid_features=self.sample_valid_features(),
                  data_bin=self.data_bin, bin_split_points=self.bin_split_points,
                  bin_sparse_points=self.bin_sparse_points,
                  runtime_idx=self.component_properties.local_partyid,
                  goss_subsample=self.enable_goss,
                  bin_num=self.bin_num,
                  complete_secure=True if (self.complete_secure and epoch_idx == 0) else False,
                  cipher_compressing=self.cipher_compressing,
                  new_ver=self.new_ver
                  )

        tree.set_tree_work_mode(tree_type, target_host_id)
        tree.set_layered_depth(self.guest_depth, self.host_depth)
        tree.set_self_host_id(self.component_properties.local_partyid)
        tree.set_host_party_idlist(self.component_properties.host_party_idlist)
        LOGGER.debug('tree work mode is {}'.format(tree_type))
        tree.fit()
        self.update_feature_importance(tree.get_feature_importance())
        return tree

    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):

        tree = HeteroFastDecisionTreeHost(self.tree_param)
        tree.load_model(model_meta, model_param)
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_idx))
        tree.set_runtime_idx(self.component_properties.local_partyid)

        tree_type, target_host_id = self.get_tree_plan(epoch_idx)

        # target_host_id and self_host_id and target_host_id are related to prediction
        tree.set_tree_work_mode(tree_type, target_host_id)
        tree.set_self_host_id(self.component_properties.local_partyid)

        if self.tree_plan[epoch_idx][0] == plan.tree_type_dict['guest_feat_only']:
            tree.use_guest_feat_only_predict_mode()

        return tree

    def generate_summary(self) -> dict:
        summary = super(HeteroFastSecureBoostingTreeHost, self).generate_summary()
        summary['feature_importance'] = make_readable_feature_importance(self.feature_name_fid_mapping,
                                                                         self.feature_importances_)
        return summary

    @staticmethod
    def traverse_host_local_trees(node_pos, sample, trees: List[HeteroFastDecisionTreeHost]):

        """
        in mix mode, a sample can reach leaf directly
        """
        new_node_pos = node_pos + 0
        for i in range(len(trees)):

            tree = trees[i]
            if len(tree.tree_node) == 0:  # this tree belongs to other party because it has no tree node
                continue
            leaf_id = tree.host_local_traverse_tree(sample, tree.tree_node, use_missing=tree.use_missing,
                                                    zero_as_missing=tree.zero_as_missing)
            new_node_pos[i] = leaf_id

        return new_node_pos

    # this func will be called by super class's predict()
    def boosting_fast_predict(self, data_inst, trees: List[HeteroFastDecisionTreeHost]):

        LOGGER.info('fast sbt running predict')

        if self.work_mode == consts.MIX_TREE:

            LOGGER.info('running mix mode predict')

            tree_num = len(trees)
            node_pos = data_inst.mapValues(lambda x: np.zeros(tree_num, dtype=np.int64))
            local_traverse_func = functools.partial(self.traverse_host_local_trees, trees=trees)
            leaf_pos = node_pos.join(data_inst, local_traverse_func)
            self.predict_transfer_inst.host_predict_data.remote(leaf_pos, idx=0, role=consts.GUEST)

        else:

            LOGGER.info('running layered mode predict')

            super(HeteroFastSecureBoostingTreeHost, self).boosting_fast_predict(data_inst, trees)

    def get_model_meta(self):

        _, model_meta = super(HeteroFastSecureBoostingTreeHost, self).get_model_meta()
        meta_name = consts.HETERO_FAST_SBT_HOST_MODEL + "Meta"
        model_meta.work_mode = self.work_mode

        return meta_name, model_meta

    def get_model_param(self):

        _, model_param = super(HeteroFastSecureBoostingTreeHost, self).get_model_param()
        param_name = consts.HETERO_FAST_SBT_HOST_MODEL + "Param"
        model_param.tree_plan.extend(plan.encode_plan(self.tree_plan))
        model_param.model_name = consts.HETERO_FAST_SBT_MIX if self.work_mode == consts.MIX_TREE else \
                                 consts.HETERO_FAST_SBT_LAYERED
        # in mix mode, host can output feature importance
        feature_importances = list(self.feature_importances_.items())
        feature_importances = sorted(feature_importances, key=itemgetter(1), reverse=True)
        feature_importance_param = []
        LOGGER.debug('host feat importance is {}'.format(feature_importances))
        for fid, importance in feature_importances:
            feature_importance_param.append(FeatureImportanceInfo(sitename=self.role,
                                                                  fid=fid,
                                                                  importance=importance.importance,
                                                                  fullname=self.feature_name_fid_mapping[fid],
                                                                  main=importance.main_type
                                                                  ))
        model_param.feature_importances.extend(feature_importance_param)

        return param_name, model_param

    def set_model_meta(self, model_meta):
        super(HeteroFastSecureBoostingTreeHost, self).set_model_meta(model_meta)
        self.work_mode = model_meta.work_mode

    def set_model_param(self, model_param):
        super(HeteroFastSecureBoostingTreeHost, self).set_model_param(model_param)
        self.tree_plan = plan.decode_plan(model_param.tree_plan)
        self.load_feature_importance(model_param.feature_importances)