import copy
import numpy as np
import lightgbm as lgb
from federatedml.model_base import ModelBase
from federatedml.transfer_variable.transfer_class.shap_transfer_variable import SHAPTransferVariable
from federatedml.ensemble import HeteroSecureBoostingTreeGuest, HeteroSecureBoostingTreeHost
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam, DecisionTreeModelParam
from federatedml.ensemble.boosting.boosting_core import Boosting
from federatedml.ensemble import HeteroDecisionTreeHost
from federatedml.protobuf.model_migrate.sbt_model_to_lgb import sbt_to_lgb
from federatedml.param.shap_param import TreeSHAPParam
from federatedml.util import LOGGER
from federatedml.util import consts


class TreeSHAP(ModelBase):

    def __init__(self):

        super(TreeSHAP, self).__init__()
        self.interpret_limit = 10
        self.tree_param = None
        self.tree_meta = None
        self.run_mode = consts.HOMO
        self.model_param = TreeSHAPParam()
        self.transfer_variable = SHAPTransferVariable()

        self.left_dir_split_val, self.right_dir_split_val = 2, 0
        self.mock_fed_feature = 1  # 1<2 go left, 0<1 go right

    def _init_model(self, param):
        self.interpret_limit = param.interpret_limit

    def convert_homo_sbt_to_lgb(self):
        model_str = sbt_to_lgb(self.tree_param, self.tree_meta)
        lgb_model = lgb.Booster(model_str=model_str)
        return lgb_model

    def convert_hetero_guest_sbt_to_lgb(self, param):
        model_str = sbt_to_lgb(param, self.tree_meta)
        LOGGER.debug('model str is {}'.format(model_str))
        lgb_model = lgb.Booster(model_str=model_str)
        return lgb_model

    @staticmethod
    def _get_model_type(model_dict):
        """
        fast-sbt or sbt ? hetero or homo ?
        """
        sbt_key_prefix = consts.HETERO_SBT_GUEST_MODEL.replace('Guest', '')
        homo_sbt_key_prefix = consts.HOMO_SBT_GUEST_MODEL.replace('Guest', '')
        for key in model_dict:
            for model_key in model_dict[key]:
                if sbt_key_prefix in model_key:
                    return consts.HETERO_SBT
                elif homo_sbt_key_prefix in model_key:
                    return consts.HOMO_SBT

        return None

    def make_output_rs(self, lgb_model, pred_contrib):
        pass

    def make_predict_format(self, to_interpret_inst):

        ids = []
        data_list = []
        for id_, inst in to_interpret_inst:
            ids.append(id_)
            data_list.append(inst.features)
        data_arr = np.array(data_list)
        return ids, data_arr

    """
    Homo functions
    """

    def homo_fit(self, data_inst):

        to_interpret_inst = data_inst.take(self.interpret_limit)
        lgb_model = self.convert_homo_sbt_to_lgb()
        ids, data_arr = self.make_predict_format(to_interpret_inst)
        contrib = lgb_model.predict(data_arr, pred_contrib=True)
        LOGGER.debug('to interpret inst {}'.format(data_arr))
        LOGGER.debug('contrib is {}'.format(contrib))
        LOGGER.info('explain model done')

    """
    Hetero Functions
    """

    def extract_host_route(self, to_interpret_inst):

        data_inst = Boosting.data_format_transform(to_interpret_inst)
        boosting_model = HeteroSecureBoostingTreeHost()
        boosting_model.set_model_meta(self.tree_meta)
        boosting_model.set_model_param(self.tree_param)
        boosting_model.component_properties = copy.deepcopy(self.component_properties)
        route = {}
        for tidx, tree_param in enumerate(boosting_model.boosting_model_list):
            host_node_route = {}
            # load host decision tree
            tree = HeteroDecisionTreeHost(boosting_model.tree_param)
            tree.load_model(boosting_model.booster_meta, tree_param)
            tree.set_runtime_idx(self.component_properties.local_partyid)
            for node in tree.tree_node:
                if node.sitename == tree.sitename:
                    direction = HeteroDecisionTreeHost.go_next_layer(node, data_inst, tree.use_missing,
                                                                     tree.zero_as_missing, None, tree.split_maskdict,
                                                                     tree.missing_dir_maskdict, tree.decode,
                                                                     return_node_id=False)
                    host_node_route[node.id] = direction
            route[tidx] = host_node_route

        return route

    def mock_hetero_tree_node(self, tree_param: BoostingTreeModelParam, route, fed_host_idx):

        """
        Mock host node, replaced by fed host feature
        """

        for t_idx, tree in enumerate(tree_param.trees_):
            node_route = route[t_idx]
            for nid in node_route:
                val = self.left_dir_split_val if node_route[nid] else self.right_dir_split_val
                tree.split_maskdict[nid] = val
                tree.missing_dir_maskdict[nid] = 1
                tree.tree_[nid].fid = fed_host_idx

    def add_host_feat_mapping(self, fed_feat_list):
        """
        add fed feature name to tree param
        """
        for i in fed_feat_list:
            self.tree_param.feature_name_fid_mapping[i] = 'fed_feature_{}'.format(i)

    def extend_host_fed_feat(self, sample, host_num):
        new_sample = np.append(sample, [1]*host_num)
        return new_sample

    def get_fed_host_feat_idx(self, host_num):
        """
        get fed host feature
        """
        feat_len = len(self.tree_param.feature_name_fid_mapping)
        rs = [i for i in range(feat_len, feat_len+host_num)]
        return rs

    def explain(self, data_arr, tree_param, routes, host_fed_feat_idx):

        tree_param = copy.deepcopy(tree_param)
        for route, host_idx in zip(routes, host_fed_feat_idx):
            self.mock_hetero_tree_node(tree_param, route, host_idx)  # modify host nodes

        lgb_model = self.convert_hetero_guest_sbt_to_lgb(tree_param)
        to_predict_sample = self.extend_host_fed_feat(data_arr, len(host_fed_feat_idx))
        predict_rs = lgb_model.predict([to_predict_sample])
        contrib = lgb_model.predict([to_predict_sample], pred_contrib=True)
        LOGGER.debug('predict rs {}, contrib {}'.format(predict_rs, contrib))
        return contrib

    def hetero_fit(self, data_inst):

        """
        Hetero Tree SHAP, host features are combined into one fed-feature
        """

        LOGGER.debug('role is {}'.format(self.role))
        if self.role == consts.HOST:
            interpret_sample = data_inst.take(self.interpret_limit)
            sample_route_map = {}
            for sample_id, sample in interpret_sample:
                route = self.extract_host_route(sample)
                sample_route_map[sample_id] = route
            LOGGER.debug('sample route map {}'.format(sample_route_map))
            self.transfer_variable.host_node_route.remote(sample_route_map, suffix='route-map')

        elif self.role == consts.GUEST:
            hosts_sample_route_map = self.transfer_variable.host_node_route.get(idx=-1, suffix='route-map')
            LOGGER.debug('get route map from {} host'.format(len(hosts_sample_route_map)))
            LOGGER.debug('get route map {} '.format(hosts_sample_route_map))

            host_fed_feat_idx = self.get_fed_host_feat_idx(len(hosts_sample_route_map))
            self.add_host_feat_mapping(host_fed_feat_idx)

            ids, data_arr = self.make_predict_format(data_inst.take(self.interpret_limit))
            contribs = []
            for sample_id, feat in zip(ids, data_arr):
                routes = [host_route[sample_id] for host_route in hosts_sample_route_map]
                contrib = self.explain(feat, self.tree_param, routes, host_fed_feat_idx)
                contribs.append(contrib)

            feat_name = data_inst.schema['header']
            import pickle
            pickle.dump([feat_name, contribs],
                        open('/home/cwj/FATE/standalone-fate-master-1.4.5/shap_result/shap_{}.pkl'.format(self.task_version_id), 'bw'))

            LOGGER.info('explain model done')

    def load_model(self, model_dict):

        key = 'isometric_model'
        model_type = self._get_model_type(model_dict[key])
        if model_type == consts.HOMO_SBT:
            self.run_mode = consts.HOMO
        elif model_type == consts.HETERO_SBT:
            self.run_mode = consts.HETERO
        else:
            raise ValueError('illegal input model: {}'.format(model_dict))

        for model_key in model_dict[key]:
            model_content = model_dict[key][model_key]
            for content_name in model_content:
                if 'Meta' in content_name:
                    self.tree_meta = model_content[content_name]
                elif 'Param' in content_name:
                    self.tree_param = model_content[content_name]

    def fit(self, data_inst):

        if self.run_mode == consts.HOMO:
            LOGGER.info('running homo tree shap')
            self.homo_fit(data_inst)
        elif self.run_mode == consts.HETERO:
            LOGGER.info('running hetero tree shap')
            self.hetero_fit(data_inst)
        else:
            raise ValueError('illegal model input')
