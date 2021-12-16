import lightgbm as lgb
from federatedml.util import consts
from federatedml.util import LOGGER
from shap import TreeExplainer
import copy
import numpy as np
from federatedml.model_interpret.explainer.explainer_base import Explainer, data_inst_table_to_arr, \
    take_inst_in_sorted_order
from federatedml.protobuf.homo_model_convert.lightgbm.gbdt import sbt_to_lgb
from federatedml.transfer_variable.transfer_class.shap_transfer_variable import SHAPTransferVariable
from federatedml.ensemble.boosting.boosting import Boosting
from federatedml.ensemble import HeteroSecureBoostingTreeHost, HeteroDecisionTreeHost
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.model_interpret.model_adaptor import HeteroModelAdaptor


class TreeSHAP(Explainer):

    def __init__(self, role, flow_id):
        super(TreeSHAP, self).__init__(role, flow_id)
        self.tree_model_meta = None
        self.tree_model_param = None
        self.class_num = 1
        self.flow_id = flow_id
        self.mo_tree = False

    @staticmethod
    def convert_sbt_to_lgb(param, meta):
        model_str = sbt_to_lgb(param, meta)
        lgb_model = lgb.Booster(model_str=model_str)
        return lgb_model

    def handle_multi_rs(self, shap_rs):

        shap_rs_by_class = []
        dim = shap_rs.shape[1] // self.class_num
        start_idx, end_idx = 0, dim
        for i in range(self.class_num):
            shap_rs_by_class.append(shap_rs[::, start_idx: end_idx])
            start_idx += dim
            end_idx += dim

        return shap_rs_by_class

"""
Homo TreeSHAP
"""


class HomoTreeSHAP(TreeSHAP):

    def __init__(self, role, flow_id):
        super(HomoTreeSHAP, self).__init__(role, flow_id)

    def init_model(self, tree_meta, tree_param):
        self.tree_model_meta = tree_meta
        self.tree_model_param = tree_param
        self.class_num = tree_param.num_classes

    def explain(self, data_inst, n=500):

        ids, header, arr = data_inst_table_to_arr(data_inst, n)
        lgb_model = self.convert_sbt_to_lgb(self.tree_model_param, self.tree_model_meta)
        contrib = lgb_model.predict(arr, pred_contrib=True)
        if self.class_num > 2:
            contrib = self.handle_multi_rs(contrib)
        return contrib

    def explain_interaction(self, data_inst, n=500):

        ids, header, arr = data_inst_table_to_arr(data_inst=data_inst, take_num=n)
        lgb_model = self.convert_sbt_to_lgb(self.tree_model_param, self.tree_model_meta)
        shap_tree_explainer = TreeExplainer(lgb_model)
        interaction_rs = shap_tree_explainer.shap_interaction_values(arr)
        return interaction_rs


"""
Hetero TreeSHAP
"""


class HeteroTreeSHAP(TreeSHAP):

    def __init__(self, role, flow_id):
        super(HeteroTreeSHAP, self).__init__(role, flow_id)
        self.transfer_variable = SHAPTransferVariable()
        self.transfer_variable.set_flowid(flow_id)
        self.class_num = 1
        self.tree_work_mode = consts.STD_TREE
        self.model_dict = None
        self.left_dir_split_val, self.right_dir_split_val = 2, 0
        self.mock_fed_feature = 1  # 1<2 go left, 0<1 go right
        self.component_properties = None
        self.full_explain = False

        # host feature mapping
        self.anonymous_mapping = None
        self.host_node_anonymous = None

        # host sbt model
        self.host_boosting_model = None
        self.decision_tree_list = []

    def init_model(self, model_dict, component_properties):

        meta, param = None, None
        key = 'model'

        for model_key in model_dict[key]:
            model_content = model_dict[key][model_key]

            for content_name in model_content:

                if 'Meta' in content_name:
                    meta = model_content[content_name]

                elif 'Param' in content_name:
                    param = model_content[content_name]

        self.model_dict = model_dict
        self.tree_model_meta = meta
        self.tree_model_param = param
        self.component_properties = component_properties

    def load_host_boosting_model(self):

        boosting_model = HeteroModelAdaptor(self.role, self.model_dict,
                                            HeteroSecureBoostingTreeHost(),
                                            self.component_properties).fate_model
        self.host_boosting_model = boosting_model

        for tidx, tree_param in enumerate(boosting_model.boosting_model_list):
            # load host decision tree
            tree = HeteroDecisionTreeHost(boosting_model.tree_param)
            tree.load_model(boosting_model.booster_meta, tree_param)
            tree.set_runtime_idx(self.component_properties.local_partyid)
            self.decision_tree_list.append(tree)

    def set_component_properties(self, component_properties):
        self.component_properties = component_properties

    def set_full_explain(self):
        self.full_explain = True

    def add_fed_feat_mapping(self, fed_feat_list):
        """
        add host fed feature name to tree param
        """
        for i in fed_feat_list:
            self.tree_model_param.feature_name_fid_mapping[i] = 'fed_feature_{}'.format(i)

    def add_anonymous_feat_mapping(self, anonymous_fid_mapping):
        """
        add host anonymous feature name to tree param
        """

        for k, v in anonymous_fid_mapping.items():
            self.tree_model_param.feature_name_fid_mapping[v] = k

        LOGGER.debug('fid mapping updated {}'.format(self.tree_model_param.feature_name_fid_mapping))

    def get_anonymous(self, node):
        return self.anonymous_mapping[node.fid]

    def generate_anonymous_map(self):
        if self.anonymous_mapping is None:
            mapping = {}
            anonymous_real_name_mapping = dict(self.tree_model_param.anonymous_name_mapping)
            revert_anonymous_real_name_mapping = {v: k for k, v in anonymous_real_name_mapping.items()}
            feature_name_fid_mapping = dict(self.tree_model_param.feature_name_fid_mapping)
            for key in feature_name_fid_mapping:
                mapping[key] = revert_anonymous_real_name_mapping[feature_name_fid_mapping[key]]
            self.anonymous_mapping = mapping

    def extract_host_route(self, to_interpret_inst):

        data_inst = Boosting.data_format_transform(to_interpret_inst)
        route = {}
        feat_anonymous = {}
        for tidx, tree in enumerate(self.decision_tree_list):
            host_node_route = {}
            tree_node_anonymous = {}

            for node in tree.tree_node:
                if node.sitename == tree.sitename:
                    direction = HeteroDecisionTreeHost.go_next_layer(node, data_inst, tree.use_missing,
                                                                     tree.zero_as_missing, None, tree.split_maskdict,
                                                                     tree.missing_dir_maskdict, tree.decode,
                                                                     return_node_id=False)
                    host_node_route[node.id] = direction
                    tree_node_anonymous[node.id] = self.get_anonymous(node)

            # tidx == tree index
            route[tidx] = host_node_route
            feat_anonymous[tidx] = tree_node_anonymous

        return route, feat_anonymous

    def get_fed_host_feat_idx(self, host_num):
        """
        get fed host feature
        """
        feat_len = len(self.tree_model_param.feature_name_fid_mapping)
        rs = [i for i in range(feat_len, feat_len+host_num)]
        return rs

    def prepare_anonymous_fid_map(self, anonymous_list):

        feat_len = len(self.tree_model_param.feature_name_fid_mapping)
        host_feat_num = 0
        flatten_anonymous_list = []
        for host_anonymous_feat in anonymous_list:
            host_feat_num += len(host_anonymous_feat)
            # sort to keep order
            flatten_anonymous_list += sorted(host_anonymous_feat, key=lambda x: int(x.split('_')[-1]))

        fid = [i for i in range(feat_len, feat_len + host_feat_num)]
        anonymous_fid_map = {k: v for k, v in zip(flatten_anonymous_list, fid)}
        return anonymous_fid_map

    def extend_host_fed_feat(self, sample, host_feat_num):
        new_sample = np.append(sample, [1] * host_feat_num)
        return new_sample

    def explain(self, data_inst, n=500):

        """
        Hetero Tree SHAP, host features are combined into one fed-feature
        """

        LOGGER.debug('role is {}'.format(self.role))

        # running host code
        if self.role == consts.HOST:

            interpret_sample = take_inst_in_sorted_order(data_inst, take_num=n, ret_arr=False)
            self.load_host_boosting_model()
            self.generate_anonymous_map()
            sample_route_map = {}
            idx = 0
            for sample_id, sample in interpret_sample:
                route, self.host_node_anonymous = self.extract_host_route(sample)
                sample_route_map[idx] = route
                idx += 1
            if self.full_explain:
                self.transfer_variable.host_anonymous_list.remote(list(self.anonymous_mapping.values()))
                self.transfer_variable.host_node_anonymous.remote(self.host_node_anonymous)

            self.transfer_variable.host_node_route.remote(sample_route_map, suffix='route-map')

        # running guest code
        elif self.role == consts.GUEST:

            ids, header, arr = take_inst_in_sorted_order(data_inst=data_inst, take_num=n)

            # for non full explain
            host_fed_feat_idx = None
            # for full explain
            anonymous_fid_map, host_node_anonymous = None, None
            host_anonymous_list = None

            if self.full_explain:
                host_anonymous_list = self.transfer_variable.host_anonymous_list.get(idx=-1)
                # gives anonymous name in every host node
                host_node_anonymous = self.transfer_variable.host_node_anonymous.get(idx=-1)

            hosts_sample_route_map = self.transfer_variable.host_node_route.get(idx=-1, suffix='route-map')

            if not self.full_explain:
                # not full explain, host are regarded as a single feature
                host_fed_feat_idx = self.get_fed_host_feat_idx(len(hosts_sample_route_map))
                self.add_fed_feat_mapping(host_fed_feat_idx)
            else:
                # full explain
                anonymous_fid_map = self.prepare_anonymous_fid_map(host_anonymous_list)
                self.add_anonymous_feat_mapping(anonymous_fid_map)

            contribs = []
            for sample_idx in range(len(ids)):
                feat = arr[sample_idx]
                routes = [host_route[sample_idx] for host_route in hosts_sample_route_map]
                contrib = self.explain_row(feat, self.tree_model_param, routes, host_fed_feat_idx,
                                           anonymous_fid_map, host_node_anonymous)
                contribs.append(contrib)

            contribs = np.array(contribs).reshape((arr.shape[0], -1))
            if self.class_num > 2:
                contribs = contribs.reshape((arr.shape[0], -1))
                contribs = self.handle_multi_rs(contribs)

            LOGGER.info('explain model done')
            return contribs

    def add_fed_feat_to_tree_node(self, tree_param: BoostingTreeModelParam, route, fed_host_idx):

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

    def add_anonymous_feat_to_tree_node(self, tree_param: BoostingTreeModelParam, route, tree_node_anonymous,
                                        anonymous_fid_map):

        """
        Mock host node, replaced by anonymous host feature
        """

        for t_idx, tree in enumerate(tree_param.trees_):
            node_route = route[t_idx]
            for nid in node_route:
                val = self.left_dir_split_val if node_route[nid] else self.right_dir_split_val
                tree.split_maskdict[nid] = val
                tree.missing_dir_maskdict[nid] = 1
                tree.tree_[nid].fid = anonymous_fid_map[tree_node_anonymous[t_idx][nid]]

    def explain_row(self, data_inst, tree_param, routes, host_fed_feat_idx=None, anonymous_fid_map=None,
                    host_node_anonymous=None):

        tree_param = copy.deepcopy(tree_param)

        if not self.full_explain:
            for route, host_idx in zip(routes, host_fed_feat_idx):
                self.add_fed_feat_to_tree_node(tree_param, route, host_idx)  # modify host nodes

            lgb_model = self.convert_sbt_to_lgb(tree_param, self.tree_model_meta)
            to_predict_sample = self.extend_host_fed_feat(data_inst, len(host_fed_feat_idx))
        else:
            for route, tree_node_anonymous in zip(routes, host_node_anonymous):
                self.add_anonymous_feat_to_tree_node(tree_param, route, tree_node_anonymous, anonymous_fid_map)
            lgb_model = self.convert_sbt_to_lgb(tree_param, self.tree_model_meta)
            to_predict_sample = self.extend_host_fed_feat(data_inst, len(anonymous_fid_map))

        contrib = lgb_model.predict([to_predict_sample], pred_contrib=True)

        return contrib


