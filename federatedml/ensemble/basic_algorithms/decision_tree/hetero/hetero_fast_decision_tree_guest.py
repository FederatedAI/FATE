import functools

from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeGuest
import federatedml.ensemble.boosting.hetero.hetero_fast_secureboost_plan as plan
from federatedml.util import consts

from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.node import Node

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class HeteroFastDecisionTreeGuest(HeteroDecisionTreeGuest):

    def __init__(self, tree_param):
        super(HeteroFastDecisionTreeGuest, self).__init__(tree_param)
        self.node_plan = []
        self.node_plan_idx = 0
        self.tree_type = None
        self.target_host_id = -1
        self.guest_depth = 0
        self.host_depth = 0
        self.cur_dep = 0
        self.use_guest_feat_when_predict = False

    def use_guest_feat_only_predict_mode(self):
        self.use_guest_feat_when_predict = True

    def set_tree_work_mode(self, tree_type, target_host_id):
        self.tree_type, self.target_host_id = tree_type, target_host_id

    def set_layered_depth(self, guest_depth, host_depth):
        self.guest_depth, self.host_depth = guest_depth, host_depth

    def initialize_node_plan(self):
        if self.tree_type == plan.tree_type_dict['layered_tree']:
            self.node_plan = plan.create_layered_tree_node_plan(guest_depth=self.guest_depth,
                                                                host_depth=self.host_depth,
                                                                host_list=self.host_party_idlist)
            self.max_depth = len(self.node_plan)
            LOGGER.debug('max depth reset to {}, cur node plan is {}'.format(self.max_depth, self.node_plan))
        else:
            self.node_plan = plan.create_node_plan(self.tree_type, self.target_host_id, self.max_depth)

    def get_node_plan(self, idx):
        return self.node_plan[idx]

    def host_id_to_idx(self, host_id):
        if host_id == -1:
            return -1
        return self.host_party_idlist.index(host_id)

    def compute_best_splits_with_node_plan(self, tree_action, target_host_idx, node_map: dict, dep: int,
                                           batch_idx: int, mode=consts.MIX_TREE):

        LOGGER.debug('node plan at dep {} is {}'.format(dep, (tree_action, target_host_idx)))

        cur_best_split = []

        if tree_action == plan.tree_actions['guest_only']:
            acc_histograms = self.get_local_histograms(node_map, ret='tensor')
            cur_best_split = self.splitter.find_split(acc_histograms, self.valid_features,
                                                      self.data_bin._partitions, self.sitename,
                                                      self.use_missing, self.zero_as_missing)
            LOGGER.debug('computing local splits done')

        if tree_action == plan.tree_actions['host_only']:

            self.federated_find_split(dep, batch_idx, idx=target_host_idx)

            if mode == consts.LAYERED_TREE:
                host_split_info = self.sync_final_split_host(dep, batch_idx, idx=target_host_idx)
                LOGGER.debug('get encrypted split value from host')

                cur_best_split = self.merge_splitinfo(splitinfo_guest=[],
                                                      splitinfo_host=host_split_info,
                                                      merge_host_split_only=True)

        return cur_best_split

    def assign_instances_to_new_node_with_node_plan(self, dep, tree_action, mode=consts.MIX_TREE):

        LOGGER.info("redispatch node of depth {}".format(dep))
        dispatch_node_method = functools.partial(self.assign_a_instance,
                                                 tree_=self.tree_node,
                                                 decoder=self.decode,
                                                 sitename=self.sitename,
                                                 split_maskdict=self.split_maskdict,
                                                 bin_sparse_points=self.bin_sparse_points,
                                                 use_missing=self.use_missing,
                                                 zero_as_missing=self.zero_as_missing,
                                                 missing_dir_maskdict=self.missing_dir_maskdict)

        dispatch_guest_result = self.data_with_node_assignments.mapValues(dispatch_node_method)
        LOGGER.info("remask dispatch node result of depth {}".format(dep))

        dispatch_to_host_result = dispatch_guest_result.filter(
            lambda key, value: isinstance(value, tuple) and len(value) > 2)

        dispatch_guest_result = dispatch_guest_result.subtractByKey(dispatch_to_host_result)
        leaf = dispatch_guest_result.filter(lambda key, value: isinstance(value, tuple) is False)
        if self.sample_weights is None:
            self.sample_weights = leaf
        else:
            self.sample_weights = self.sample_weights.union(leaf)

        dispatch_guest_result = dispatch_guest_result.subtractByKey(leaf)

        if tree_action == plan.tree_actions['host_only'] and mode == consts.LAYERED_TREE:
            dispatch_guest_result = dispatch_guest_result.subtractByKey(leaf)
            dispatch_node_host_result = self.sync_dispatch_node_host(dispatch_to_host_result, dep, idx=-1)

            self.inst2node_idx = None
            for idx in range(len(dispatch_node_host_result)):
                if self.inst2node_idx is None:
                    self.inst2node_idx = dispatch_node_host_result[idx]
                else:
                    self.inst2node_idx = self.inst2node_idx.join(dispatch_node_host_result[idx],
                                                                 lambda unleaf_state_nodeid1,
                                                                        unleaf_state_nodeid2:
                                                                 unleaf_state_nodeid1 if len(
                                                                     unleaf_state_nodeid1) == 2 else unleaf_state_nodeid2)
            self.inst2node_idx = self.inst2node_idx.union(dispatch_guest_result)
        else:
            LOGGER.debug('skip host only inst2node_idx computation')
            self.inst2node_idx = dispatch_guest_result

    def sync_sample_leaf_pos(self, idx):
        leaf_pos = self.transfer_inst.dispatch_node_host_result.get(idx=idx, suffix=('final sample pos',))
        return leaf_pos

    @staticmethod
    def get_node_weights(node_id, tree_nodes):
        return tree_nodes[node_id].weight

    def extract_sample_weights_from_node(self, sample_leaf_pos):
        """
        Given a dtable contains leaf positions of samples, return leaf weights
        """
        func = functools.partial(self.get_node_weights, tree_nodes=self.tree_node)
        sample_weights = sample_leaf_pos.mapValues(func)
        return sample_weights

    def sync_host_cur_layer_nodes(self, dep, host_idx):
        nodes = self.transfer_inst.host_cur_to_split_node_num.get(idx=host_idx, suffix=(dep, ))
        for n in nodes:
            n.sum_grad = self.decrypt(n.sum_grad)
            n.sum_hess = self.decrypt(n.sum_hess)
        return nodes

    def sync_host_leaf_nodes(self, idx):
        return self.transfer_inst.host_leafs.get(idx=idx)

    def handle_leaf_nodes(self, nodes):
        """
        decrypte hess and grad and return tree node list that only contains leaves
        """
        max_node_id = -1
        for n in nodes:
            n.sum_hess = self.decrypt(n.sum_hess)
            n.sum_grad = self.decrypt(n.sum_grad)
            n.weight = self.splitter.node_weight(n.sum_grad, n.sum_hess)
            if n.id > max_node_id:
                max_node_id = n.id
        new_nodes = [Node() for i in range(max_node_id+1)]
        for n in nodes:
            new_nodes[n.id] = n
        return new_nodes

    def mix_mode_fit(self):

        LOGGER.info('running mix mode')

        self.initialize_node_plan()

        if self.tree_type != plan.tree_type_dict['guest_feat_only']:
            self.sync_encrypted_grad_and_hess(idx=self.host_id_to_idx(self.target_host_id))
        else:
            root_node = self.initialize_root_node()
            self.cur_layer_nodes = [root_node]
            self.inst2node_idx = self.assign_instance_to_root_node(self.data_bin, root_node_id=root_node.id)

        for dep in range(self.max_depth):

            tree_action, layer_target_host_id = self.get_node_plan(dep)
            host_idx = self.host_id_to_idx(layer_target_host_id)

            # get cur_layer_node_num
            if self.tree_type == plan.tree_type_dict['host_feat_only']:
                self.cur_layer_nodes = self.sync_host_cur_layer_nodes(dep, host_idx)
                LOGGER.debug('printing cur layer nodes')
                for n in self.cur_layer_nodes:
                    LOGGER.debug(n)

            if len(self.cur_layer_nodes) == 0:
                break

            if self.tree_type == plan.tree_type_dict['guest_feat_only']:
                self.data_with_node_assignments = self.data_bin.join(self.inst2node_idx, lambda data_inst, dispatch_info:(
                    data_inst, dispatch_info))

            split_info = []
            for batch_idx, i in enumerate(range(0, len(self.cur_layer_nodes), self.max_split_nodes)):
                self.cur_to_split_nodes = self.cur_layer_nodes[i: i + self.max_split_nodes]
                cur_splitinfos = self.compute_best_splits_with_node_plan(tree_action, host_idx, node_map=
                                                                         self.get_node_map(self.cur_to_split_nodes),
                                                                         dep=dep, batch_idx=batch_idx,
                                                                         mode=consts.MIX_TREE)
                split_info.extend(cur_splitinfos)

            reach_max_depth = True if dep + 1 == self.max_depth else False

            if self.tree_type == plan.tree_type_dict['guest_feat_only']:
                self.update_tree(split_info, reach_max_depth)
                self.assign_instances_to_new_node_with_node_plan(dep, tree_action, host_idx)

        if self.tree_type == plan.tree_type_dict['host_feat_only']:
            target_idx = self.host_id_to_idx(self.get_node_plan(0)[1])
            leaves = self.sync_host_leaf_nodes(target_idx)
            self.tree_node = self.handle_leaf_nodes(leaves)
            sample_pos = self.sync_sample_leaf_pos(idx=target_idx)
            self.sample_weights = self.extract_sample_weights_from_node(sample_pos)
        else:
            self.convert_bin_to_real()

    def layered_mode_fit(self):

        LOGGER.info('running layered mode')

        self.initialize_node_plan()

        self.sync_encrypted_grad_and_hess(idx=-1)

        root_node = self.initialize_root_node()
        self.cur_layer_nodes = [root_node]
        self.inst2node_idx = self.assign_instance_to_root_node(self.data_bin, root_node_id=root_node.id)

        for dep in range(self.max_depth):

            tree_action, layer_target_host_id = self.get_node_plan(dep)
            host_idx = self.host_id_to_idx(layer_target_host_id)

            self.sync_cur_to_split_nodes(self.cur_layer_nodes, dep, idx=-1)

            if len(self.cur_layer_nodes) == 0:
                break

            if layer_target_host_id != -1:
                self.sync_node_positions(dep, idx=-1)
            self.data_with_node_assignments = self.data_bin.join(self.inst2node_idx, lambda data_inst, dispatch_info: (
                data_inst, dispatch_info))

            split_info = []
            for batch_idx, i in enumerate(range(0, len(self.cur_layer_nodes), self.max_split_nodes)):
                self.cur_to_split_nodes = self.cur_layer_nodes[i: i + self.max_split_nodes]
                cur_splitinfos = self.compute_best_splits_with_node_plan(tree_action, host_idx, node_map=
                                                                         self.get_node_map(self.cur_to_split_nodes),
                                                                         dep=dep, batch_idx=batch_idx,
                                                                         mode=consts.LAYERED_TREE)
                split_info.extend(cur_splitinfos)

            reach_max_depth = True if dep + 1 == self.max_depth else False

            self.update_tree(split_info, reach_max_depth)

            self.assign_instances_to_new_node_with_node_plan(dep, tree_action, mode=consts.LAYERED_TREE)

        self.convert_bin_to_real()
        self.sync_tree(idx=-1)
        LOGGER.debug('final sample weights are {}'.format(list(self.sample_weights.collect())))

    def fit(self):

        LOGGER.info('fitting a hetero decision tree')

        if self.tree_type == plan.tree_type_dict['host_feat_only'] or \
           self.tree_type == plan.tree_type_dict['guest_feat_only']:

            self.mix_mode_fit()

        elif self.tree_type == plan.tree_type_dict['layered_tree']:

            self.layered_mode_fit()

        LOGGER.info("end to fit guest decision tree")

    def mix_mode_predict(self, data_inst):

        LOGGER.info("running mix mode predict")

        if self.use_guest_feat_when_predict:
            LOGGER.debug('predicting using guest local tree')
            predict_data = data_inst.mapValues(lambda inst: (0, 1))
            traverse_tree = functools.partial(self.traverse_tree,
                                              tree_=self.tree_node,
                                              decoder=self.decode,
                                              sitename=self.sitename,
                                              split_maskdict=self.split_maskdict,
                                              use_missing=self.use_missing,
                                              zero_as_missing=self.zero_as_missing,
                                              missing_dir_maskdict=self.missing_dir_maskdict)
            predict_result = predict_data.join(data_inst, traverse_tree)
            LOGGER.debug('guest_predict_inst_count is {}'.format(predict_result.count()))

        else:
            LOGGER.debug('predicting using host local tree')
            leaf_node_info = self.sync_sample_leaf_pos(idx=self.host_id_to_idx(self.target_host_id))
            predict_result = self.extract_sample_weights_from_node(leaf_node_info)

        self.transfer_inst.sync_flag.remote(True, idx=-1)
        return predict_result

    def predict(self, data_inst):
        LOGGER.info("start to predict!")
        if self.tree_type == plan.tree_type_dict['guest_feat_only'] or \
                self.tree_type == plan.tree_type_dict['host_feat_only']:
            predict_res = self.mix_mode_predict(data_inst)
            LOGGER.debug('input result count {} , out count {}'.format(data_inst.count(), predict_res.count()))
            return predict_res
        else:
            LOGGER.debug('running layered mode predict')
            return super(HeteroFastDecisionTreeGuest, self).predict(data_inst)

    def get_model_meta(self):
        return super(HeteroFastDecisionTreeGuest, self).get_model_meta()

    def get_model_param(self):
        return super(HeteroFastDecisionTreeGuest, self).get_model_param()

