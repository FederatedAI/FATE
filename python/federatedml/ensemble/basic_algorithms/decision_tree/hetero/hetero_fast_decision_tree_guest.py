import functools
import copy
from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeGuest
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core import tree_plan as plan
from federatedml.util import consts
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.node import Node
from federatedml.util import LOGGER


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

    """
    Setting
    """

    def use_guest_feat_only_predict_mode(self):
        self.use_guest_feat_when_predict = True

    def set_tree_work_mode(self, tree_type, target_host_id):
        self.tree_type, self.target_host_id = tree_type, target_host_id

    def set_layered_depth(self, guest_depth, host_depth):
        self.guest_depth, self.host_depth = guest_depth, host_depth

    """
    Tree Plan
    """

    def initialize_node_plan(self):
        if self.tree_type == plan.tree_type_dict['layered_tree']:
            self.node_plan = plan.create_layered_tree_node_plan(guest_depth=self.guest_depth,
                                                                host_depth=self.host_depth,
                                                                host_list=self.host_party_idlist)
            self.max_depth = len(self.node_plan)
            LOGGER.info('max depth reset to {}, cur node plan is {}'.format(self.max_depth, self.node_plan))
        else:
            self.node_plan = plan.create_node_plan(self.tree_type, self.target_host_id, self.max_depth)

    def get_node_plan(self, idx):
        return self.node_plan[idx]

    def host_id_to_idx(self, host_id):
        if host_id == -1:
            return -1
        return self.host_party_idlist.index(host_id)

    """
    Compute split point
    """

    def compute_best_splits_with_node_plan(self, tree_action, target_host_idx, cur_to_split_nodes, node_map: dict,
                                           dep: int, batch_idx: int, mode=consts.MIX_TREE):

        LOGGER.debug('node plan at dep {} is {}'.format(dep, (tree_action, target_host_idx)))

        # In layered mode, guest hist computation does not start from root node, so need to disable hist-sub
        hist_sub = True if mode == consts.MIX_TREE else False

        if tree_action == plan.tree_actions['guest_only']:
            inst2node_idx = self.get_computing_inst2node_idx()
            node_sample_count = self.count_node_sample_num(inst2node_idx, node_map)
            LOGGER.debug('sample count is {}'.format(node_sample_count))
            acc_histograms = self.get_local_histograms(dep, self.data_with_node_assignments, self.grad_and_hess,
                                                       node_sample_count, cur_to_split_nodes, node_map, ret='tensor',
                                                       hist_sub=hist_sub)

            best_split_info_guest = self.splitter.find_split(acc_histograms, self.valid_features,
                                                             self.data_bin.partitions, self.sitename,
                                                             self.use_missing, self.zero_as_missing)

            return best_split_info_guest

        if tree_action == plan.tree_actions['host_only']:

            split_info_table = self.transfer_inst.encrypted_splitinfo_host.get(
                idx=target_host_idx, suffix=(dep, batch_idx))

            host_split_info = self.splitter.find_host_best_split_info(
                split_info_table, self.get_host_sitename(target_host_idx), self.encrypter, gh_packer=self.packer)

            split_info_list = [None for i in range(len(host_split_info))]
            for key in host_split_info:
                split_info_list[node_map[key]] = host_split_info[key]

            if mode == consts.MIX_TREE:
                for split_info in split_info_list:
                    split_info.sum_grad, split_info.sum_hess, split_info.gain = self.encrypt(split_info.sum_grad), \
                        self.encrypt(split_info.sum_hess), \
                        self.encrypt(split_info.gain)
                return_split_info = split_info_list
            else:
                return_split_info = copy.deepcopy(split_info_list)
                for split_info in return_split_info:
                    split_info.sum_grad, split_info.sum_hess, split_info.gain = None, None, None
            self.transfer_inst.federated_best_splitinfo_host.remote(return_split_info,
                                                                    suffix=(dep, batch_idx),
                                                                    idx=target_host_idx,
                                                                    role=consts.HOST)
            if mode == consts.MIX_TREE:
                return []
            elif mode == consts.LAYERED_TREE:
                cur_best_split = self.merge_splitinfo(splitinfo_guest=[],
                                                      splitinfo_host=[split_info_list],
                                                      merge_host_split_only=True,
                                                      need_decrypt=False)
                return cur_best_split

    """
    Tree update
    """

    def assign_instances_to_new_node_with_node_plan(self, dep, tree_action, mode=consts.MIX_TREE, ):

        LOGGER.info("redispatch node of depth {}".format(dep))
        dispatch_node_method = functools.partial(self.assign_an_instance,
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

        if self.sample_leaf_pos is None:
            self.sample_leaf_pos = leaf
        else:
            self.sample_leaf_pos = self.sample_leaf_pos.union(leaf)

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
                                                                     unleaf_state_nodeid1) == 2 else
                                                                 unleaf_state_nodeid2)
            self.inst2node_idx = self.inst2node_idx.union(dispatch_guest_result)
        else:
            LOGGER.debug('skip host only inst2node_idx computation')
            self.inst2node_idx = dispatch_guest_result

    """
    Layered Mode
    """

    def layered_mode_fit(self):

        LOGGER.info('running layered mode')

        self.initialize_node_plan()
        self.init_packer_and_sync_gh()
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

            self.update_instances_node_positions()

            split_info = []
            for batch_idx, i in enumerate(range(0, len(self.cur_layer_nodes), self.max_split_nodes)):
                self.cur_to_split_nodes = self.cur_layer_nodes[i: i + self.max_split_nodes]
                cur_splitinfos = self.compute_best_splits_with_node_plan(
                    tree_action,
                    host_idx,
                    node_map=self.get_node_map(self.cur_to_split_nodes),
                    cur_to_split_nodes=self.cur_to_split_nodes,
                    dep=dep,
                    batch_idx=batch_idx,
                    mode=consts.LAYERED_TREE)
                split_info.extend(cur_splitinfos)

            self.update_tree(split_info, False)
            self.assign_instances_to_new_node_with_node_plan(dep, tree_action, mode=consts.LAYERED_TREE, )

        if self.cur_layer_nodes:
            self.assign_instance_to_leaves_and_update_weights()

        self.convert_bin_to_real()
        self.round_leaf_val()
        self.sync_tree(idx=-1)
        self.sample_weights_post_process()

    """
    Mix Mode
    """

    def sync_en_g_sum_h_sum(self):
        root_sum_grad, root_sum_hess = self.get_grad_hess_sum(self.grad_and_hess)
        en_g, en_h = self.encrypt(root_sum_grad), self.encrypt(root_sum_hess)
        self.transfer_inst.encrypted_grad_and_hess.remote(idx=self.host_id_to_idx(self.target_host_id),
                                                          obj=[en_g, en_h], suffix='ghsum', role=consts.HOST)

    def mix_mode_fit(self):

        LOGGER.info('running mix mode')

        self.initialize_node_plan()

        if self.tree_type != plan.tree_type_dict['guest_feat_only']:
            self.init_packer_and_sync_gh(idx=self.host_id_to_idx(self.target_host_id))
            self.sync_en_g_sum_h_sum()
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

            if len(self.cur_layer_nodes) == 0:
                break

            if self.tree_type == plan.tree_type_dict['guest_feat_only']:
                self.update_instances_node_positions()

            split_info = []
            for batch_idx, i in enumerate(range(0, len(self.cur_layer_nodes), self.max_split_nodes)):
                self.cur_to_split_nodes = self.cur_layer_nodes[i: i + self.max_split_nodes]
                cur_splitinfos = self.compute_best_splits_with_node_plan(tree_action, host_idx,
                                                                         node_map=self.get_node_map(
                                                                             self.cur_to_split_nodes),
                                                                         cur_to_split_nodes=self.cur_to_split_nodes,
                                                                         dep=dep, batch_idx=batch_idx,
                                                                         mode=consts.MIX_TREE)
                split_info.extend(cur_splitinfos)

            if self.tree_type == plan.tree_type_dict['guest_feat_only']:
                self.update_tree(split_info, False)
                self.assign_instances_to_new_node_with_node_plan(dep, tree_action, host_idx)

        if self.tree_type == plan.tree_type_dict['host_feat_only']:
            target_idx = self.host_id_to_idx(self.get_node_plan(0)[1])  # get host id
            leaves = self.sync_host_leaf_nodes(target_idx)  # get leaves node from host
            self.tree_node = self.handle_leaf_nodes(leaves)  # decrypt node info
            self.sample_leaf_pos = self.sync_sample_leaf_pos(idx=target_idx)  # get final sample leaf id from host

            # checking sample number
            assert self.sample_leaf_pos.count() == self.data_bin.count(), 'numbers of sample positions failed to match, ' \
                                                                          'sample leaf pos number:{}, instance number {}'. \
                format(self.sample_leaf_pos.count(), self.data_bin.count())
        else:
            if self.cur_layer_nodes:
                self.assign_instance_to_leaves_and_update_weights()  # guest local updates
            self.convert_bin_to_real()  # convert bin id to real value features

        self.sample_weights_post_process()
        self.round_leaf_val()

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

    """
    Federation Functions
    """

    def sync_sample_leaf_pos(self, idx):
        leaf_pos = self.transfer_inst.dispatch_node_host_result.get(idx=idx, suffix=('final sample pos',))
        return leaf_pos

    def sync_host_cur_layer_nodes(self, dep, host_idx):
        nodes = self.transfer_inst.host_cur_to_split_node_num.get(idx=host_idx, suffix=(dep,))
        for n in nodes:
            n.sum_grad = self.decrypt(n.sum_grad)
            n.sum_hess = self.decrypt(n.sum_hess)
        return nodes

    def sync_host_leaf_nodes(self, idx):
        return self.transfer_inst.host_leafs.get(idx=idx)

    """
    Mix Functions
    """

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

    def handle_leaf_nodes(self, nodes):
        """
        decrypte hess and grad and return tree node list that only contains leaves
        """
        max_node_id = -1
        for n in nodes:
            n.sum_hess = self.decrypt(n.sum_hess)
            n.sum_grad = self.decrypt(n.sum_grad)
            n.weight = self.splitter.node_weight(n.sum_grad, n.sum_hess)
            n.sitename = self.sitename
            if n.id > max_node_id:
                max_node_id = n.id
        new_nodes = [Node() for i in range(max_node_id + 1)]
        for n in nodes:
            new_nodes[n.id] = n
        return new_nodes

    """
    Fit & Predict
    """

    def fit(self):

        LOGGER.info('fitting a hetero decision tree')

        if self.tree_type == plan.tree_type_dict['host_feat_only'] or \
                self.tree_type == plan.tree_type_dict['guest_feat_only']:

            self.mix_mode_fit()

        elif self.tree_type == plan.tree_type_dict['layered_tree']:

            self.layered_mode_fit()

        LOGGER.info("end to fit guest decision tree")

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
