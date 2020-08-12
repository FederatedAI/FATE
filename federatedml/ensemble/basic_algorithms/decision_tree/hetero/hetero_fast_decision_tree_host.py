from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeHost

from federatedml.ensemble.boosting.hetero import hetero_fast_secureboost_plan as plan
from federatedml.util import consts

from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.splitter import SplitInfo
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.node import Node
from federatedml.feature.fate_element_type import NoneType

from arch.api.utils import log_utils
import functools
import copy

LOGGER = log_utils.getLogger()


class HeteroFastDecisionTreeHost(HeteroDecisionTreeHost):

    def __init__(self, tree_param):
        super(HeteroFastDecisionTreeHost, self).__init__(tree_param)
        self.node_plan = []
        self.node_plan_idx = 0
        self.tree_type = None
        self.target_host_id = -1
        self.guest_depth = 0
        self.host_depth = 0
        self.cur_dep = 0
        self.self_host_id = -1
        self.use_guest_feat_when_predict = False

        self.tree_node = []  # keep tree structure for faster node dispatch
        self.sample_leaf_pos = None  # record leaf position of samples

    def use_guest_feat_only_predict_mode(self):
        self.use_guest_feat_when_predict = True

    def set_tree_work_mode(self, tree_type, target_host_id):
        self.tree_type, self.target_host_id = tree_type, target_host_id

    def set_layered_depth(self, guest_depth, host_depth):
        self.guest_depth, self.host_depth = guest_depth, host_depth

    def set_self_host_id(self, self_host_id):
        self.self_host_id = self_host_id

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

    def get_host_split_info(self, splitinfo_host, federated_best_splitinfo_host):

        final_splitinfos = []
        for i in range(len(splitinfo_host)):
            best_idx, best_gain = federated_best_splitinfo_host[i]
            if best_idx != -1:
                LOGGER.debug('sitename is {}, self.sitename is {}'
                             .format(splitinfo_host[i][best_idx].sitename, self.sitename))
                assert splitinfo_host[i][best_idx].sitename == self.sitename
                splitinfo = splitinfo_host[i][best_idx]
                splitinfo.best_fid = splitinfo.best_fid
                assert splitinfo.best_fid is not None
                splitinfo.best_bid = splitinfo.best_bid
                splitinfo.missing_dir = splitinfo.missing_dir
                splitinfo.gain = best_gain
            else:
                splitinfo = SplitInfo(sitename=self.sitename, best_fid=-1, best_bid=-1, gain=best_gain)

            final_splitinfos.append(splitinfo)

        return final_splitinfos

    def update_host_side_tree(self, split_info, reach_max_depth):

        LOGGER.info("update tree node, splitlist length is {}, tree node queue size is".format(
            len(split_info), len(self.cur_layer_nodes)))

        new_tree_node_queue = []
        for i in range(len(self.cur_layer_nodes)):

            sum_grad = self.cur_layer_nodes[i].sum_grad
            sum_hess = self.cur_layer_nodes[i].sum_hess

            # when host node can not be further split, fid/bid is set to -1
            if reach_max_depth or split_info[i].best_fid == -1:
                self.cur_layer_nodes[i].is_leaf = True
            else:
                self.cur_layer_nodes[i].left_nodeid = self.tree_node_num + 1
                self.cur_layer_nodes[i].right_nodeid = self.tree_node_num + 2
                self.tree_node_num += 2

                left_node = Node(id=self.cur_layer_nodes[i].left_nodeid,
                                 sitename=self.sitename,
                                 sum_grad=split_info[i].sum_grad,
                                 sum_hess=split_info[i].sum_hess)
                right_node = Node(id=self.cur_layer_nodes[i].right_nodeid,
                                  sitename=self.sitename,
                                  sum_grad=sum_grad - split_info[i].sum_grad,
                                  sum_hess=sum_hess - split_info[i].sum_hess,)

                new_tree_node_queue.append(left_node)
                new_tree_node_queue.append(right_node)

                self.cur_layer_nodes[i].sitename = split_info[i].sitename

                self.cur_layer_nodes[i].fid = split_info[i].best_fid
                self.cur_layer_nodes[i].bid = split_info[i].best_bid
                self.cur_layer_nodes[i].missing_dir = split_info[i].missing_dir

                if self.feature_importance_type == 'split':
                    self.update_feature_importance(split_info[i], )

            self.tree_node.append(self.cur_layer_nodes[i])

        self.cur_layer_nodes = new_tree_node_queue

    def compute_best_splits_with_node_plan(self, tree_action, target_host_id, node_map: dict, dep: int, batch_idx: int,
                                           mode=consts.LAYERED_TREE):

        LOGGER.debug('node plan at dep {} is {}'.format(dep, (tree_action, target_host_id)))

        if tree_action == plan.tree_actions['host_only'] and target_host_id == self.self_host_id:
            acc_histograms = self.get_local_histograms(node_map, ret='tb')
            splitinfo_host, encrypted_splitinfo_host = self.splitter.find_split_host(histograms=acc_histograms,
                                                                                     node_map=node_map,
                                                                                     use_missing=self.use_missing,
                                                                                     zero_as_missing=self.zero_as_missing,
                                                                                     valid_features=self.valid_features,
                                                                                     sitename=self.sitename
                                                                                     )
            LOGGER.debug('sending en_splitinfo {}'.format(encrypted_splitinfo_host))
            self.sync_encrypted_splitinfo_host(encrypted_splitinfo_host, dep, batch_idx)
            federated_best_splitinfo_host = self.sync_federated_best_splitinfo_host(dep, batch_idx)

            host_split_info = self.get_host_split_info(copy.deepcopy(splitinfo_host),
                                                       copy.deepcopy(federated_best_splitinfo_host))

            if mode == consts.LAYERED_TREE:
                LOGGER.debug('sending split info to guest')
                self.sync_final_splitinfo_host(splitinfo_host, federated_best_splitinfo_host, dep, batch_idx)
                LOGGER.debug('computing host splits done')

            return host_split_info
        else:
            LOGGER.debug('skip best split computation')
            return None

    @staticmethod
    def host_assign_a_instance(value, tree_, bin_sparse_points, use_missing, zero_as_missing):

        unleaf_state, nodeid = value[1]

        if tree_[nodeid].is_leaf is True:
            return nodeid

        fid = tree_[nodeid].fid
        bid = tree_[nodeid].bid

        if not use_missing:
            if value[0].features.get_data(fid, bin_sparse_points[fid]) <= bid:
                return 1, tree_[nodeid].left_nodeid
            else:
                return 1, tree_[nodeid].right_nodeid
        else:
            missing_dir = tree_[nodeid].missing_dir

            missing_val = False
            if zero_as_missing:
                if value[0].features.get_data(fid, None) is None or \
                        value[0].features.get_data(fid) == NoneType():
                    missing_val = True
            elif use_missing and value[0].features.get_data(fid) == NoneType():
                missing_val = True

            if missing_val:
                if missing_dir == 1:
                    return 1, tree_[nodeid].right_nodeid
                else:
                    return 1, tree_[nodeid].left_nodeid
            else:
                if value[0].features.get_data(fid, bin_sparse_points[fid]) <= bid:
                    return 1, tree_[nodeid].left_nodeid
                else:
                    return 1, tree_[nodeid].right_nodeid

    def host_local_assign_instances_to_new_node(self):

        assign_node_method = functools.partial(self.host_assign_a_instance,
                                               tree_=self.tree_node,
                                               bin_sparse_points=self.bin_sparse_points,
                                               use_missing=self.use_missing,
                                               zero_as_missing=self.zero_as_missing,)

        assign_result = self.data_with_node_assignments.mapValues(assign_node_method)
        leaf = assign_result.filter(lambda key, value: isinstance(value, tuple) is False)

        if self.sample_leaf_pos is None:
            self.sample_leaf_pos = leaf
        else:
            self.sample_leaf_pos = self.sample_leaf_pos.union(leaf)

        assign_result = assign_result.subtractByKey(leaf)

        return assign_result

    def sync_sample_leaf_pos(self, sample_leaf_pos):
        LOGGER.debug('final sample pos sent')
        self.transfer_inst.dispatch_node_host_result.remote(sample_leaf_pos, idx=0,
                                                            suffix=('final sample pos', ), role=consts.GUEST)

    def remove_encrypted_info(self):
        for node in self.tree_node:
            node.sum_grad = None
            node.sum_hess = None

    def sync_leaf_nodes(self):
        leaves = []
        for node in self.tree_node:
            if node.is_leaf:
                leaves.append(node)
        to_send_leaves = copy.deepcopy(leaves)
        self.transfer_inst.host_leafs.remote(to_send_leaves)

    def mask_node_id(self, nodes):
        for n in nodes:
            n.id = -1
        return nodes

    def sync_cur_layer_nodes(self, nodes, dep):
        # self.mask_node_id(copy.deepcopy(nodes))
        self.transfer_inst.host_cur_to_split_node_num.\
            remote(nodes, idx=0, role=consts.GUEST, suffix=(dep, ))

    def convert_bin_to_real(self):
        LOGGER.info("convert tree node bins to real value")
        for i in range(len(self.tree_node)):
            if self.tree_node[i].is_leaf is True:
                continue
            if self.tree_node[i].sitename == self.sitename:
                fid = self.decode("feature_idx", self.tree_node[i].fid, split_maskdict=self.split_maskdict)
                bid = self.decode("feature_val", self.tree_node[i].bid, self.tree_node[i].id, self.split_maskdict)
                real_splitval = self.encode("feature_val", self.bin_split_points[fid][bid], self.tree_node[i].id)
                self.tree_node[i].bid = real_splitval

    def convert_bin_to_real2(self):
        """
        convert current bid in tree nodes to real value
        """
        for node in self.tree_node:
            if not node.is_leaf:
                node.bid = self.bin_split_points[node.fid][node.bid]

    def mix_mode_fit(self):

        LOGGER.info('running mix mode')

        if self.tree_type == plan.tree_type_dict['guest_feat_only']:
            LOGGER.debug('this tree uses guest feature only, skip')
            return
        if self.self_host_id != self.target_host_id:
            LOGGER.debug('not selected host, skip')
            return

        LOGGER.debug('use local host feature to build tree')

        self.sync_encrypted_grad_and_hess()
        root_sum_grad, root_sum_hess = self.get_grad_hess_sum(self.grad_and_hess)
        self.inst2node_idx = self.assign_instance_to_root_node(self.data_bin,
                                                               root_node_id=0)  # root node id is 0

        self.cur_layer_nodes = [Node(id=0, sitename=self.sitename, sum_grad=root_sum_grad, sum_hess=root_sum_hess,)]

        for dep in range(self.max_depth):

            tree_action, layer_target_host_id = self.get_node_plan(dep)

            self.sync_cur_layer_nodes(self.cur_layer_nodes, dep)
            if len(self.cur_layer_nodes) == 0:
                break

            self.data_with_node_assignments = self.data_bin.join(self.inst2node_idx, lambda v1, v2: (v1, v2))

            batch = 0
            split_info = []
            for i in range(0, len(self.cur_layer_nodes), self.max_split_nodes):
                self.cur_to_split_nodes = self.cur_layer_nodes[i: i + self.max_split_nodes]
                batch_split_info = self.compute_best_splits_with_node_plan(tree_action, layer_target_host_id,
                                                                           node_map=self.get_node_map(
                                                                            self.cur_to_split_nodes),
                                                                           dep=dep, batch_idx=batch,
                                                                           mode=consts.MIX_TREE)
                batch += 1
                split_info.extend(batch_split_info)

            reach_max_depth = True if dep + 1 == self.max_depth else False
            self.update_host_side_tree(split_info, reach_max_depth=reach_max_depth)
            self.inst2node_idx = self.host_local_assign_instances_to_new_node()

        self.sync_sample_leaf_pos(self.sample_leaf_pos)
        self.convert_bin_to_real2()
        self.sync_leaf_nodes()
        self.remove_encrypted_info()

    def layered_mode_fit(self):

        LOGGER.info('running layered mode')

        self.initialize_node_plan()

        self.sync_encrypted_grad_and_hess()

        for dep in range(self.max_depth):

            tree_action, layer_target_host_id = self.get_node_plan(dep)

            self.sync_tree_node_queue(dep)
            if len(self.cur_layer_nodes) == 0:
                break

            if self.self_host_id == layer_target_host_id:
                self.inst2node_idx = self.sync_node_positions(dep)
                self.data_with_node_assignments = self.data_bin.join(self.inst2node_idx, lambda v1, v2: (v1, v2))

            batch = 0
            for i in range(0, len(self.cur_layer_nodes), self.max_split_nodes):
                self.cur_to_split_nodes = self.cur_layer_nodes[i: i + self.max_split_nodes]
                self.compute_best_splits_with_node_plan(tree_action, layer_target_host_id,
                                                        node_map=self.get_node_map(self.cur_to_split_nodes),
                                                        dep=dep, batch_idx=batch, )
                batch += 1

            if layer_target_host_id == self.self_host_id:
                dispatch_node_host = self.sync_dispatch_node_host(dep)
                self.assign_instances_to_new_node(dispatch_node_host, dep)

        self.sync_tree()
        self.convert_bin_to_real()

    def fit(self):

        LOGGER.info("begin to fit fast host decision tree")

        self.initialize_node_plan()

        if self.tree_type == plan.tree_type_dict['guest_feat_only'] or \
                self.tree_type == plan.tree_type_dict['host_feat_only']:

            self.mix_mode_fit()

        else:
            self.layered_mode_fit()

        LOGGER.info("end to fit host decision tree")

    @staticmethod
    def host_local_traverse_tree(data_inst, tree_node, use_missing=True, zero_as_missing=True):

        nid = 0  # root node id
        while True:

            if tree_node[nid].is_leaf:
                return nid

            cur_node = tree_node[nid]
            fid, bid = cur_node.fid, cur_node.bid
            missing_dir = cur_node.missing_dir

            if use_missing and zero_as_missing:

                if data_inst.features.get_data(fid) == NoneType() or data_inst.features.get_data(fid, None) is None:

                    nid = tree_node[nid].right_nodeid if missing_dir == 1 else tree_node[nid].left_nodeid

                elif data_inst.features.get_data(fid) <= bid:
                    nid = tree_node[nid].left_nodeid
                else:
                    nid = tree_node[nid].right_nodeid

            elif data_inst.features.get_data(fid) == NoneType():

                nid = tree_node[nid].right_nodeid if missing_dir == 1 else tree_node[nid].left_nodeid

            elif data_inst.features.get_data(fid, 0) <= bid:
                nid = tree_node[nid].left_nodeid
            else:
                nid = tree_node[nid].right_nodeid

    def mix_mode_predict(self, data_inst):

        LOGGER.debug('running mix mode predict')

        if not self.use_guest_feat_when_predict and self.target_host_id == self.self_host_id:
            LOGGER.info('predicting using local nodes')
            traverse_tree = functools.partial(self.host_local_traverse_tree,
                                              tree_node=self.tree_node,
                                              use_missing=self.use_missing,
                                              zero_as_missing=self.zero_as_missing, )
            leaf_nodes = data_inst.mapValues(traverse_tree)
            LOGGER.debug('leaf nodes count is {}'.format(leaf_nodes.count()))
            self.sync_sample_leaf_pos(leaf_nodes)
        else:
            LOGGER.info('this tree belongs to other parties, skip prediction')

        # sync status
        _ = self.transfer_inst.sync_flag.get(idx=0)

    def predict(self, data_inst):

        LOGGER.info("start to predict!")

        if self.tree_type == plan.tree_type_dict['guest_feat_only'] or \
                self.tree_type == plan.tree_type_dict['host_feat_only']:

            self.mix_mode_predict(data_inst)

        else:
            LOGGER.debug('running layered mode predict')
            super(HeteroFastDecisionTreeHost, self).predict(data_inst)

        LOGGER.info('predict done')

    def get_model_meta(self):
        return super(HeteroFastDecisionTreeHost, self).get_model_meta()

    def get_model_param(self):
        return super(HeteroFastDecisionTreeHost, self).get_model_param()