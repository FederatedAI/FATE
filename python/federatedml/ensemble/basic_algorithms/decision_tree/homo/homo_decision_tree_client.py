import functools
from federatedml.util import LOGGER
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import CriterionMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import DecisionTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import DecisionTreeModelParam
from federatedml.transfer_variable.transfer_class.homo_decision_tree_transfer_variable import \
    HomoDecisionTreeTransferVariable
from federatedml.util import consts
from federatedml.ensemble import FeatureHistogram
from federatedml.ensemble import DecisionTree
from federatedml.ensemble import Splitter
from federatedml.ensemble import Node
from federatedml.ensemble import HistogramBag
from federatedml.ensemble import SplitInfo
from federatedml.ensemble import DecisionTreeClientAggregator
from federatedml.feature.fate_element_type import NoneType
from federatedml.feature.instance import Instance
from federatedml.param import DecisionTreeParam
import numpy as np
from typing import List


class HomoDecisionTreeClient(DecisionTree):

    def __init__(self, tree_param: DecisionTreeParam, data_bin=None, bin_split_points: np.array = None,
                 bin_sparse_point=None, g_h=None, valid_feature: dict = None, epoch_idx: int = None,
                 role: str = None, tree_idx: int = None, flow_id: int = None, mode='train'):

        """
        Parameters
        ----------
        tree_param: decision tree parameter object
        data_bin binned: data instance
        bin_split_points: data split points
        bin_sparse_point: sparse data point
        g_h computed: g val and h val of instances
        valid_feature: dict points out valid features {valid:true,invalid:false}
        epoch_idx: current epoch index
        role: host or guest
        flow_id: flow id
        mode: train / predict
        """

        super(HomoDecisionTreeClient, self).__init__(tree_param)
        self.splitter = Splitter(self.criterion_method, self.criterion_params, self.min_impurity_split,
                                 self.min_sample_split, self.min_leaf_node)
        self.data_bin = data_bin
        self.g_h = g_h
        self.bin_split_points = bin_split_points
        self.bin_sparse_points = bin_sparse_point
        self.epoch_idx = epoch_idx
        self.tree_idx = tree_idx
        self.transfer_inst = HomoDecisionTreeTransferVariable()

        """
        initializing here
        """
        self.valid_features = valid_feature
        self.tree_node = []  # start from root node
        self.tree_node_num = 0
        self.cur_layer_node = []
        self.runtime_idx = 0
        self.sitename = consts.GUEST
        self.feature_importance = {}

        # secure aggregator, class SecureBoostClientAggregator
        if mode == 'train':
            self.role = role
            self.set_flowid(flow_id)
            self.aggregator = DecisionTreeClientAggregator(verbose=False)

        elif mode == 'predict':
            self.role, self.aggregator = None, None

        self.check_max_split_nodes()

    def set_flowid(self, flowid=0):
        LOGGER.info("set flowid, flowid is {}".format(flowid))
        self.transfer_inst.set_flowid(flowid)

    """
    Federation functions
    """

    def sync_local_node_histogram(self, acc_histogram: List[HistogramBag], suffix):
        # sending local histogram
        self.aggregator.send_histogram(acc_histogram, suffix=suffix)
        LOGGER.debug('local histogram sent at layer {}'.format(suffix[0]))

    def sync_cur_layer_node_num(self, node_num, suffix):
        self.transfer_inst.cur_layer_node_num.remote(node_num, role=consts.ARBITER, idx=-1, suffix=suffix)

    def sync_best_splits(self, suffix) -> List[SplitInfo]:

        best_splits = self.transfer_inst.best_split_points.get(idx=0, suffix=suffix)
        return best_splits

    """
    Computing functions
    """

    def get_node_map(self, nodes: List[Node], left_node_only=True):
        node_map = {}
        idx = 0
        for node in nodes:
            if node.id != 0 and (not node.is_left_node and left_node_only):
                continue
            node_map[node.id] = idx
            idx += 1
        return node_map

    def get_grad_hess_sum(self, grad_and_hess_table):
        LOGGER.info("calculate the sum of grad and hess")
        grad, hess = grad_and_hess_table.reduce(
            lambda value1, value2: (value1[0] + value2[0], value1[1] + value2[1]))
        return grad, hess

    def get_local_histogram(self, cur_to_split: List[Node], g_h, table_with_assign,
                            split_points, sparse_point, valid_feature):

        LOGGER.info("start to get node histograms")
        node_map = self.get_node_map(nodes=cur_to_split)
        histograms = FeatureHistogram.calculate_histogram(
                    table_with_assign, g_h,
                    split_points, sparse_point,
                    valid_feature, node_map,
                    self.use_missing, self.zero_as_missing)

        hist_bags = []
        for hist_list in histograms:
            hist_bags.append(HistogramBag(hist_list))

        return hist_bags

    def get_left_node_local_histogram(self, cur_nodes: List[Node], tree: List[Node], g_h, table_with_assign,
                                      split_points, sparse_point, valid_feature):

        node_map = self.get_node_map(cur_nodes, left_node_only=True)

        LOGGER.info("start to get node histograms")
        histograms = self.hist_computer.calculate_histogram(
            table_with_assign, g_h,
            split_points, sparse_point,
            valid_feature, node_map,
            self.use_missing, self.zero_as_missing)

        hist_bags = []
        for hist_list in histograms:
            hist_bags.append(HistogramBag(hist_list))

        left_nodes = []
        for node in cur_nodes:
            if node.is_left_node or node.id == 0:
                left_nodes.append(node)

        # set histogram id and parent histogram id
        for node, hist_bag in zip(left_nodes, hist_bags):
            # LOGGER.debug('node id {}, node parent id {}, cur tree {}'.format(node.id, node.parent_nodeid, len(tree)))
            hist_bag.hid = node.id
            hist_bag.p_hid = node.parent_nodeid

        return hist_bags

    """
    Tree Updating
    """

    def update_tree(self, cur_to_split: List[Node], split_info: List[SplitInfo]):
        """
        update current tree structure
        ----------
        split_info
        """
        LOGGER.debug('updating tree_node, cur layer has {} node'.format(len(cur_to_split)))
        next_layer_node = []
        assert len(cur_to_split) == len(split_info)

        for idx in range(len(cur_to_split)):
            sum_grad = cur_to_split[idx].sum_grad
            sum_hess = cur_to_split[idx].sum_hess
            if split_info[idx].best_fid is None or split_info[idx].gain <= self.min_impurity_split + consts.FLOAT_ZERO:
                cur_to_split[idx].is_leaf = True
                self.tree_node.append(cur_to_split[idx])
                continue

            cur_to_split[idx].fid = split_info[idx].best_fid
            cur_to_split[idx].bid = split_info[idx].best_bid
            cur_to_split[idx].missing_dir = split_info[idx].missing_dir

            p_id = cur_to_split[idx].id
            l_id, r_id = self.tree_node_num + 1, self.tree_node_num + 2
            cur_to_split[idx].left_nodeid, cur_to_split[idx].right_nodeid = l_id, r_id
            self.tree_node_num += 2

            l_g, l_h = split_info[idx].sum_grad, split_info[idx].sum_hess

            # create new left node and new right node
            left_node = Node(id=l_id,
                             sitename=self.sitename,
                             sum_grad=l_g,
                             sum_hess=l_h,
                             weight=self.splitter.node_weight(l_g, l_h),
                             parent_nodeid=p_id,
                             sibling_nodeid=r_id,
                             is_left_node=True)
            right_node = Node(id=r_id,
                              sitename=self.sitename,
                              sum_grad=sum_grad - l_g,
                              sum_hess=sum_hess - l_h,
                              weight=self.splitter.node_weight(sum_grad - l_g, sum_hess - l_h),
                              parent_nodeid=p_id,
                              sibling_nodeid=l_id,
                              is_left_node=False)

            next_layer_node.append(left_node)
            next_layer_node.append(right_node)
            self.tree_node.append(cur_to_split[idx])

            self.update_feature_importance(split_info[idx], record_site_name=False)

        return next_layer_node

    @staticmethod
    def assign_an_instance(row, tree: List[Node], bin_sparse_point, use_missing, use_zero_as_missing):

        leaf_status, nodeid = row[1]
        node = tree[nodeid]
        if node.is_leaf:
            return node.weight

        fid = node.fid
        bid = node.bid

        missing_dir = node.missing_dir

        missing_val = False
        if use_zero_as_missing:
            if row[0].features.get_data(fid, None) is None or \
                    row[0].features.get_data(fid) == NoneType():
                missing_val = True
        elif use_missing and row[0].features.get_data(fid) == NoneType():
            missing_val = True

        if missing_val:
            if missing_dir == 1:
                return 1, tree[nodeid].right_nodeid
            else:
                return 1, tree[nodeid].left_nodeid
        else:
            if row[0].features.get_data(fid, bin_sparse_point[fid]) <= bid:
                return 1, tree[nodeid].left_nodeid
            else:
                return 1, tree[nodeid].right_nodeid

    def assign_instances_to_new_node(self, table_with_assignment, tree_node: List[Node]):

        LOGGER.debug('re-assign instance to new nodes')
        assign_method = functools.partial(self.assign_an_instance, tree=tree_node, bin_sparse_point=
                                          self.bin_sparse_points, use_missing=self.use_missing, use_zero_as_missing
                                          =self.zero_as_missing)

        assign_result = table_with_assignment.mapValues(assign_method)
        leaf_val = assign_result.filter(lambda key, value: isinstance(value, tuple) is False)

        assign_result = assign_result.subtractByKey(leaf_val)

        return assign_result, leaf_val

    def update_instances_node_positions(self, ):
        return self.data_bin.join(self.inst2node_idx, lambda inst, assignment: (inst, assignment))

    """
    Pre/Post process
    """

    @staticmethod
    def get_node_sample_weights(inst2node, tree_node: List[Node]):
        """
        get samples' weights which correspond to its node assignment
        """
        func = functools.partial(lambda inst, nodes: nodes[inst[1]].weight, nodes=tree_node)
        return inst2node.mapValues(func)

    def get_feature_importance(self):
        return self.feature_importance

    def convert_bin_to_real(self):
        """
        convert current bid in tree nodes to real value
        """
        for node in self.tree_node:
            if not node.is_leaf:
                node.bid = self.bin_split_points[node.fid][node.bid]

    def assign_instance_to_root_node(self, data_bin, root_node_id):
        return data_bin.mapValues(lambda inst: (1, root_node_id))

    """
    Fit & Predict
    """

    def fit(self):
        """
        start to fit
        """
        LOGGER.info('begin to fit homo decision tree, epoch {}, tree idx {}'.format(self.epoch_idx, self.tree_idx))

        # compute local g_sum and h_sum
        g_sum, h_sum = self.get_grad_hess_sum(self.g_h)

        # get aggregated root info
        self.aggregator.send_local_root_node_info(g_sum, h_sum, suffix=('root_node_sync1', self.epoch_idx))
        g_h_dict = self.aggregator.get_aggregated_root_info(suffix=('root_node_sync2', self.epoch_idx))
        global_g_sum, global_h_sum = g_h_dict['g_sum'], g_h_dict['h_sum']

        # initialize node
        root_node = Node(id=0, sitename=consts.GUEST, sum_grad=global_g_sum, sum_hess=global_h_sum, weight=
                         self.splitter.node_weight(global_g_sum, global_h_sum))

        self.cur_layer_node = [root_node]
        LOGGER.debug('assign samples to root node')
        self.inst2node_idx = self.assign_instance_to_root_node(self.data_bin, 0)

        tree_height = self.max_depth + 1  # non-leaf node height + 1 layer leaf

        for dep in range(tree_height):

            if dep + 1 == tree_height:

                for node in self.cur_layer_node:
                    node.is_leaf = True
                    self.tree_node.append(node)

                rest_sample_weights = self.get_node_sample_weights(self.inst2node_idx, self.tree_node)
                if self.sample_weights is None:
                    self.sample_weights = rest_sample_weights
                else:
                    self.sample_weights = self.sample_weights.union(rest_sample_weights)

                # stop fitting
                break

            LOGGER.debug('start to fit layer {}'.format(dep))

            table_with_assignment = self.update_instances_node_positions()

            # send current layer node number:
            self.sync_cur_layer_node_num(len(self.cur_layer_node), suffix=(dep, self.epoch_idx, self.tree_idx))

            split_info, agg_histograms = [], []
            for batch_id, i in enumerate(range(0, len(self.cur_layer_node), self.max_split_nodes)):
                cur_to_split = self.cur_layer_node[i:i+self.max_split_nodes]

                node_map = self.get_node_map(nodes=cur_to_split)
                LOGGER.debug('node map is {}'.format(node_map))
                LOGGER.debug('computing histogram for batch{} at depth{}'.format(batch_id, dep))
                local_histogram = self.get_left_node_local_histogram(
                    cur_nodes=cur_to_split,
                    tree=self.tree_node,
                    g_h=self.g_h,
                    table_with_assign=table_with_assignment,
                    split_points=self.bin_split_points,
                    sparse_point=self.bin_sparse_points,
                    valid_feature=self.valid_features
                )

                LOGGER.debug('federated finding best splits for batch{} at layer {}'.format(batch_id, dep))
                self.sync_local_node_histogram(local_histogram, suffix=(batch_id, dep, self.epoch_idx, self.tree_idx))

                agg_histograms += local_histogram

            split_info = self.sync_best_splits(suffix=(dep, self.epoch_idx))
            LOGGER.debug('got best splits from arbiter')

            new_layer_node = self.update_tree(self.cur_layer_node, split_info)
            self.cur_layer_node = new_layer_node

            self.inst2node_idx, leaf_val = self.assign_instances_to_new_node(table_with_assignment, self.tree_node)

            # record leaf val
            if self.sample_weights is None:
                self.sample_weights = leaf_val
            else:
                self.sample_weights = self.sample_weights.union(leaf_val)

            LOGGER.debug('assigning instance to new nodes done')

        self.convert_bin_to_real()
        LOGGER.debug('fitting tree done')

    def traverse_tree(self, data_inst: Instance, tree: List[Node], use_missing, zero_as_missing):

        nid = 0 # root node id
        while True:

            if tree[nid].is_leaf:
                return tree[nid].weight

            cur_node = tree[nid]
            fid, bid = cur_node.fid,cur_node.bid
            missing_dir = cur_node.missing_dir

            if use_missing and zero_as_missing:

                if data_inst.features.get_data(fid) == NoneType() or data_inst.features.get_data(fid, None) is None:

                    nid = tree[nid].right_nodeid if missing_dir == 1 else tree[nid].left_nodeid

                elif data_inst.features.get_data(fid) <= bid + consts.FLOAT_ZERO:
                    nid = tree[nid].left_nodeid
                else:
                    nid = tree[nid].right_nodeid

            elif data_inst.features.get_data(fid) == NoneType():

                nid = tree[nid].right_nodeid if missing_dir == 1 else tree[nid].left_nodeid

            elif data_inst.features.get_data(fid, 0) <= bid + consts.FLOAT_ZERO:
                nid = tree[nid].left_nodeid
            else:
                nid = tree[nid].right_nodeid

    def predict(self, data_inst):

        LOGGER.debug('tree start to predict')

        traverse_tree = functools.partial(self.traverse_tree,
                                          tree=self.tree_node,
                                          use_missing=self.use_missing,
                                          zero_as_missing=self.zero_as_missing,)

        predicted_weights = data_inst.mapValues(traverse_tree)

        return predicted_weights

    """
    Model Outputs
    """

    def get_model_meta(self):
        model_meta = DecisionTreeModelMeta()
        model_meta.criterion_meta.CopyFrom(CriterionMeta(criterion_method=self.criterion_method,
                                                         criterion_param=self.criterion_params))

        model_meta.max_depth = self.max_depth
        model_meta.min_sample_split = self.min_sample_split
        model_meta.min_impurity_split = self.min_impurity_split
        model_meta.min_leaf_node = self.min_leaf_node
        model_meta.use_missing = self.use_missing
        model_meta.zero_as_missing = self.zero_as_missing

        return model_meta

    def set_model_meta(self, model_meta):
        self.max_depth = model_meta.max_depth
        self.min_sample_split = model_meta.min_sample_split
        self.min_impurity_split = model_meta.min_impurity_split
        self.min_leaf_node = model_meta.min_leaf_node
        self.criterion_method = model_meta.criterion_meta.criterion_method
        self.criterion_params = list(model_meta.criterion_meta.criterion_param)
        self.use_missing = model_meta.use_missing
        self.zero_as_missing = model_meta.zero_as_missing

    def get_model_param(self):
        model_param = DecisionTreeModelParam()
        for node in self.tree_node:
            model_param.tree_.add(id=node.id,
                                  sitename=self.role,
                                  fid=node.fid,
                                  bid=node.bid,
                                  weight=node.weight,
                                  is_leaf=node.is_leaf,
                                  left_nodeid=node.left_nodeid,
                                  right_nodeid=node.right_nodeid,
                                  missing_dir=node.missing_dir)

        LOGGER.debug('output tree: epoch_idx:{} tree_idx:{}'.format(self.epoch_idx, self.tree_idx))
        return model_param

    def set_model_param(self, model_param):
        self.tree_node = []
        for node_param in model_param.tree_:
            _node = Node(id=node_param.id,
                         sitename=node_param.sitename,
                         fid=node_param.fid,
                         bid=node_param.bid,
                         weight=node_param.weight,
                         is_leaf=node_param.is_leaf,
                         left_nodeid=node_param.left_nodeid,
                         right_nodeid=node_param.right_nodeid,
                         missing_dir=node_param.missing_dir)

            self.tree_node.append(_node)

    def get_model(self):
        model_meta = self.get_model_meta()
        model_param = self.get_model_param()
        return model_meta, model_param

    def load_model(self, model_meta=None, model_param=None):
        LOGGER.info("load tree model")
        self.set_model_meta(model_meta)
        self.set_model_param(model_param)

    def compute_best_splits(self, *args):
        # not implemented in homo tree
        pass

    def initialize_root_node(self, *args):
        # not implemented in homo tree
        pass
