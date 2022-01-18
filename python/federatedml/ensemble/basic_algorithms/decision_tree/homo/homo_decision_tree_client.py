import functools
import numpy as np
from typing import List
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
from federatedml.feature.instance import Instance
from federatedml.param import DecisionTreeParam
from sklearn.ensemble._hist_gradient_boosting.grower import HistogramBuilder
from fate_arch.session import computing_session as session


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

        # memory backend
        self.arr_bin_data = None
        self.memory_hist_builder_list = []
        self.sample_id_arr = None
        self.bin_num = 0

        # secure aggregator, class SecureBoostClientAggregator
        if mode == 'train':
            self.role = role
            self.set_flowid(flow_id)
            self.aggregator = DecisionTreeClientAggregator(verbose=False)

        elif mode == 'predict':
            self.role, self.aggregator = None, None

        self.check_max_split_nodes()

        LOGGER.debug('use missing status {} {}'.format(self.use_missing, self.zero_as_missing))

    def set_flowid(self, flowid):
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
            return node.id

        data_inst = row[0]
        new_layer_nodeid = DecisionTree.go_next_layer(node, data_inst, use_missing, use_zero_as_missing,
                                                      bin_sparse_point=bin_sparse_point)
        return 1, new_layer_nodeid

    def assign_instances_to_new_node(self, table_with_assignment, tree_node: List[Node]):

        LOGGER.debug('re-assign instance to new nodes')
        assign_method = functools.partial(
            self.assign_an_instance,
            tree=tree_node,
            bin_sparse_point=self.bin_sparse_points,
            use_missing=self.use_missing,
            use_zero_as_missing=self.zero_as_missing)

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

    def convert_bin_to_real(self):
        """
        convert current bid in tree nodes to real value
        """
        for node in self.tree_node:
            if not node.is_leaf:
                node.bid = self.bin_split_points[node.fid][node.bid]

    def assign_instance_to_root_node(self, data_bin, root_node_id):
        return data_bin.mapValues(lambda inst: (1, root_node_id))

    def init_root_node_and_gh_sum(self):
        # compute local g_sum and h_sum
        g_sum, h_sum = self.get_grad_hess_sum(self.g_h)
        # get aggregated root info
        self.aggregator.send_local_root_node_info(g_sum, h_sum, suffix=('root_node_sync1', self.epoch_idx))
        g_h_dict = self.aggregator.get_aggregated_root_info(suffix=('root_node_sync2', self.epoch_idx))
        global_g_sum, global_h_sum = g_h_dict['g_sum'], g_h_dict['h_sum']
        # initialize node
        root_node = Node(
            id=0,
            sitename=consts.GUEST,
            sum_grad=global_g_sum,
            sum_hess=global_h_sum,
            weight=self.splitter.node_weight(
                global_g_sum,
                global_h_sum))
        self.cur_layer_node = [root_node]

    """
    Memory backend functions
    """

    def get_g_h_arr(self):

        g_, h_ = [], []
        for id_, gh in self.g_h.collect():
            g_.append(gh[0])
            h_.append(gh[1])
        g_, h_ = np.array(g_).astype(np.float32), np.array(h_).astype(np.float32)
        return g_, h_

    def init_node2index(self, sample_num):
        root_sample_idx = np.array([i for i in range(sample_num)]).astype(np.uint32)
        return root_sample_idx

    def init_memory_hist_builder(self, g, h, bin_data, bin_num):

        if len(g.shape) == 2:  # mo case
            idx_end = g.shape[1]
            for i in range(0, idx_end):
                g_arr = np.ascontiguousarray(g[::, i], dtype=np.float32)
                h_arr = np.ascontiguousarray(h[::, i], dtype=np.float32)
                hist_builder = HistogramBuilder(bin_data, bin_num, g_arr, h_arr, False)
                self.memory_hist_builder_list.append(hist_builder)
        else:
            self.memory_hist_builder_list.append(HistogramBuilder(bin_data, bin_num, g, h, False))

    def sklearn_compute_agg_hist(self, data_indices):

        hist = []
        for memory_hist_builder in self.memory_hist_builder_list:
            hist_memory_view = memory_hist_builder.compute_histograms_brute(data_indices)
            hist_arr = np.array(hist_memory_view)
            g = hist_arr['sum_gradients'].cumsum(axis=1)
            h = hist_arr['sum_hessians'].cumsum(axis=1)
            count = hist_arr['count'].cumsum(axis=1)

            final_hist = []
            for feat_idx in range(len(g)):
                arr = np.array([g[feat_idx], h[feat_idx], count[feat_idx]]).transpose()
                final_hist.append(arr)
            hist.append(np.array(final_hist))

        # non-mo case, return nd array
        if len(hist) == 1:
            return hist[0]

        # handle mo case, return list
        multi_dim_g, multi_dim_h, count = None, None, None
        for dimension_hist in hist:
            cur_g, cur_h = dimension_hist[::, ::, 0], dimension_hist[::, ::, 1]
            cur_g = cur_g.reshape(cur_g.shape[0], cur_g.shape[1], 1)
            cur_h = cur_h.reshape(cur_h.shape[0], cur_h.shape[1], 1)
            if multi_dim_g is None and multi_dim_h is None:
                multi_dim_g = cur_g
                multi_dim_h = cur_h
            else:
                multi_dim_g = np.concatenate([multi_dim_g, cur_g], axis=-1)
                multi_dim_h = np.concatenate([multi_dim_h, cur_h], axis=-1)

            if count is None:
                count = dimension_hist[::, ::, 2]

        # is a slow realization, to improve
        rs = []
        for feat_g, feat_h, feat_c in zip(multi_dim_g, multi_dim_h, count):
            feat_hist = [[g_arr, h_arr, c] for g_arr, h_arr, c in zip(feat_g, feat_h, feat_c)]
            rs.append(feat_hist)

        return rs

    def assign_arr_inst(self, node, data_arr, data_indices, missing_bin_index=None):

        # a fast inst assign using memory computing
        inst = data_arr[data_indices]
        fid = node.fid
        bid = node.bid
        decision = inst[::, fid] <= bid
        if self.use_missing and missing_bin_index is not None:
            missing_dir = True if node.missing_dir == -1 else False
            missing_samples = (inst[::, fid] == missing_bin_index)
            decision[missing_samples] = missing_dir
        left_samples = data_indices[decision]
        right_samples = data_indices[~decision]
        return left_samples, right_samples

    """
    Fit & Predict
    """

    def memory_fit(self):
        """
        fitting using memory backend
        """

        LOGGER.info('begin to fit homo decision tree, epoch {}, tree idx {},'
                    'running on memory backend'.format(self.epoch_idx, self.tree_idx))

        self.init_root_node_and_gh_sum()
        g, h = self.get_g_h_arr()
        self.init_memory_hist_builder(g, h, self.arr_bin_data, self.bin_num + self.use_missing)  # last missing bin
        root_indices = self.init_node2index(len(self.arr_bin_data))
        self.cur_layer_node[0].inst_indices = root_indices  # root node

        tree_height = self.max_depth + 1  # non-leaf node height + 1 layer leaf

        for dep in range(tree_height):

            if dep + 1 == tree_height:
                for node in self.cur_layer_node:
                    node.is_leaf = True
                    self.tree_node.append(node)
                break

            self.sync_cur_layer_node_num(len(self.cur_layer_node), suffix=(dep, self.epoch_idx, self.tree_idx))

            node_map = self.get_node_map(self.cur_layer_node)
            node_hists = []
            for batch_id, i in enumerate(range(0, len(self.cur_layer_node), self.max_split_nodes)):

                cur_to_split = self.cur_layer_node[i:i + self.max_split_nodes]

                for node in cur_to_split:
                    if node.id in node_map:
                        hist = self.sklearn_compute_agg_hist(node.inst_indices)
                        hist_bag = HistogramBag(hist)
                        hist_bag.hid = node.id
                        hist_bag.p_hid = node.parent_nodeid
                        node_hists.append(hist_bag)

                self.sync_local_node_histogram(node_hists, suffix=(batch_id, dep, self.epoch_idx, self.tree_idx))
                node_hists = []

            split_info = self.sync_best_splits(suffix=(dep, self.epoch_idx))
            new_layer_node = self.update_tree(self.cur_layer_node, split_info)
            node2inst_idx = []

            for node in self.cur_layer_node:
                if node.is_leaf:
                    continue
                l, r = self.assign_arr_inst(node, self.arr_bin_data, node.inst_indices, missing_bin_index=self.bin_num)
                node2inst_idx.append(l)
                node2inst_idx.append(r)
            assert len(node2inst_idx) == len(new_layer_node)

            for node, indices in zip(new_layer_node, node2inst_idx):
                node.inst_indices = indices
            self.cur_layer_node = new_layer_node

        sample_indices, weights = [], []

        for node in self.tree_node:
            if node.is_leaf:
                sample_indices += list(node.inst_indices)
                weights += [node.weight] * len(node.inst_indices)
            else:
                node.bid = self.bin_split_points[node.fid][int(node.bid)]

        # post-processing of memory backend fit
        sample_id = self.sample_id_arr[sample_indices]
        self.leaf_count = {}
        for node in self.tree_node:
            if node.is_leaf:
                self.leaf_count[node.id] = len(node.inst_indices)
        LOGGER.debug('leaf count is {}'.format(self.leaf_count))
        sample_id_type = type(self.g_h.take(1)[0][0])
        self.sample_weights = session.parallelize([(sample_id_type(id_), weight) for id_, weight in zip(
            sample_id, weights)], include_key=True, partition=self.data_bin.partitions)

    def fit(self):
        """
        start to fit
        """
        LOGGER.info('begin to fit homo decision tree, epoch {}, tree idx {},'
                    'running on distributed backend'.format(self.epoch_idx, self.tree_idx))

        self.init_root_node_and_gh_sum()
        LOGGER.debug('assign samples to root node')
        self.inst2node_idx = self.assign_instance_to_root_node(self.data_bin, 0)

        tree_height = self.max_depth + 1  # non-leaf node height + 1 layer leaf

        for dep in range(tree_height):

            if dep + 1 == tree_height:

                for node in self.cur_layer_node:
                    node.is_leaf = True
                    self.tree_node.append(node)

                rest_sample_leaf_pos = self.inst2node_idx.mapValues(lambda x: x[1])
                if self.sample_leaf_pos is None:
                    self.sample_leaf_pos = rest_sample_leaf_pos
                else:
                    self.sample_leaf_pos = self.sample_leaf_pos.union(rest_sample_leaf_pos)
                # stop fitting
                break

            LOGGER.debug('start to fit layer {}'.format(dep))

            table_with_assignment = self.update_instances_node_positions()

            # send current layer node number:
            self.sync_cur_layer_node_num(len(self.cur_layer_node), suffix=(dep, self.epoch_idx, self.tree_idx))

            split_info, agg_histograms = [], []
            for batch_id, i in enumerate(range(0, len(self.cur_layer_node), self.max_split_nodes)):
                cur_to_split = self.cur_layer_node[i:i + self.max_split_nodes]

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
            if self.sample_leaf_pos is None:
                self.sample_leaf_pos = leaf_val
            else:
                self.sample_leaf_pos = self.sample_leaf_pos.union(leaf_val)

            LOGGER.debug('assigning instance to new nodes done')

        self.convert_bin_to_real()
        self.sample_weights_post_process()
        LOGGER.debug('fitting tree done')

    def traverse_tree(self, data_inst: Instance, tree: List[Node], use_missing, zero_as_missing):

        nid = 0  # root node id
        while True:

            if tree[nid].is_leaf:
                return tree[nid].weight

            nid = DecisionTree.go_next_layer(tree[nid], data_inst, use_missing, zero_as_missing)

    def predict(self, data_inst):

        LOGGER.debug('tree start to predict')

        traverse_tree = functools.partial(self.traverse_tree,
                                          tree=self.tree_node,
                                          use_missing=self.use_missing,
                                          zero_as_missing=self.zero_as_missing, )

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
            weight, mo_weight = self.mo_weight_extract(node)

            model_param.tree_.add(id=node.id,
                                  sitename=self.role,
                                  fid=node.fid,
                                  bid=node.bid,
                                  weight=weight,
                                  is_leaf=node.is_leaf,
                                  left_nodeid=node.left_nodeid,
                                  right_nodeid=node.right_nodeid,
                                  missing_dir=node.missing_dir,
                                  mo_weight=mo_weight
                                  )

        model_param.leaf_count.update(self.leaf_count)
        return model_param

    def set_model_param(self, model_param):
        self.tree_node = []
        for node_param in model_param.tree_:
            weight = self.mo_weight_load(node_param)

            _node = Node(id=node_param.id,
                         sitename=node_param.sitename,
                         fid=node_param.fid,
                         bid=node_param.bid,
                         weight=weight,
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
