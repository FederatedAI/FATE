from typing import List
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import DecisionTreeModelParam
from federatedml.transfer_variable.transfer_class.homo_decision_tree_transfer_variable import \
    HomoDecisionTreeTransferVariable
from federatedml.util import consts
from federatedml.ensemble import DecisionTree
from federatedml.ensemble import Splitter
from federatedml.ensemble import HistogramBag
from federatedml.ensemble import SplitInfo
from federatedml.util import LOGGER
from federatedml.ensemble import DecisionTreeArbiterAggregator


class HomoDecisionTreeArbiter(DecisionTree):

    def __init__(self, tree_param: DecisionTreeModelParam, valid_feature: dict, epoch_idx: int,
                 tree_idx: int, flow_id: int):

        super(HomoDecisionTreeArbiter, self).__init__(tree_param)
        self.splitter = Splitter(self.criterion_method, self.criterion_params, self.min_impurity_split,
                                 self.min_sample_split, self.min_leaf_node, self.min_child_weight)

        self.transfer_inst = HomoDecisionTreeTransferVariable()
        """
        initializing here
        """
        self.valid_features = valid_feature

        self.tree_node = []  # start from root node
        self.tree_node_num = 0
        self.cur_layer_node = []

        self.runtime_idx = 0
        self.sitename = consts.ARBITER
        self.epoch_idx = epoch_idx
        self.tree_idx = tree_idx

        # secure aggregator
        self.set_flowid(flow_id)
        self.aggregator = DecisionTreeArbiterAggregator(verbose=False)

        # stored histogram for faster computation {node_id:histogram_bag}
        self.stored_histograms = {}

    def set_flowid(self, flowid=0):
        LOGGER.info("set flowid, flowid is {}".format(flowid))
        self.transfer_inst.set_flowid(flowid)

    """
    Federation Functions
    """

    def sync_node_sample_numbers(self, suffix):
        cur_layer_node_num = self.transfer_inst.cur_layer_node_num.get(-1, suffix=suffix)
        for num in cur_layer_node_num[1:]:
            assert num == cur_layer_node_num[0]
        return cur_layer_node_num[0]

    def sync_best_splits(self, split_info, suffix):
        LOGGER.debug('sending best split points')
        self.transfer_inst.best_split_points.remote(split_info, idx=-1, suffix=suffix)

    def sync_local_histogram(self, suffix) -> List[HistogramBag]:

        node_local_histogram = self.aggregator.aggregate_histogram(suffix=suffix)
        LOGGER.debug('num of histograms {}'.format(len(node_local_histogram)))
        return node_local_histogram

    """
    Split finding
    """

    def federated_find_best_split(self, node_histograms, parallel_partitions=10) -> List[SplitInfo]:

        LOGGER.debug('aggregating histograms')
        acc_histogram = node_histograms
        best_splits = self.splitter.find_split(acc_histogram, self.valid_features, parallel_partitions,
                                               self.sitename, self.use_missing, self.zero_as_missing)
        return best_splits

    @staticmethod
    def histogram_subtraction(left_node_histogram, stored_histograms):
        # histogram subtraction
        all_histograms = []
        for left_hist in left_node_histogram:
            all_histograms.append(left_hist)
            # LOGGER.debug('hist id is {}, pid is {}'.format(left_hist.hid, left_hist.p_hid))
            # root node hist
            if left_hist.hid == 0:
                continue
            right_hist = stored_histograms[left_hist.p_hid] - left_hist
            right_hist.hid, right_hist.p_hid = left_hist.hid + 1, right_hist.p_hid
            all_histograms.append(right_hist)

        return all_histograms

    """
    Fit
    """

    def fit(self):

        LOGGER.info('begin to fit homo decision tree, epoch {}, tree idx {}'.format(self.epoch_idx, self.tree_idx))

        g_sum, h_sum = self.aggregator.aggregate_root_node_info(suffix=('root_node_sync1', self.epoch_idx))

        self.aggregator.broadcast_root_info(g_sum, h_sum, suffix=('root_node_sync2', self.epoch_idx))

        if self.max_split_nodes != 0 and self.max_split_nodes % 2 == 1:
            self.max_split_nodes += 1
            LOGGER.warning('an even max_split_nodes value is suggested when using histogram-subtraction, '
                           'max_split_nodes reset to {}'.format(self.max_split_nodes))

        tree_height = self.max_depth + 1  # non-leaf node height + 1 layer leaf

        for dep in range(tree_height):

            if dep + 1 == tree_height:
                break

            LOGGER.debug('current dep is {}'.format(dep))

            split_info = []
            # get cur layer node num:
            cur_layer_node_num = self.sync_node_sample_numbers(suffix=(dep, self.epoch_idx, self.tree_idx))

            layer_stored_hist = {}

            for batch_id, i in enumerate(range(0, cur_layer_node_num, self.max_split_nodes)):

                left_node_histogram = self.sync_local_histogram(suffix=(batch_id, dep, self.epoch_idx, self.tree_idx))
                all_histograms = self.histogram_subtraction(left_node_histogram, self.stored_histograms)
                # store histogram
                for hist in all_histograms:
                    layer_stored_hist[hist.hid] = hist

                # FIXME stable parallel_partitions
                best_splits = self.federated_find_best_split(all_histograms, parallel_partitions=10)
                split_info += best_splits

            self.stored_histograms = layer_stored_hist
            self.sync_best_splits(split_info, suffix=(dep, self.epoch_idx))
            LOGGER.debug('best_splits_sent')

    def predict(self, data_inst=None):
        """
        Do nothing
        """
        LOGGER.debug('start predicting')

    """
    These functions are not needed in homo-decision-tree
    """

    def initialize_root_node(self, *args):
        pass

    def compute_best_splits(self, *args):
        pass

    def assign_an_instance(self, *args):
        pass

    def assign_instances_to_new_node(self, *args):
        pass

    def update_tree(self, *args):
        pass

    def convert_bin_to_real(self, *args):
        pass

    def get_model_meta(self):
        pass

    def get_model_param(self):
        pass

    def set_model_param(self, model_param):
        pass

    def set_model_meta(self, model_meta):
        pass

    def traverse_tree(self, *args):
        pass

    def update_instances_node_positions(self, *args):
        pass
