from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import DecisionTree, Node, _get_sample_on_local_nodes, _update_sample_pos
from fate.ml.ensemble.learner.decision_tree.tree_core.hist import get_hist_builder
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import SklearnSplitter
from fate.arch import Context
from fate.arch.dataframe import DataFrame
import numpy as np
from typing import List
import functools
import logging


logger = logging.getLogger(__name__)


class HeteroDecisionTreeGuest(DecisionTree):

    def __init__(self, max_depth=3, feature_importance_type='split', valid_features=None, max_split_nodes=1024, sitename='local'):
        super().__init__(max_depth, use_missing=False, zero_as_missing=False, feature_importance_type=feature_importance_type, valid_features=valid_features)
        self.sitename = sitename
        self.max_split_nodes = max_split_nodes
        self._tree_node_num = 0
        self.hist_builder = None
        self.splitter = None

    def get_column_max_bin(self, result_dict):
        bin_len = {}
        
        for column, values in result_dict.items():
            bin_num = len(values)
            bin_len[column] = bin_num 
        
        max_max_value = max(bin_len.values())
        
        return bin_len, max_max_value
    
    def update_sample_pos(self, cur_to_split: List[Node], sample_pos: DataFrame, data: DataFrame, node_map: dict, sitename: str):

        data_with_pos = DataFrame.hstack([data, sample_pos])
        map_func = functools.partial(_get_sample_on_local_nodes, cur_layer_node=cur_to_split, node_map=node_map, sitename=sitename)
        local_sample_idx = data_with_pos.apply_row(map_func).values.as_tensor()
        local_samples = data_with_pos[local_sample_idx]
        logger.info('{} samples on local nodes'.format(len(local_samples)))
        new_sample_pos = sample_pos.loc(local_samples.get_indexer(target="sample_id"), preserve_order=True).create_frame()
        update_func = functools.partial(_update_sample_pos, cur_layer_node=cur_to_split, node_map=node_map)
        new_sample_pos['node_idx'] = local_samples.apply_row(update_func)
        return new_sample_pos
    
    def send_gh(self, ctx: Context, grad_and_hess: DataFrame):
        
        # now skip encrypt
        ctx.hosts.put('en_gh', grad_and_hess.as_pd_df())

    def booster_fit(self, ctx: Context, bin_train_data: DataFrame, grad_and_hess: DataFrame, bining_dict: dict):
        
        feat_max_bin, max_bin = self.get_column_max_bin(bining_dict)
        sample_pos = self._init_sample_pos(bin_train_data)
        root_node = self._initialize_root_node(grad_and_hess, self.sitename)
        # Send Encrypted Grad and Hess
        self.send_gh(ctx, grad_and_hess)
        # init histogram builder
        self.hist_builder = get_hist_builder(bin_train_data, grad_and_hess, root_node, max_bin, hist_type='sklearn')
        # init splitter
        self.splitter = SklearnSplitter(bining_dict)
        node_map = {}
        cur_layer_node = [root_node]
        for cur_depth in range(self.max_depth):
            
            if len(cur_layer_node) == 0:
                logger.info('no nodes to split, stop training')
                break

            node_map = {n.nid: idx for idx, n in enumerate(cur_layer_node)}
            # compute histogram
            hist = self.hist_builder.compute_hist(cur_layer_node, bin_train_data, grad_and_hess, sample_pos, node_map)
            # compute best splits
            split_info = self.splitter.split(ctx, hist, cur_layer_node)
            # update tree with best splits
            next_layer_nodes = self.update_tree(cur_layer_node, split_info)
            sample_pos = self.update_sample_pos(cur_layer_node, sample_pos, bin_train_data, node_map, self.sitename)
            bin_train_data, sample_pos = self.drop_leaf_samples(sample_pos, bin_train_data)
            cur_layer_node = next_layer_nodes

            logger.info('layer {} done: next layer will split {} nodes, active samples num {}'.format(cur_depth, len(cur_layer_node), len(sample_pos)))

        # handle final leaves
        if len(cur_layer_node) != 0:
            for node in cur_layer_node:
                node.is_leaf = True
                self._nodes.append(node)

    def fit(self, ctx: Context, train_data: DataFrame):
        pass

    def predict(self, ctx: Context, data_inst: DataFrame):
        pass
    
