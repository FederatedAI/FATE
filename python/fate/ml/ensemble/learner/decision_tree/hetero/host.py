from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import DecisionTree, Node, _get_sample_on_local_nodes, _update_sample_pos
from fate.ml.ensemble.learner.decision_tree.tree_core.hist import get_hist_builder
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import FedSklearnSplitter
from fate.arch import Context
from fate.arch.dataframe import DataFrame
import numpy as np
from typing import List
import functools
import logging


logger = logging.getLogger(__name__)


class HeteroDecisionTreeHost(DecisionTree):

    def __init__(self, max_depth=3, valid_features=None, max_split_nodes=1024, sitename='local'):
        super().__init__(max_depth, use_missing=False, zero_as_missing=False, valid_features=valid_features)
        self.sitename = sitename
        self.max_split_nodes = max_split_nodes
        self._tree_node_num = 0
        self.hist_builder = None
        self.splitter = None
    
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
    
    def get_gh(self, ctx: Context):
        grad_and_hess = ctx.guest.get('en_gh')
        
        return grad_and_hess
    
    def booster_fit(self, ctx: Context, bin_train_data: DataFrame, bining_dict: dict):
        
        feat_max_bin, max_bin = self.get_column_max_bin(bining_dict)
        sample_pos = self._init_sample_pos(bin_train_data)

        # Get Encrypted Grad And Hess
        grad_and_hess = ctx.guest.get('en_gh')
        print('got grad and hess {}'.format(grad_and_hess))
        root_node = self._initialize_root_node(grad_and_hess, self.sitename)
        self.hist_builder = get_hist_builder(bin_train_data, grad_and_hess, root_node, max_bin, hist_type='sklearn')
        self.splitter = FedSklearnSplitter(bining_dict)

        node_map = {}
        cur_layer_node = [root_node]
        for cur_depth in range(self.max_depth):
            
            if len(cur_layer_node) == 0:
                logger.info('no nodes to split, stop training')
                break

            node_map = {n.nid: idx for idx, n in enumerate(cur_layer_node)}
            # compute histogram
            hist = self.hist_builder.compute_hist(cur_layer_node, bin_train_data, grad_and_hess, sample_pos, node_map)
            split_ret = self.splitter.split(ctx, hist, cur_layer_node)
            return hist

        return grad_and_hess

    def fit(self, ctx: Context, train_data: DataFrame):
        pass

    def predict(self, ctx: Context, data_inst: DataFrame):
        pass
    
