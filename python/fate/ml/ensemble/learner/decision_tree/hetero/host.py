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

    def __init__(self, max_depth=3, valid_features=None, max_split_nodes=1024):
        super().__init__(max_depth, use_missing=False, zero_as_missing=False, valid_features=valid_features)
        self.max_split_nodes = max_split_nodes
        self._tree_node_num = 0
        self.hist_builder = None
        self.splitter = None
        self.node_fid_bid = {}

    def _convert_split_id_and_record(self, ctx: Context, cur_layer_nodes: List[Node], map_dict: List[dict], record_dict: dict):

        sitename = ctx.local.party[0] + '_' + ctx.local.party[1]
        for idx, n in enumerate(cur_layer_nodes):
            if (not n.is_leaf) and n.sitename == sitename:
                node_mapdict = map_dict[idx]
                fid, bid = node_mapdict[n.split_id]
                record_dict[n.nid] = (fid, bid)
                n.fid = fid
                n.bid = bid

    def _update_sample_pos(self, ctx, cur_layer_nodes: List[Node], sample_pos: DataFrame, data: DataFrame, node_map: dict):

        sitename = ctx.local.party[0] + '_' + ctx.local.party[1]
        data_with_pos = DataFrame.hstack([data, sample_pos])
        map_func = functools.partial(_get_sample_on_local_nodes, cur_layer_node=cur_layer_nodes, node_map=node_map, sitename=sitename)
        local_sample_idx = data_with_pos.apply_row(map_func).values.as_tensor()
        local_samples = data_with_pos[local_sample_idx]
        logger.info('{} samples on local nodes'.format(len(local_samples)))

        if len(local_samples) == 0:
            updated_sample_pos = None
        else:
            updated_sample_pos = sample_pos.loc(local_samples.get_indexer(target="sample_id"), preserve_order=True).create_frame()
            update_func = functools.partial(_update_sample_pos, cur_layer_node=cur_layer_nodes, node_map=node_map)
            updated_sample_pos['node_idx'] = local_samples.apply_row(update_func)

        # synchronize sample pos
        if updated_sample_pos is None:
            update_data = (False, None)
        else:
            pos_data = updated_sample_pos.as_tensor()
            pos_index = updated_sample_pos.get_indexer(target='sample_id')
            update_data = (True, (pos_data, pos_index))
        ctx.guest.put('updated_data', update_data)
        new_pos_data, new_pos_indexer = ctx.guest.get('new_sample_pos')
        new_sample_pos = sample_pos.create_frame()
        new_sample_pos = new_sample_pos.loc(new_pos_indexer, preserve_order=True)
        new_sample_pos['node_idx'] = new_pos_data

        return new_sample_pos
    
    def _get_gh(self, ctx: Context):
        grad_and_hess = ctx.guest.get('en_gh')
        
        return grad_and_hess
    
    def _sync_nodes(self, ctx: Context):
        
        nodes = ctx.guest.get('sync_nodes')
        cur_layer_nodes, next_layer_nodes = nodes
        return cur_layer_nodes, next_layer_nodes
    
    def booster_fit(self, ctx: Context, bin_train_data: DataFrame, bining_dict: dict):
        
        train_df = bin_train_data
        feat_max_bin, max_bin = self._get_column_max_bin(bining_dict)
        sample_pos = self._init_sample_pos(train_df)

        # Get Encrypted Grad And Hess
        grad_and_hess = ctx.guest.get('en_gh')
        root_node = self._initialize_root_node(ctx, grad_and_hess)
        self.hist_builder = get_hist_builder(train_df, grad_and_hess, root_node, max_bin, hist_type='sklearn')
        self.splitter = FedSklearnSplitter(bining_dict)

        node_map = {}
        cur_layer_node = [root_node]
        for cur_depth, sub_ctx in ctx.on_iterations.ctxs_range(self.max_depth):
            
            if len(cur_layer_node) == 0:
                logger.info('no nodes to split, stop training')
                break

            node_map = {n.nid: idx for idx, n in enumerate(cur_layer_node)}
            # compute histogram
            hist = self.hist_builder.compute_hist(cur_layer_node, train_df, grad_and_hess, sample_pos, node_map)
            _, split_id_map = self.splitter.split(sub_ctx, hist, cur_layer_node)
            cur_layer_node, next_layer_nodes = self._sync_nodes(sub_ctx)
            self._convert_split_id_and_record(sub_ctx, cur_layer_node, split_id_map, self.node_fid_bid)
            logger.info('cur layer node num: {}, next layer node num: {}'.format(len(cur_layer_node), len(next_layer_nodes)))
            sample_pos = self._update_sample_pos(sub_ctx, cur_layer_node, sample_pos, train_df, node_map)
            train_df, sample_pos = self._drop_samples_on_leaves(sample_pos, train_df)
            cur_layer_node = next_layer_nodes
            logger.info('layer {} done: next layer will split {} nodes, active samples num {}'.format(cur_depth, len(cur_layer_node), len(sample_pos)))

        # convert bid to split value
        # self._nodes = self._convert_bin_idx_to_split_val(self._nodes, bining_dict)


    def fit(self, ctx: Context, train_data: DataFrame):
        pass

    def predict(self, ctx: Context, data_inst: DataFrame):
        pass
    
