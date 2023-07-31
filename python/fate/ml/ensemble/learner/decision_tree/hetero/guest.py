from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import DecisionTree, Node, _get_sample_on_local_nodes, _update_sample_pos
from fate.ml.ensemble.learner.decision_tree.tree_core.hist import get_hist_builder
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import FedSklearnSplitter
from fate.arch import Context
from fate.arch.dataframe import DataFrame
import numpy as np
from typing import List
import functools
import logging
import copy


logger = logging.getLogger(__name__)


class HeteroDecisionTreeGuest(DecisionTree):

    def __init__(self, max_depth=3, feature_importance_type='split', valid_features=None, max_split_nodes=1024):
        super().__init__(max_depth, use_missing=False, zero_as_missing=False, feature_importance_type=feature_importance_type, valid_features=valid_features)
        self.host_sitenames = None
        self.max_split_nodes = max_split_nodes
        self._tree_node_num = 0
        self.hist_builder = None
        self.splitter = None

    def _get_column_max_bin(self, result_dict):
        bin_len = {}
        
        for column, values in result_dict.items():
            bin_num = len(values)
            bin_len[column] = bin_num 
        
        max_max_value = max(bin_len.values())
        
        return bin_len, max_max_value
    
    def _update_sample_pos(self, ctx, cur_layer_nodes: List[Node], sample_pos: DataFrame, data: DataFrame, node_map: dict):

        sitename = ctx.local.party[0] + '_' + ctx.local.party[1]
        data_with_pos = DataFrame.hstack([data, sample_pos])
        map_func = functools.partial(_get_sample_on_local_nodes, cur_layer_node=cur_layer_nodes, node_map=node_map, sitename=sitename)
        local_sample_idx = data_with_pos.apply_row(map_func)
        # local_sample_idx = data_with_pos.apply_row(map_func).values.as_tensor()
        local_samples = data_with_pos.loc(local_sample_idx.get_indexer(target="sample_id"), preserve_order=True)[local_sample_idx.values.as_tensor()]
        logger.info('{}/{} samples on local nodes'.format(len(local_samples), len(data)))

        if len(local_samples) == 0:
            updated_sample_pos = None
        else:
            updated_sample_pos = sample_pos.loc(local_samples.get_indexer(target="sample_id"), preserve_order=True).create_frame()
            update_func = functools.partial(_update_sample_pos, cur_layer_node=cur_layer_nodes, node_map=node_map)
            map_rs = local_samples.apply_row(update_func)
            updated_sample_pos["node_idx"] = map_rs # local_samples.apply_row(update_func)

        # synchronize sample pos
        host_update_sample_pos = ctx.hosts.get('updated_data')
        new_sample_pos = sample_pos.empty_frame()

        for host_data in host_update_sample_pos:
            if host_data[0]:  # True
                pos_data, pos_index = host_data[1]
                tmp_frame = sample_pos.create_frame()
                tmp_frame = tmp_frame.loc(pos_index, preserve_order=True)
                tmp_frame['node_idx'] = pos_data
                new_sample_pos = DataFrame.vstack([new_sample_pos, tmp_frame])

        if updated_sample_pos is not None:
            if len(updated_sample_pos) == len(data):  # all samples are on local
                new_sample_pos = updated_sample_pos
            else:
                logger.info('stack new sample pos, guest len {}, host len {}'.format(len(updated_sample_pos), len(new_sample_pos)))
                new_sample_pos = DataFrame.vstack([updated_sample_pos, new_sample_pos])
        else:
            new_sample_pos = new_sample_pos  # all samples are on host

       # share new sample position with all hosts
        print('new sample pos {}'.format(new_sample_pos.as_pd_df()))
        ctx.hosts.put('new_sample_pos', (new_sample_pos.as_tensor(), new_sample_pos.get_indexer(target='sample_id')))
        return new_sample_pos

    def _send_gh(self, ctx: Context, grad_and_hess: DataFrame):
        
        # now skip encrypt
        ctx.hosts.put('en_gh', grad_and_hess.as_pd_df())

    def _mask_node(self, ctx: Context, nodes: List[Node]):
        new_nodes = []
        for n in nodes:
            new_nodes.append(Node(nid=n.nid, is_leaf=n.is_leaf,  l=n.l, r=n.r, is_left_node=n.is_left_node, split_id=n.split_id, sitename=n.sitename))
        return new_nodes

    def _check_assign_result(self, sample_pos: DataFrame, cur_layer_node: List[Node]):
        # debugging function
        sample_pos_df = sample_pos.as_pd_df()
        sample_pos_count = sample_pos_df.groupby('node_idx').count().to_dict()['sample_id']
        for node in cur_layer_node:
            nid = node.nid
            sample_count_0 = node.sample_num
            sample_count_1 = sample_pos_count[nid]
            if sample_count_0 != sample_count_1:
                parent_nid = node.parent_nodeid
                for i in self._nodes:
                    if i.nid == parent_nid:
                        logger.info('parent node {}'.format(i))
                raise ValueError('node {} sample count not match, {} vs {}, node details {}'.format(nid, sample_count_0, sample_count_1, node))

    def _sync_nodes(self, ctx: Context, cur_layer_nodes: List[Node], next_layer_nodes: List[Node]):
        
        mask_cur_layer = self._mask_node(ctx, cur_layer_nodes)
        mask_next_layer = self._mask_node(ctx, next_layer_nodes)
        ctx.hosts.put('sync_nodes', [mask_cur_layer, mask_next_layer])

    def booster_fit(self, ctx: Context, bin_train_data: DataFrame, grad_and_hess: DataFrame, bining_dict: dict):
        
        # Initialization
        train_df = bin_train_data
        feat_max_bin, max_bin = self._get_column_max_bin(bining_dict)
        sample_pos = self._init_sample_pos(train_df)
        self._sample_on_leaves = sample_pos.empty_frame()
        root_node = self._initialize_root_node(ctx, grad_and_hess)

        # Send Encrypted Grad and Hess
        self._send_gh(ctx, grad_and_hess)

        # init histogram builder
        self.hist_builder = get_hist_builder(train_df, grad_and_hess, root_node, max_bin, hist_type='sklearn')

        # init splitter
        self.splitter = FedSklearnSplitter(bining_dict)

        # Prepare for training
        node_map = {}
        cur_layer_node = [root_node]

        for cur_depth, sub_ctx in ctx.on_iterations.ctxs_range(self.max_depth):
            
            if len(cur_layer_node) == 0:
                logger.info('no nodes to split, stop training')
                break
            
            self._check_assign_result(sample_pos, cur_layer_node)
            node_map = {n.nid: idx for idx, n in enumerate(cur_layer_node)}
            # compute histogram
            hist, indices = self.hist_builder.compute_hist(cur_layer_node, train_df, grad_and_hess, sample_pos, node_map, debug=True)
            # compute best splits
            split_info = self.splitter.split(sub_ctx, hist, cur_layer_node)
            # update tree with best splits
            next_layer_nodes = self._update_tree(ctx, cur_layer_node, split_info)
            # sync nodes
            self._sync_nodes(sub_ctx, cur_layer_node, next_layer_nodes)
            # update sample positions
            sample_pos = self._update_sample_pos(sub_ctx, cur_layer_node, sample_pos, train_df, node_map)
            # if sample reaches leaf nodes, drop them
            sample_on_leaves = self._get_samples_on_leaves(sample_pos)
            train_df, sample_pos = self._drop_samples_on_leaves(sample_pos, train_df)
            self._sample_on_leaves = DataFrame.vstack([self._sample_on_leaves, sample_on_leaves])
            # next layer nodes
            cur_layer_node = next_layer_nodes
            logger.info('layer {} done: next layer will split {} nodes, active samples num {}'.format(cur_depth, len(cur_layer_node), len(sample_pos)))
            self.next_layer_node = next_layer_nodes

        # handle final leaves
        if len(cur_layer_node) != 0:
            for node in cur_layer_node:
                node.is_leaf = True
                node.sitename = ctx.guest.party[0] + '_' + ctx.guest.party[1]
                self._nodes.append(node)
                self._sample_on_leaves = DataFrame.vstack([self._sample_on_leaves, sample_pos])

        # when training is done, all samples must be on leaves
        assert len(self._sample_on_leaves) == len(bin_train_data), 'sample on leaves num not match, {} vs {}'.format(len(self._sample_on_leaves), len(bin_train_data))
        # convert sample pos to weights
        self._sample_weights = self._convert_sample_pos_to_weight(self._sample_on_leaves, self._nodes)
        # convert bid to split value
        # self._nodes = self._convert_bin_idx_to_split_val(ctx, self._nodes, bining_dict, bin_train_data.schema)

    def fit(self, ctx: Context, train_data: DataFrame):
        pass

    def predict(self, ctx: Context, data_inst: DataFrame):
        pass
    
