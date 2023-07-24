from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import DecisionTree, Node, FLOAT_ZERO
from fate.ml.ensemble.learner.decision_tree.tree_core.hist import SklearnHistBuilder
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import SklearnSplitter, SplitInfo
from fate.arch import Context
from fate.arch.dataframe import DataFrame
import numpy as np
import pandas as pd
from typing import List
import functools
import logging


logger = logging.getLogger(__name__)


def make_decision(feat_val, bid, missing_dir=None, use_missing=None, zero_as_missing=None, zero_val=0):

    # no missing val
    left, right = True, False
    direction = left if feat_val <= bid + FLOAT_ZERO else right
    return direction


def _update_sample_pos(s: pd.Series, cur_layer_node: List[Node], node_map: dict):

    node_id = s[-1]
    node = cur_layer_node[node_map[node_id]]
    if node.is_leaf:
        return -1  # reach leaf
    feat_val = s[node.fid]
    bid = node.bid
    
    dir_ = make_decision(feat_val, bid)
    if dir_:  # go left
        ret_node = node.l
    else:
        ret_node = node.r
    # print(node_id, feat_val, bid, dir_, ret_node)
    return ret_node

class HeteroDecisionTreeGuest(DecisionTree):

    def __init__(self, max_depth=3, feature_importance_type='split', valid_features=None, max_split_nodes=1024, sitename='local'):
        super().__init__(max_depth, use_missing=False, zero_as_missing=False, feature_importance_type=feature_importance_type, valid_features=valid_features)
        self.sitename = sitename
        self.max_split_nodes = max_split_nodes
        self._tree_node_num = 0
        self.hist_builder = None
        self.splitter = None

    def _compute_best_splits(self):
        pass

    def get_column_max_bin(self, result_dict):
        bin_len = {}
        
        for column, values in result_dict.items():
            bin_num = len(values)
            bin_len[column] = bin_num 
        
        max_max_value = max(bin_len.values())
        
        return bin_len, max_max_value
    
    def get_sklearn_hist_builder(self, bin_train_data, grad_and_hess, root_node, max_bin):
        data = bin_train_data.as_pd_df()
        data['sample_id'] = data['sample_id'].astype(np.uint32)
        gh = grad_and_hess.as_pd_df()
        gh['sample_id'] = gh['sample_id'].astype(np.uint32)
        collect_data = data.sort_values(by='sample_id')
        collect_gh = gh.sort_values(by='sample_id')
        root_node.set_inst_indices(collect_gh['sample_id'].values)
        feat_arr = collect_data.drop(columns=[bin_train_data.schema.sample_id_name, bin_train_data.schema.label_name, bin_train_data.schema.match_id_name]).values
        g = collect_gh['g'].values
        h = collect_gh['h'].values
        feat_arr = np.asfortranarray(feat_arr.astype(np.uint8))
        return SklearnHistBuilder(feat_arr, max_bin, g, h)
    
    def update_tree(self, split_infos, cur_layer_node):
        pass
    
    def get_sklearn_splitter(self, bin_train_data, grad_and_hess, root_node, max_bin):
        pass

    def get_distribute_hist_builder(self, bin_train_data, grad_and_hess, root_node, max_bin):
        pass
    
    def update(self, nodes, cur_to_split: List[Node], split_info: List[SplitInfo], sample_pos: DataFrame, data: DataFrame, node_map: dict):
        
        assert len(cur_to_split) == len(split_info), 'node num not match split info num, got {} node vs {} split info'.format(len(cur_to_split), len(split_info))

        next_layer_node = []

        data_with_pos = DataFrame.hstack([data, sample_pos])
        new_sample_pos = data.create_frame()

        for idx in range(len(split_info)):

            node: Node = cur_to_split[idx]

            if split_info[idx] is None:
                node.is_leaf = True
                self._nodes.append(node)
                continue

            print('split info is {}'.format(split_info))

            sum_grad = node.grad
            sum_hess = node.hess
            node.fid = split_info[idx].best_fid
            node.bid = split_info[idx].best_bid
            node.missing_dir = split_info[idx].missing_dir

            p_id = node.nid
            l_id, r_id = self._tree_node_num + 1, self._tree_node_num + 2
            self._tree_node_num += 2
            node.l, node.r = l_id, r_id

            l_g, l_h = split_info[idx].sum_grad, split_info[idx].sum_hess

            # create new left node and new right node
            left_node = Node(nid=l_id,
                             sitename=self.sitename,
                             grad=float(l_g),
                             hess=float(l_h),
                             weight=float(self.splitter.node_weight(l_g, l_h)),
                             parent_nodeid=p_id,
                             sibling_nodeid=r_id,
                             is_left_node=True)
            right_node = Node(nid=r_id,
                              sitename=self.sitename,
                              grad=float(sum_grad - l_g),
                              hess=float(sum_hess - l_h),
                              weight=float(self.splitter.node_weight(sum_grad - l_g, sum_hess - l_h)),
                              parent_nodeid=p_id,
                              sibling_nodeid=l_id,
                              is_left_node=False)
            
            next_layer_node.append(left_node)
            next_layer_node.append(right_node)
            self._nodes.append(node)

        map_func = functools.partial(_update_sample_pos, cur_layer_node=cur_to_split, node_map=node_map)
        new_sample_pos['node_idx'] = data_with_pos.apply_row(map_func)

        need_drop_idx = (new_sample_pos['node_idx'] != -1).as_tensor()
        new_sample_pos = new_sample_pos[need_drop_idx]
        new_data = data[need_drop_idx]

        return next_layer_node, new_sample_pos, new_data

    def booster_fit(self, ctx: Context, bin_train_data: DataFrame, grad_and_hess: DataFrame, bining_dict: dict):
        
        feat_max_bin, max_bin = self.get_column_max_bin(bining_dict)
        sample_pos = self._init_sample_pos(bin_train_data)
        root_node = self._initialize_root_node(grad_and_hess, ctx.guest.party[0] + '_' + ctx.guest.party[1])
        self._nodes.append(root_node)
        # init histogram builder
        self.hist_builder = self.get_sklearn_hist_builder(bin_train_data, grad_and_hess, root_node, max_bin)
        # init splitter
        self.splitter = SklearnSplitter(bining_dict)

        cur_layer_node = [root_node]
        for cur_depth in range(self.max_depth):
            
            if len(cur_layer_node) == 0:
                break

            node_map = {n.nid: idx for idx, n in enumerate(cur_layer_node)}
            # compute histogram
            hist = self.hist_builder.compute_hist(cur_layer_node, bin_train_data, grad_and_hess, sample_pos, node_map)
            # compute best splits
            split_info = self.splitter.split(hist, cur_layer_node)
            # update tree with best splits
            cur_layer_node, sample_pos, bin_train_data = self.update(self._nodes, cur_layer_node, split_info, sample_pos, bin_train_data, node_map)

            logger.info('layer {} done: next layer will split {} nodes, active samples num {}'.format(cur_depth, len(cur_layer_node), len(sample_pos)))

        # handle final leaves
        if len(cur_layer_node) != 0:
            for node in cur_layer_node:
                node.is_leaf = True
                self._nodes.append(node)

        return cur_layer_node, sample_pos
    
    def get_nodes(self):
        return self._nodes
    
    def print_tree(self, show_path=False):
        nodes = self._nodes
        def print_node(node, prefix=""):
            if node is not None:
                info_str = "(id: " + str(node.nid)
                if node.is_leaf:
                    info_str += ", weight: " + str(node.weight)
                    info_str += " leaf)"
                else:
                    info_str += ", fid {}, split {}".format(node.fid, node.bid)
                    info_str += ")"
                
                if not node.is_leaf:
                    print_node(next((n for n in nodes if n.nid == node.r), None), prefix + "--R--> ")
                
                if not show_path:
                    prefix = " " * len(prefix)
                print(f"{prefix}id: {node.nid}", info_str)

                if not node.is_leaf:
                    print_node(next((n for n in nodes if n.nid == node.l), None), prefix + "--L--> ")

        print_node(nodes[0])

    def fit(self, ctx: Context, train_data: DataFrame):
        pass

    def predict(self, ctx: Context, data_inst: DataFrame):
        pass
    
