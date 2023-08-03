import pandas as pd
import numpy as np
from typing import List
from fate.arch import Context
from fate.arch.dataframe import DataFrame
import copy
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import DecisionTree, _make_decision, Node
import functools


def get_dtype(max_int):
    if max_int < (2**8) / 2:
        return np.int8
    elif max_int < (2**16) / 2:
        return np.int16
    else:
        return np.int64
    

def all_reach_leaf(pos: np.array):
    return np.all(pos < 0)


def not_finished(pos: np.array):
    return not np.all(pos < 0)


def generate_pos_array(tree_num, max_node_num):
    dtype = get_dtype(max_node_num)
    # return list as a column of dataframe
    return [[np.zeros(tree_num, dtype=dtype)]]


def go_deep(s: pd.Series, tree: List[Node], sitename, cur_node_id):
    
    node: Node = tree[cur_node_id]
    while True:
        if node.is_leaf:
            return -(node.nid + 1)
        elif node.sitename != sitename:
            return node.nid
        else:
            fid = node.fid
            split_val = node.bid
            sample_feat_val = s[fid]
            is_left = _make_decision(sample_feat_val, split_val)
            if is_left:
                node = tree[node.l]
            else:
                node = tree[node.r]
    

def traverse_tree(s: pd.Series, trees: List[List[Node]], sitename: str):
    
    sample_pos = s['sample_pos'][0]
    new_sample_pos = copy.deepcopy(sample_pos)  # deepcopy to avoid inplace modification, for spark
    
    tree_idx = 0
    for node_pos, tree in zip(sample_pos, trees):
        
        if node_pos < 0:  # sample already reaches leaf node in this tree
            continue
        
        cur_node_id = node_pos
        end_node_id = go_deep(s, tree, sitename, cur_node_id)
        new_sample_pos[tree_idx] = end_node_id
        tree_idx += 1

    return [new_sample_pos]


def predict_guest(ctx: Context, trees: List[DecisionTree], data: DataFrame):
    
    tree_list = [tree.get_nodes() for tree in trees]
    max_node_num = max([len(tree) for tree in tree_list])
    map_func = functools.partial(generate_pos_array, tree_num=len(trees), max_node_num=max_node_num)

    sample_pos = data.create_frame()
    sample_pos['sample_pos'] = data.apply_row(lambda x: map_func())
    result_sample_pos = sample_pos.empty_frame()

    sample_with_pos = DataFrame.hstack([data, sample_pos])
    sitename = ctx.local.party[0] + '_' + ctx.local.party[1]

    # start loop here
    comm_round = 0
    sub_ctx = ctx.sub_ctx('predict_round').indexed_ctx(comm_round)
    map_func = functools.partial(traverse_tree, trees=tree_list, sitename=sitename)
    new_pos = sample_with_pos.create_frame()
    new_pos['sample_pos'] = sample_with_pos.apply_row(map_func)

    done_sample_idx = new_pos.apply_row(lambda x: all_reach_leaf(x['sample_pos'][0]))  # samples that reach leaf node in all trees
    not_finished_sample_idx = new_pos.apply_row(lambda x: not_finished(x['sample_pos'][0]))  # samples that not reach leaf node in all trees

    indexer = done_sample_idx.get_indexer('sample_id')
    done_sample = new_pos.loc(indexer, preserve_order=True)[done_sample_idx.as_tensor()]
    indexer = not_finished_sample_idx.get_indexer('sample_id')
    pending_samples = new_pos.loc(indexer, preserve_order=True)[not_finished_sample_idx.as_tensor()]

    # send not-finished samples to host
    sub_ctx.hosts.put('pending_samples', (pending_samples.as_tensor(), pending_samples.get_indexer(target='sample_id')))
    # get result from host and merge

    result_sample_pos = DataFrame.vstack([result_sample_pos, done_sample])

    return sample_with_pos, new_pos, result_sample_pos, pending_samples


def predict_host(ctx: Context, trees: List[DecisionTree], data: DataFrame):
    
    tree_list = [tree.get_nodes() for tree in trees]

    sample_pos = data.create_frame()
    # help guest to traverse tree
    comm_round = 0

    # start loop here
    sub_ctx = ctx.sub_ctx('predict_round').indexed_ctx(comm_round)
    
    pending_samples_data, pending_samples_indexer = sub_ctx.guest.get('pending_samples')
    pending_samples = sample_pos.loc(pending_samples_indexer, preserve_order=True)
    pending_samples['sample_pos'] = pending_samples_data

    return pending_samples