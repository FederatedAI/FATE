#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import pandas as pd
import numpy as np
from typing import List
from fate.arch import Context
from fate.arch.dataframe import DataFrame
import copy
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import DecisionTree, _make_decision, Node
import functools
from logging import getLogger

logger = getLogger(__name__)


def get_dtype(max_int):
    if max_int < (2**8) / 2:
        return np.int8
    elif max_int < (2**16) / 2:
        return np.int16
    else:
        return np.int64


def all_reach_leaf(pos: np.array):
    if isinstance(pos, list):
        pos = np.array(pos)
    return np.all(pos < 0)


def not_finished(pos: np.array):
    if isinstance(pos, list):
        pos = np.array(pos)
    return not np.all(pos < 0)


def generate_pos_array(tree_num, max_node_num):
    dtype = get_dtype(max_node_num)
    # return list as a column of dataframe
    return [np.zeros(tree_num, dtype=dtype)]


def go_deep(s: pd.Series, tree: List[Node], sitename, cur_node_id, tree_idx=None):
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
    sample_pos = s["sample_pos"]
    new_sample_pos = np.copy(sample_pos)  # deepcopy to avoid inplace modification, for spark

    tree_idx = 0
    for node_pos, tree in zip(sample_pos, trees):
        if node_pos < 0:  # sample already reaches leaf node in this tree
            tree_idx += 1
            continue

        cur_node_id = node_pos
        end_node_id = go_deep(s, tree, sitename, cur_node_id, tree_idx=tree_idx)
        new_sample_pos[tree_idx] = end_node_id
        tree_idx += 1

    return [new_sample_pos]


def _merge_pos_arr(s: pd.Series):
    arr_1 = s["sample_pos"]
    arr_2 = s["host_sample_pos"]
    arr_1 = np.array(arr_1)
    arr_2 = np.array(arr_2)
    assert len(arr_1) == len(arr_2)
    merge_rs = np.copy(arr_1)
    already_on_leaf = arr_1 < 0
    on_leaf = arr_2 < 0
    updated = ~already_on_leaf & (on_leaf | (arr_2 > arr_1))
    merge_rs[updated] = arr_2[updated]
    return [merge_rs]


def _merge_pos(guest_pos: DataFrame, host_pos: List[DataFrame]):
    for host_df in host_pos:
        # assert alignment
        indexer = guest_pos.get_indexer(target="sample_id")
        host_df = host_df.loc(indexer=indexer, preserve_order=True)
        stack_df = DataFrame.hstack([guest_pos, host_df])
        guest_pos["sample_pos"] = stack_df.apply_row(_merge_pos_arr)

    return guest_pos


def predict_leaf_guest(ctx: Context, trees: List[DecisionTree], data: DataFrame):
    predict_data = data
    tree_list = [tree.get_nodes() for tree in trees]
    max_node_num = max([len(tree) for tree in tree_list])
    map_func = functools.partial(generate_pos_array, tree_num=len(trees), max_node_num=max_node_num)

    sample_pos = data.create_frame()
    sample_pos["sample_pos"] = data.apply_row(lambda x: map_func())
    result_sample_pos = sample_pos.empty_frame()

    sitename = ctx.local.name

    # start loop here
    comm_round = 0

    while True:
        sub_ctx = ctx.sub_ctx("predict_round").indexed_ctx(comm_round)

        if comm_round:
            predict_data = predict_data.loc(indexer=sample_pos.get_indexer(target="sample_id"), preserve_order=True)

        sample_with_pos = DataFrame.hstack([predict_data, sample_pos])
        logger.info("predict round {} has {} samples to predict".format(comm_round, len(sample_with_pos)))
        map_func = functools.partial(traverse_tree, trees=tree_list, sitename=sitename)
        new_pos = sample_with_pos.create_frame()
        new_pos["sample_pos"] = sample_with_pos.apply_row(map_func)
        done_sample_idx = new_pos.apply_row(
            lambda x: all_reach_leaf(x["sample_pos"])
        )  # samples that reach leaf node in all trees
        # not_finished_sample_idx = ~done_sample_idx
        not_finished_sample_idx = new_pos.apply_row(
            lambda x: not_finished(x["sample_pos"])
        )  # samples that not reach leaf node in all trees

        done_sample = new_pos.iloc(done_sample_idx)
        result_sample_pos = DataFrame.vstack([result_sample_pos, done_sample])
        if len(result_sample_pos) == len(data):
            sub_ctx.hosts.put("need_stop", True)
            break

        sub_ctx.hosts.put("need_stop", False)
        pending_samples = new_pos.iloc(not_finished_sample_idx)

        # send not-finished samples to host
        sub_ctx.hosts.put("pending_samples", (pending_samples))
        # get result from host and merge
        updated_pos = sub_ctx.hosts.get("updated_pos")
        sample_pos = _merge_pos(pending_samples, updated_pos)
        comm_round += 1

    logger.info("predict done")

    assert len(result_sample_pos) == len(data), "result sample pos length not equal to data length, {} vs {}".format(
        len(result_sample_pos), len(data)
    )
    return result_sample_pos


def predict_leaf_host(ctx: Context, trees: List[DecisionTree], data: DataFrame):
    tree_list = [tree.get_nodes() for tree in trees]
    sitename = ctx.local.name
    map_func = functools.partial(traverse_tree, trees=tree_list, sitename=sitename)

    # help guest to traverse tree
    comm_round = 0

    # start loop here
    while True:
        sub_ctx = ctx.sub_ctx("predict_round").indexed_ctx(comm_round)
        need_stop = sub_ctx.guest.get("need_stop")
        if need_stop:
            break
        pending_samples = sub_ctx.guest.get("pending_samples")
        logger.info("got {} pending samples".format(len(pending_samples)))
        sample_features = data.loc(pending_samples.get_indexer("sample_id"), preserve_order=True)
        sample_with_pos = DataFrame.hstack([sample_features, pending_samples])
        new_pos = sample_with_pos.create_frame()
        new_pos["host_sample_pos"] = sample_with_pos.apply_row(map_func)
        sub_ctx.guest.put("updated_pos", (new_pos))
        comm_round += 1

    logger.info("predict done")
