import functools
import numpy as np
from typing import List
from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeGuest, HeteroDecisionTreeHost, \
    HeteroFastDecisionTreeGuest, HeteroFastDecisionTreeHost
from federatedml.util import LOGGER
from federatedml.transfer_variable.transfer_class.hetero_secure_boosting_predict_transfer_variable import \
    HeteroSecureBoostTransferVariable
from federatedml.util import consts

"""
Hetero guest predict utils
"""


def generate_leaf_pos_dict(x, tree_num):
    """
    x: just occupy the first parameter position
    return: a numpy array record sample pos, and a counter counting how many trees reach a leaf node
    """
    node_pos = np.zeros(tree_num, dtype=np.int64) + 0
    reach_leaf_node = np.zeros(tree_num, dtype=np.bool)
    return {'node_pos': node_pos, 'reach_leaf_node': reach_leaf_node}


def guest_traverse_a_tree(tree: HeteroDecisionTreeGuest, sample, cur_node_idx):
    reach_leaf = False
    # only need nid here, predict state is not needed
    rs = tree.traverse_tree(tree_=tree.tree_node, data_inst=sample, predict_state=(cur_node_idx, -1),
                            decoder=tree.decode, sitename=tree.sitename, use_missing=tree.use_missing,
                            split_maskdict=tree.split_maskdict, missing_dir_maskdict=tree.missing_dir_maskdict,
                            zero_as_missing=tree.zero_as_missing, return_leaf_id=True)

    if not isinstance(rs, tuple):
        reach_leaf = True
        leaf_id = rs
        return leaf_id, reach_leaf
    else:
        cur_node_idx = rs[0]
        return cur_node_idx, reach_leaf


def guest_traverse_trees(node_pos, sample, trees: List[HeteroDecisionTreeGuest]):
    if node_pos['reach_leaf_node'].all():
        return node_pos

    for t_idx, tree in enumerate(trees):

        cur_node_idx = node_pos['node_pos'][t_idx]

        # reach leaf
        if cur_node_idx == -1:
            continue

        rs, reach_leaf = guest_traverse_a_tree(tree, sample, cur_node_idx)

        if reach_leaf:
            node_pos['reach_leaf_node'][t_idx] = True

        node_pos['node_pos'][t_idx] = rs

    return node_pos


def merge_predict_pos(node_pos1, node_pos2):
    pos_arr1 = node_pos1['node_pos']
    pos_arr2 = node_pos2['node_pos']
    stack_arr = np.stack([pos_arr1, pos_arr2])
    node_pos1['node_pos'] = np.max(stack_arr, axis=0)
    return node_pos1


def add_y_hat(leaf_pos, init_score, learning_rate, trees: List[HeteroDecisionTreeGuest], multi_class_num=None):
    # finally node pos will hold weights
    weights = []
    for leaf_idx, tree in zip(leaf_pos, trees):
        weights.append(tree.tree_node[int(leaf_idx)].weight)
    weights = np.array(weights)
    if multi_class_num > 2:
        weights = weights.reshape((-1, multi_class_num))
    return np.sum(weights * learning_rate, axis=0) + init_score


def get_predict_scores(leaf_pos, learning_rate, init_score, trees: List[HeteroDecisionTreeGuest]
                       , multi_class_num=-1, predict_cache=None):
    if predict_cache:
        init_score = 0  # prevent init_score re-add

    predict_func = functools.partial(add_y_hat,
                                     learning_rate=learning_rate, init_score=init_score, trees=trees,
                                     multi_class_num=multi_class_num)
    predict_result = leaf_pos.mapValues(predict_func)

    if predict_cache:
        predict_result = predict_result.join(predict_cache, lambda v1, v2: v1 + v2)

    return predict_result


def save_leaf_pos_helper(v1, v2):
    reach_leaf_idx = v2['reach_leaf_node']
    select_idx = reach_leaf_idx & (v2['node_pos'] != -1)  # reach leaf and are not recorded( if recorded idx is -1)
    v1[select_idx] = v2['node_pos'][select_idx]
    return v1


def mask_leaf_pos(v):
    reach_leaf_idx = v['reach_leaf_node']
    v['node_pos'][reach_leaf_idx] = -1
    return v


def save_leaf_pos_and_mask_leaf_pos(node_pos_tb, final_leaf_pos):
    # save leaf pos
    saved_leaf_pos = final_leaf_pos.join(node_pos_tb, save_leaf_pos_helper)
    rest_part = final_leaf_pos.subtractByKey(saved_leaf_pos)
    final_leaf_pos = saved_leaf_pos.union(rest_part)
    # mask leaf pos
    node_pos_tb = node_pos_tb.mapValues(mask_leaf_pos)

    return node_pos_tb, final_leaf_pos


def merge_leaf_pos(pos1, pos2):
    return pos1 + pos2


def traverse_guest_local_trees(node_pos, sample, trees: List[HeteroFastDecisionTreeGuest]):
    """
    in mix mode, a sample can reach leaf directly
    """

    for t_idx, tree in enumerate(trees):

        cur_node_idx = node_pos[t_idx]

        if not tree.use_guest_feat_only_predict_mode:
            continue

        rs, reach_leaf = guest_traverse_a_tree(tree, sample, cur_node_idx)
        node_pos[t_idx] = rs

    return node_pos


"""
Hetero guest predict function
"""


def sbt_guest_predict(data_inst, transfer_var: HeteroSecureBoostTransferVariable,
                      trees: List[HeteroDecisionTreeGuest], learning_rate, init_score, booster_dim,
                      predict_cache=None, pred_leaf=False):
    tree_num = len(trees)
    generate_func = functools.partial(generate_leaf_pos_dict, tree_num=tree_num)
    node_pos_tb = data_inst.mapValues(generate_func)  # record node pos
    final_leaf_pos = data_inst.mapValues(lambda x: np.zeros(tree_num, dtype=np.int64) + np.nan)  # record final leaf pos
    traverse_func = functools.partial(guest_traverse_trees, trees=trees)
    comm_round = 0

    while True:

        # LOGGER.info('cur predict round is {}'.format(comm_round))
        node_pos_tb = node_pos_tb.join(data_inst, traverse_func)
        node_pos_tb, final_leaf_pos = save_leaf_pos_and_mask_leaf_pos(node_pos_tb, final_leaf_pos)

        # remove sample that reaches leaves of all trees
        reach_leaf_samples = node_pos_tb.filter(lambda key, value: value['reach_leaf_node'].all())
        node_pos_tb = node_pos_tb.subtractByKey(reach_leaf_samples)

        if node_pos_tb.count() == 0:
            transfer_var.predict_stop_flag.remote(True, idx=-1, suffix=(comm_round,))
            break

        transfer_var.predict_stop_flag.remote(False, idx=-1, suffix=(comm_round,))
        transfer_var.guest_predict_data.remote(node_pos_tb, idx=-1, suffix=(comm_round,))

        host_pos_tbs = transfer_var.host_predict_data.get(idx=-1, suffix=(comm_round,))

        for host_pos_tb in host_pos_tbs:
            node_pos_tb = node_pos_tb.join(host_pos_tb, merge_predict_pos)

        comm_round += 1

    # LOGGER.info('federated prediction process done')

    if pred_leaf:  # return leaf position only
        return final_leaf_pos

    else:  # get final predict scores from leaf pos
        predict_result = get_predict_scores(leaf_pos=final_leaf_pos, learning_rate=learning_rate,
                                            init_score=init_score, trees=trees,
                                            multi_class_num=booster_dim, predict_cache=predict_cache)
        return predict_result


def mix_sbt_guest_predict(data_inst, transfer_var: HeteroSecureBoostTransferVariable,
                          trees: List[HeteroDecisionTreeGuest], learning_rate, init_score, booster_dim,
                          predict_cache=None, pred_leaf=False):
    LOGGER.info('running mix mode predict')

    tree_num = len(trees)
    node_pos = data_inst.mapValues(lambda x: np.zeros(tree_num, dtype=np.int64))

    # traverse local trees
    traverse_func = functools.partial(traverse_guest_local_trees, trees=trees)
    guest_leaf_pos = node_pos.join(data_inst, traverse_func)

    # get leaf node from other host parties
    host_leaf_pos_list = transfer_var.host_predict_data.get(idx=-1)

    for host_leaf_pos in host_leaf_pos_list:
        guest_leaf_pos = guest_leaf_pos.join(host_leaf_pos, merge_leaf_pos)

    if pred_leaf:  # predict leaf, return leaf position only
        return guest_leaf_pos
    else:
        predict_result = get_predict_scores(leaf_pos=guest_leaf_pos, learning_rate=learning_rate,
                                            init_score=init_score, trees=trees,
                                            multi_class_num=booster_dim, predict_cache=predict_cache)
        return predict_result


"""
Hetero host predict utils
"""


def host_traverse_a_tree(tree: HeteroDecisionTreeHost, sample, cur_node_idx):
    nid, _ = tree.traverse_tree(predict_state=(cur_node_idx, -1), data_inst=sample,
                                decoder=tree.decode, split_maskdict=tree.split_maskdict,
                                missing_dir_maskdict=tree.missing_dir_maskdict, sitename=tree.sitename,
                                tree_=tree.tree_node, zero_as_missing=tree.zero_as_missing,
                                use_missing=tree.use_missing)

    return nid, _


def host_traverse_trees(leaf_pos, sample, trees: List[HeteroDecisionTreeHost]):
    for t_idx, tree in enumerate(trees):

        cur_node_idx = leaf_pos['node_pos'][t_idx]
        # idx is set as -1 when a sample reaches leaf
        if cur_node_idx == -1:
            continue
        nid, _ = host_traverse_a_tree(tree, sample, cur_node_idx)
        leaf_pos['node_pos'][t_idx] = nid

    return leaf_pos


def traverse_host_local_trees(node_pos, sample, trees: List[HeteroFastDecisionTreeHost]):
    """
    in mix mode, a sample can reach leaf directly
    """

    for i in range(len(trees)):

        tree = trees[i]
        if len(tree.tree_node) == 0:  # this tree belongs to other party because it has no tree node
            continue
        leaf_id = tree.host_local_traverse_tree(sample, tree.tree_node, use_missing=tree.use_missing,
                                                zero_as_missing=tree.zero_as_missing)
        node_pos[i] = leaf_id

    return node_pos


"""
Hetero host predict function
"""


def sbt_host_predict(data_inst, transfer_var: HeteroSecureBoostTransferVariable, trees: List[HeteroDecisionTreeHost]):
    comm_round = 0

    traverse_func = functools.partial(host_traverse_trees, trees=trees)

    while True:

        LOGGER.debug('cur predict round is {}'.format(comm_round))

        stop_flag = transfer_var.predict_stop_flag.get(idx=0, suffix=(comm_round,))
        if stop_flag:
            break

        guest_node_pos = transfer_var.guest_predict_data.get(idx=0, suffix=(comm_round,))
        host_node_pos = guest_node_pos.join(data_inst, traverse_func)
        if guest_node_pos.count() != host_node_pos.count():
            raise ValueError('sample count mismatch: guest table {}, host table {}'.format(guest_node_pos.count(),
                                                                                           host_node_pos.count()))
        transfer_var.host_predict_data.remote(host_node_pos, idx=-1, suffix=(comm_round,))

        comm_round += 1


def mix_sbt_host_predict(data_inst, transfer_var: HeteroSecureBoostTransferVariable,
                         trees: List[HeteroDecisionTreeHost]):
    LOGGER.info('running mix mode predict')

    tree_num = len(trees)
    node_pos = data_inst.mapValues(lambda x: np.zeros(tree_num, dtype=np.int64))
    local_traverse_func = functools.partial(traverse_host_local_trees, trees=trees)
    leaf_pos = node_pos.join(data_inst, local_traverse_func)
    transfer_var.host_predict_data.remote(leaf_pos, idx=0, role=consts.GUEST)
