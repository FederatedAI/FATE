import functools
import numpy as np
import random
from typing import List
from federatedml.util import consts
from federatedml.secureprotol import PaillierEncrypt
from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeGuest, HeteroDecisionTreeHost, \
    HeteroFastDecisionTreeGuest, HeteroFastDecisionTreeHost
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.decision_tree import DecisionTree, Node
from federatedml.util import LOGGER
from federatedml.transfer_variable.transfer_class.hetero_secure_boosting_predict_transfer_variable import \
    HeteroSecureBoostTransferVariable

"""
Hetero guest predict utils
"""


def generate_leaf_pos_dict(x, tree_num, np_int_type=np.int8):
    """
    x: just occupy the first parameter position
    return: a numpy array record sample pos, and a counter counting how many trees reach a leaf node
    """

    node_pos = np.zeros(tree_num, dtype=np_int_type)
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


def get_predict_scores(
        leaf_pos,
        learning_rate,
        init_score,
        trees: List[HeteroDecisionTreeGuest],
        multi_class_num=-1,
        predict_cache=None):
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


def get_dtype(max_int):
    if max_int < (2**8) / 2:
        return np.int8
    elif max_int < (2**16) / 2:
        return np.int16
    else:
        return np.int64


def sbt_guest_predict(data_inst, transfer_var: HeteroSecureBoostTransferVariable,
                      trees: List[HeteroDecisionTreeGuest], learning_rate, init_score, booster_dim,
                      predict_cache=None, pred_leaf=False):
    tree_num = len(trees)
    max_depth = trees[0].max_depth
    max_int = 2 ** max_depth
    dtype = get_dtype(max_int)
    LOGGER.debug('chosen np dtype is {}'.format(dtype))
    generate_func = functools.partial(generate_leaf_pos_dict, tree_num=tree_num, np_int_type=dtype)
    node_pos_tb = data_inst.mapValues(generate_func)  # record node pos
    final_leaf_pos = data_inst.mapValues(lambda x: np.zeros(tree_num, dtype=dtype) + np.nan)  # record final leaf pos
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

    if pred_leaf:  # return leaf position only
        return final_leaf_pos

    else:  # get final predict scores from leaf pos
        predict_result = get_predict_scores(leaf_pos=final_leaf_pos, learning_rate=learning_rate,
                                            init_score=init_score, trees=trees,
                                            multi_class_num=booster_dim, predict_cache=predict_cache)
        LOGGER.debug('predict result 2 is {}'.format(list(predict_result.collect())))
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


def host_traverse_trees(sample, leaf_pos, trees: List[HeteroDecisionTreeHost]):
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
        host_node_pos = data_inst.join(guest_node_pos, traverse_func)
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


"""
Fed-EINI predict func
"""


def get_leaf_idx_map(trees):
    id_pos_map_list = []

    for tree in trees:
        array_idx = 0
        id_pos_map = {}
        for node in tree.tree_node:
            if node.is_leaf:
                id_pos_map[node.id] = array_idx
                array_idx += 1
        id_pos_map_list.append(id_pos_map)

    return id_pos_map_list


def go_to_children_branches(data_inst, tree_node, tree, sitename: str, candidate_list: List):
    if tree_node.is_leaf:
        candidate_list.append(tree_node)
    else:
        tree_node_list = tree.tree_node
        if tree_node.sitename != sitename:
            go_to_children_branches(data_inst, tree_node_list[tree_node.left_nodeid],
                                    tree, sitename, candidate_list)
            go_to_children_branches(data_inst, tree_node_list[tree_node.right_nodeid],
                                    tree, sitename, candidate_list)
        else:
            next_layer_node_id = tree.go_next_layer(tree_node, data_inst, use_missing=tree.use_missing,
                                                    zero_as_missing=tree.zero_as_missing, decoder=tree.decode,
                                                    split_maskdict=tree.split_maskdict,
                                                    missing_dir_maskdict=tree.missing_dir_maskdict,
                                                    bin_sparse_point=None
                                                    )
            go_to_children_branches(data_inst, tree_node_list[next_layer_node_id], tree, sitename, candidate_list)


def generate_leaf_candidates_guest(data_inst, sitename, trees, node_pos_map_list,
                                   init_score, learning_rate, booster_dim):
    candidate_nodes_of_all_tree = []

    if booster_dim > 2:
        epoch_num = len(trees) // booster_dim
    else:
        epoch_num = len(trees)
    init_score = init_score / epoch_num
    score_idx = 0

    for tree, node_pos_map in zip(trees, node_pos_map_list):
        if booster_dim > 2:
            tree_init_score = init_score[score_idx]
            score_idx += 1
            if score_idx == booster_dim:
                score_idx = 0
        else:
            tree_init_score = init_score
        candidate_list = []
        go_to_children_branches(data_inst, tree.tree_node[0], tree, sitename, candidate_list)

        # check if it is mo tree:
        if len(candidate_list) < 1:
            raise ValueError('incorrect candidate list length,: {}'.format(len(candidate_list)))
        node = candidate_list[0]

        result_vec = np.zeros(len(node_pos_map))
        if isinstance(node.weight, np.ndarray):
            if len(node.weight) > 1:
                result_vec = [np.array([0 for i in range(len(node.weight))]) for i in range(len(node_pos_map))]

        for node in candidate_list:
            result_vec[node_pos_map[node.id]] = node.weight * learning_rate + tree_init_score

        candidate_nodes_of_all_tree.extend(result_vec)

    return np.array(candidate_nodes_of_all_tree)


def EINI_guest_predict(data_inst, trees: List[HeteroDecisionTreeGuest], learning_rate, init_score, booster_dim,
                       encrypt_key_length, transfer_var: HeteroSecureBoostTransferVariable,
                       sitename=None, party_list=None, predict_cache=None, pred_leaf=False):

    if sitename is None:
        raise ValueError('input sitename is None, not able to run EINI predict algorithm')

    if pred_leaf:
        raise ValueError('EINI predict mode does not support leaf idx prediction')

    # EINI algorithms
    id_pos_map_list = get_leaf_idx_map(trees)
    map_func = functools.partial(generate_leaf_candidates_guest, sitename=sitename, trees=trees,
                                 node_pos_map_list=id_pos_map_list, init_score=init_score,
                                 learning_rate=learning_rate, booster_dim=booster_dim)
    position_vec = data_inst.mapValues(map_func)

    # encryption
    encrypter = PaillierEncrypt()
    encrypter.generate_key(encrypt_key_length)
    encrypter_vec_table = position_vec.mapValues(encrypter.recursive_encrypt)

    # federation part
    # send to first host party
    transfer_var.guest_predict_data.remote(encrypter_vec_table, idx=0, suffix='position_vec', role=consts.HOST)

    # get from last host party
    result_table = transfer_var.host_predict_data.get(idx=len(party_list) - 1, suffix='merge_result', role=consts.HOST)

    # decode result
    result = result_table.mapValues(encrypter.recursive_decrypt)
    # reformat
    result = result.mapValues(lambda x: np.array(x))
    if predict_cache:
        result = result.join(predict_cache, lambda v1, v2: v1 + v2)

    return result


def generate_leaf_candidates_host(data_inst, sitename, trees, node_pos_map_list):
    candidate_nodes_of_all_tree = []

    for tree, node_pos_map in zip(trees, node_pos_map_list):

        result_vec = [0 for i in range(len(node_pos_map))]
        candidate_list = []
        go_to_children_branches(data_inst, tree.tree_node[0], tree, sitename, candidate_list)

        for node in candidate_list:
            result_vec[node_pos_map[node.id]] = 1  # create 0-1 vector
        candidate_nodes_of_all_tree.extend(result_vec)

    return np.array(candidate_nodes_of_all_tree)


def generate_leaf_idx_dimension_map(trees, booster_dim):
    cur_dim = 0
    leaf_dim_map = {}
    leaf_idx = 0
    for tree in trees:
        for node in tree.tree_node:
            if node.is_leaf:
                leaf_dim_map[leaf_idx] = cur_dim
                leaf_idx += 1
        cur_dim += 1
        if cur_dim == booster_dim:
            cur_dim = 0
    return leaf_dim_map


def merge_position_vec(host_vec, guest_encrypt_vec, booster_dim=1, leaf_idx_dim_map=None, random_mask=None):

    leaf_idx = -1
    rs = [0 for i in range(booster_dim)]
    for en_num, vec_value in zip(guest_encrypt_vec, host_vec):
        leaf_idx += 1
        if vec_value == 0:
            continue
        else:
            dim = leaf_idx_dim_map[leaf_idx]
            rs[dim] += en_num

    if random_mask:
        for i in range(len(rs)):
            rs[i] = rs[i] * random_mask  # a pos random mask btw 1 and 2

    return rs


def position_vec_element_wise_mul(guest_encrypt_vec, host_vec):
    new_vec = []
    for en_num, vec_value in zip(guest_encrypt_vec, host_vec):
        new_vec.append(en_num * vec_value)
    return new_vec


def count_complexity_helper(node, node_list, host_sitename, meet_host_node):

    if node.is_leaf:
        return 1 if meet_host_node else 0
    if node.sitename == host_sitename:
        meet_host_node = True

    return count_complexity_helper(node_list[node.left_nodeid], node_list, host_sitename, meet_host_node) + \
        count_complexity_helper(node_list[node.right_nodeid], node_list, host_sitename, meet_host_node)


def count_complexity(trees, sitename):

    tree_valid_leaves_num = []
    for tree in trees:
        valid_leaf_num = count_complexity_helper(tree.tree_node[0], tree.tree_node, sitename, False)
        if valid_leaf_num != 0:
            tree_valid_leaves_num.append(valid_leaf_num)

    complexity = 1
    for num in tree_valid_leaves_num:
        complexity *= num

    return complexity


def EINI_host_predict(data_inst, trees: List[HeteroDecisionTreeHost], sitename, self_party_id, party_list,
                      booster_dim, transfer_var: HeteroSecureBoostTransferVariable,
                      complexity_check=False, random_mask=False):

    if complexity_check:
        complexity = count_complexity(trees, sitename)
        LOGGER.debug('checking EINI complexity: {}'.format(complexity))
        if complexity < consts.EINI_TREE_COMPLEXITY:
            raise ValueError('tree complexity: {}, is lower than safe '
                             'threshold, inference is not allowed.'.format(complexity))
    id_pos_map_list = get_leaf_idx_map(trees)
    map_func = functools.partial(generate_leaf_candidates_host, sitename=sitename, trees=trees,
                                 node_pos_map_list=id_pos_map_list)
    position_vec = data_inst.mapValues(map_func)
    booster_dim = booster_dim
    random_mask = random.SystemRandom().random() + 1 if random_mask else 0  # generate a random mask btw 1 and 2

    self_idx = party_list.index(self_party_id)
    if len(party_list) == 1:
        guest_position_vec = transfer_var.guest_predict_data.get(idx=0, suffix='position_vec')
        leaf_idx_dim_map = generate_leaf_idx_dimension_map(trees, booster_dim)
        merge_func = functools.partial(merge_position_vec, booster_dim=booster_dim,
                                       leaf_idx_dim_map=leaf_idx_dim_map, random_mask=random_mask)
        result_table = position_vec.join(guest_position_vec, merge_func)
        transfer_var.host_predict_data.remote(result_table, suffix='merge_result')
    else:
        # multi host case
        # if is first host party, get encrypt vec from guest, else from previous host party
        if self_party_id == party_list[0]:
            guest_position_vec = transfer_var.guest_predict_data.get(idx=0, suffix='position_vec')

        else:
            guest_position_vec = transfer_var.inter_host_data.get(idx=self_idx - 1, suffix='position_vec')

        if self_party_id == party_list[-1]:
            leaf_idx_dim_map = generate_leaf_idx_dimension_map(trees, booster_dim)
            func = functools.partial(merge_position_vec, booster_dim=booster_dim,
                                     leaf_idx_dim_map=leaf_idx_dim_map, random_mask=random_mask)
            result_table = position_vec.join(guest_position_vec, func)
            transfer_var.host_predict_data.remote(result_table, suffix='merge_result')
        else:
            result_table = position_vec.join(guest_position_vec, position_vec_element_wise_mul)
            transfer_var.inter_host_data.remote(result_table, idx=self_idx + 1, suffix='position_vec', role=consts.HOST)
