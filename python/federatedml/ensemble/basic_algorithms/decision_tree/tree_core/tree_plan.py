from typing import Tuple, List
from federatedml.util import LOGGER
from federatedml.util import consts

tree_type_dict = {
    'guest_feat_only': 0,  # use only guest feature to build this tree
    'host_feat_only': 1,  # use only host feature to build this tree
    'normal_tree': 2,  # a normal decision tree
    'layered_tree': 3  # a layered decision tree
}

tree_actions = {
    'guest_only': 0,  # use only guest feature to build this layer
    'host_only': 1,  # use only host feature to build this layer
    'guest_and_host': 2,  # use global feature to build this layer
}


def create_tree_plan(work_mode: str, k=1, tree_num=10, host_list=None, complete_secure=True):
    """
    Args:
        work_mode:
        k: k is needed when work_mode is 'layered'
        tree_num: decision tree number
        host_list: need to specify host idx when under multi-host scenario, default is None
        complete_secure:
    Returns: tree plan: (work mode, host id) host id -1 is default value
    """

    LOGGER.info('boosting_core trees work mode is {}'.format(work_mode))
    tree_plan = []

    if work_mode == consts.MIX_TREE:
        assert k > 0
        assert len(host_list) > 0

        one_round = [(tree_type_dict['guest_feat_only'], -1)] * k
        for host_idx, host_id in enumerate(host_list):
            one_round += [(tree_type_dict['host_feat_only'], host_id)] * k

        round_num = (tree_num // (2 * k)) + 1
        tree_plan = (one_round * round_num)[0:tree_num]

    elif work_mode == consts.LAYERED_TREE:
        tree_plan = [(tree_type_dict['layered_tree'], -1) for i in range(tree_num)]
        if complete_secure:
            tree_plan[0] = (tree_type_dict['guest_feat_only'], -1)

    return tree_plan


def create_node_plan(tree_type, target_host_id, max_depth) -> List[Tuple[int, int]]:
    LOGGER.debug('cur tree working mode is {}'.format((tree_type, target_host_id)))
    node_plan = []

    if tree_type == tree_type_dict['guest_feat_only']:
        node_plan = [(tree_actions['guest_only'], target_host_id) for i in range(max_depth)]

    elif tree_type == tree_type_dict['host_feat_only']:
        node_plan = [(tree_actions['host_only'], target_host_id) for i in range(max_depth)]

    return node_plan


def create_layered_tree_node_plan(guest_depth=0, host_depth=0, host_list=None):
    assert guest_depth > 0 and host_depth > 0
    assert len(host_list) > 0

    one_round = []
    for host_idx, host_id in enumerate(host_list):
        one_round += [(tree_type_dict['host_feat_only'], host_id)] * host_depth
    one_round += [(tree_type_dict['guest_feat_only'], -1)] * guest_depth

    return one_round


def encode_plan(p, split_token='_'):
    result = []
    for tree_type_or_action, host_id in p:
        result.append(str(tree_type_or_action) + split_token + str(host_id))
    return result


def decode_plan(s, split_token='_'):
    result = []
    for string in s:
        t = string.split(split_token)
        result.append((int(t[0]), int(t[1])))

    return result
