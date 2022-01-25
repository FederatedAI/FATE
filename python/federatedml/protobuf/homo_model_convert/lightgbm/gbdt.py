import numpy as np
import lightgbm as lgb
from ..component_converter import ComponentConverterBase
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam, \
    DecisionTreeModelParam, NodeParam
from federatedml.util import consts
from federatedml.util import LOGGER


"""
We only keep the necessary variable to make sure that lightgbm can run predict function on the converted model
"""

FAKE_FEATURE_INFO_STR = '[0:1] '

END_OF_TREE = 'end of trees'
END_OF_PARA = 'end of parameters'

SPLIT = '\n\n'

HEADER_TEMPLATE = """tree
version=v3
num_class={}
num_tree_per_iteration={}
label_index={}
max_feature_idx={}
objective={}
feature_names={}
feature_infos={}
"""

TREE_TEMPLATE = """Tree={}
num_leaves={}
num_cat={}
split_feature={}
threshold={}
decision_type={}
left_child={}
right_child={}
leaf_value={}
internal_value={}
shrinkage={}
"""

PARA_TEMPLATE = """parameters:
[boosting: gbdt]
[objective: {}]
[num_iterations: {}]
[learning_rate: {}]
[max_depth: {}]
[max_bin: {}]
[use_missing: {}]
[zero_as_missing: {}]
[num_class: {}]
[lambda_l1: {}]
[lambda_l2: {}]
[min_data_in_leaf: {}]
[min_gain_to_split: {}]
"""

LGB_OBJECTIVE = {
    consts.BINARY: "binary sigmoid:1",
    consts.REGRESSION: "regression",
    consts.MULTY: 'multiclass num_class:{}'
}

PARA_OBJECTIVE = {
    consts.BINARY: "binary",
    consts.REGRESSION: "regression",
    consts.MULTY: 'multiclass'
}


def get_decision_type(node: NodeParam, use_missing, zero_as_missing):

    # 00                     0                         0
    # Nan,0 or None         default left or right?    cat feature or not？
    default_type = 0  # 0000 None, default right, not cat feat

    if not use_missing:
        return default_type

    if node.missing_dir == -1:
        default_type = default_type | 2   # 0010

    if zero_as_missing:
        default_type = default_type | 4   # 0100 0
    else:
        default_type = default_type | 8  # 1000 np.Nan

    return default_type


def get_lgb_objective(task_type, num_classes, ret_dict, need_multi_format=True):

    if task_type == consts.CLASSIFICATION:
        if num_classes == 1:
            objective = ret_dict[consts.BINARY]
        else:
            objective = ret_dict[consts.MULTY].format(num_classes) if need_multi_format else ret_dict[consts.MULTY]
    else:
        objective = ret_dict[consts.REGRESSION]
    return objective


def list_to_str(l_):
    return str(l_).replace('[', '').replace(']', '').replace(',', '')


def parse_header(param: BoostingTreeModelParam, meta: BoostingTreeModelMeta):
    # generated header of lgb str model file
    # binary/regression num class is 1 in lgb
    num_classes = len(param.classes_) if len(param.classes_) > 2 else 1
    objective = get_lgb_objective(meta.task_type, num_classes, LGB_OBJECTIVE, need_multi_format=True)
    num_tree_per_iteration = param.tree_dim
    label_index = 0  # by default
    max_feature_idx = len(param.feature_name_fid_mapping) - 1
    feature_names = ''
    for name in [param.feature_name_fid_mapping[i] for i in range(max_feature_idx + 1)]:
        if ' ' in name:  # space is not allowed
            name = name.replace(' ', '-')
        feature_names += name + ' '
    feature_names = feature_names[:-1]
    feature_info = FAKE_FEATURE_INFO_STR * (max_feature_idx + 1)  # need to make fake feature info
    feature_info = feature_info[:-1]
    result_str = HEADER_TEMPLATE.format(num_classes, num_tree_per_iteration, label_index, max_feature_idx,
                                        objective, feature_names, feature_info)
    return result_str


def internal_count_computer(cur_id, tree_node, leaf_count, internal_count):

    if cur_id in leaf_count:
        return leaf_count[cur_id]

    left_count = internal_count_computer(tree_node[cur_id].left_nodeid, tree_node, leaf_count, internal_count)
    right_count = internal_count_computer(tree_node[cur_id].right_nodeid, tree_node, leaf_count, internal_count)
    internal_count[cur_id] = left_count + right_count
    return internal_count[cur_id]


def compute_internal_count(tree_param: DecisionTreeModelParam):

    root = tree_param.tree_[0]
    internal_count = {}
    leaf_count = tree_param.leaf_count
    root_count = internal_count_computer(root.id, tree_param.tree_, leaf_count, internal_count)
    if root.id not in internal_count:
        internal_count[root_count] = root_count
    return internal_count


def update_leaf_count(param):
    # in homo sbt, sometimes a leaf covers no sample, so need to add 1 to leaf count
    tmp = {}
    for i in param.leaf_count:
        tmp[i] = param.leaf_count[i]
    for i in tmp:
        if tmp[i] == 0:
            param.leaf_count[i] += 1


def parse_a_tree(
        param: DecisionTreeModelParam,
        tree_idx: int,
        use_missing=False,
        zero_as_missing=False,
        learning_rate=0.1,
        init_score=None):

    split_feature = []
    split_threshold = []
    decision_type = []
    internal_weight = []
    leaf_weight = []
    left, right = [], []
    leaf_idx = -1
    lgb_node_idx = 0
    sbt_lgb_node_map = {}
    is_leaf = []
    leaf_count = []
    internal_count, internal_count_dict = [], {}
    has_count_info = len(param.leaf_count) != 0

    # compute internal count
    if has_count_info:
        update_leaf_count(param)
        internal_count_dict = compute_internal_count(param)  # get internal count from leaf count

    # mark leaf nodes and get sbt-lgb node mapping
    for node in param.tree_:
        is_leaf.append(node.is_leaf)
        if not node.is_leaf:
            sbt_lgb_node_map[node.id] = lgb_node_idx
            lgb_node_idx += 1

    for cur_idx, node in enumerate(param.tree_):

        if not node.is_leaf:

            split_feature.append(node.fid)

            # if is hetero model need to decode split point and missing dir
            if param.split_maskdict and param.missing_dir_maskdict is not None:
                node.bid = param.split_maskdict[node.id]
                node.missing_dir = param.missing_dir_maskdict[node.id]

            # extract split point and weight
            split_threshold.append(node.bid)
            internal_weight.append(node.weight)

            # add internal count
            if has_count_info:
                internal_count.append(internal_count_dict[node.id])

            if is_leaf[node.left_nodeid]:  # generate lgb leaf idx
                left.append(leaf_idx)
                if has_count_info:
                    leaf_count.append(param.leaf_count[node.left_nodeid])
                leaf_idx -= 1
            else:
                left.append(sbt_lgb_node_map[node.left_nodeid])

            if is_leaf[node.right_nodeid]:  # generate lgb leaf idx
                right.append(leaf_idx)
                if has_count_info:
                    leaf_count.append(param.leaf_count[node.right_nodeid])
                leaf_idx -= 1
            else:
                right.append(sbt_lgb_node_map[node.right_nodeid])

            # get lgb decision type
            decision_type.append(get_decision_type(node, use_missing, zero_as_missing))
        else:
            # regression model need to add init score
            if init_score is not None:
                score = node.weight * learning_rate + init_score
            else:
                # leaf value is node.weight * learning_rate in lgb
                score = node.weight * learning_rate

            leaf_weight.append(score)

    leaves_num = len(leaf_weight)
    num_cat = 0
    # to string
    result_str = TREE_TEMPLATE.format(tree_idx, leaves_num, num_cat, list_to_str(split_feature),
                                      list_to_str(split_threshold), list_to_str(decision_type),
                                      list_to_str(left), list_to_str(right), list_to_str(leaf_weight),
                                      list_to_str(internal_weight), learning_rate)

    if len(internal_count) != 0:
        result_str += 'internal_count={}\n'.format(list_to_str(internal_count))
    if len(leaf_count) != 0:
        result_str += 'leaf_count={}\n'.format(list_to_str(leaf_count))

    return result_str


def parse_feature_importance(param):

    feat_importance_str = "feature_importances:\n"
    mapping = param.feature_name_fid_mapping
    for impt in param.feature_importances:
        impt_val = impt.importance
        try:
            if impt.main == 'split':
                impt_val = int(impt_val)
        except BaseException:
            LOGGER.warning("old version protobuf contains no filed 'main'")
        feat_importance_str += '{}={}\n'.format(mapping[impt.fid], impt_val)

    return feat_importance_str


def parse_parameter(param, meta):
    """
    we only keep parameters offered by SBT
    """
    tree_meta = meta.tree_meta
    num_classes = 1 if meta.task_type == consts.CLASSIFICATION and param.num_classes < 3 else param.num_classes
    objective = get_lgb_objective(meta.task_type, num_classes, PARA_OBJECTIVE, need_multi_format=False)
    rs = PARA_TEMPLATE.format(objective, meta.num_trees, meta.learning_rate, tree_meta.max_depth,
                              meta.quantile_meta.bin_num, meta.tree_meta.use_missing + 0,
                              meta.tree_meta.zero_as_missing + 0,
                              num_classes, tree_meta.criterion_meta.criterion_param[0],
                              tree_meta.criterion_meta.criterion_param[1],
                              tree_meta.min_leaf_node,
                              tree_meta.min_impurity_split
                              )
    return rs


def sbt_to_lgb(model_param: BoostingTreeModelParam,
               model_meta: BoostingTreeModelMeta,
               load_feature_importance=True):
    """
    Transform sbt model to lgb model
    """

    result = ''
    # parse header
    header_str = parse_header(model_param, model_meta)
    use_missing = model_meta.tree_meta.use_missing
    zero_as_missing = model_meta.tree_meta.zero_as_missing
    learning_rate = model_meta.learning_rate
    tree_str_list = []

    # parse tree
    for idx, param in enumerate(model_param.trees_):

        if idx == 0 and model_meta.task_type == consts.REGRESSION:  # regression task has init score
            init_score = model_param.init_score[0]
        else:
            init_score = 0
        tree_str_list.append(parse_a_tree(param, idx, use_missing, zero_as_missing, learning_rate, init_score))

    # add header and tree str to result
    result += header_str + '\n'
    for s in tree_str_list:
        result += s
        result += SPLIT
    result += END_OF_TREE

    # handle feature importance
    if load_feature_importance:
        feat_importance_str = parse_feature_importance(model_param)
        result += SPLIT + feat_importance_str

    # parameters
    para_str = parse_parameter(model_param, model_meta)
    result += '\n' + para_str + '\n' + END_OF_PARA + '\n'
    result += '\npandas_categorical:[]\n'

    return result


def save_lgb(model: lgb.Booster, path):
    model_str = model.model_to_string()
    f = open(path, 'w')
    f.write(model_str)
    f.close()


def load_lgb(path):
    f = open(path, 'r')
    model_str = f.read()
    f.close()
    lgb_model = lgb.Booster(model_str=model_str)
    return lgb_model


class HomoSBTComponentConverter(ComponentConverterBase):

    @staticmethod
    def get_target_modules():
        return ['HomoSecureboost']

    def convert(self, model_dict):

        param_obj = model_dict["HomoSecureBoostingTreeGuestParam"]
        meta_obj = model_dict["HomoSecureBoostingTreeGuestMeta"]

        lgb_model_str = sbt_to_lgb(param_obj, meta_obj)
        lgb_model = lgb.Booster(model_str=lgb_model_str)

        return lgb_model
