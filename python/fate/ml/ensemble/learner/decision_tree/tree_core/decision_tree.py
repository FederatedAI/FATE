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
#

# =============================================================================
# DecisionTree Base Class
# =============================================================================
import numpy as np
import pandas as pd
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import SplitInfo
from typing import List
import logging


FLOAT_ZERO = 1e-8
LEAF_IDX = -1


logger = logging.getLogger(__name__)


class FeatureImportance(object):
    def __init__(self, gain=0):
        self.gain = gain
        self.split = 1

    def add_gain(self, val):
        self.gain += val

    def add_split(self, val):
        self.split += val

    def __repr__(self):
        return "gain: {}, split {}".format(self.gain, self.split)

    def __add__(self, other):
        new_importance = FeatureImportance(gain=self.gain + other.gain)
        new_importance.split = self.split + other.split
        return new_importance

    def to_dict(self):
        return {"gain": self.gain, "split": self.split}

    @staticmethod
    def from_dict(dict_):
        importance = FeatureImportance()
        importance.gain = dict_["gain"]
        importance.split = dict_["split"]
        return importance


class Node(object):

    """
    Parameters:
        -----------
        nid : int, optional
            ID of the node.
        sitename : str, optional
            Name of the site that the node belongs to.
        fid : int, optional
            ID of the feature that the node splits on.
        bid : float or int, optional
            Feature value that the node splits on.
        weight : float, optional
            Weight of the node.
        is_leaf : bool, optional
            Boolean indicating whether the node is a leaf node.
        grad : float, optional
            Gradient value of the node.
        hess : float, optional
            Hessian value of the node.
        l : int, optional
            ID of the left child node.
        r : int, optional
            ID of the right child node.
        missing_dir : int, optional
            Direction for missing values (1 for left, -1 for right).
        sample_num : int, optional
            Number of samples in the node.
        is_left_node : bool, optional
            Boolean indicating whether the node is a left child node.
        sibling_nodeid : int, optional
            ID of the sibling node.
    """

    def __init__(
        self,
        nid=None,
        sitename=None,
        fid=None,
        bid=None,
        weight=0,
        is_leaf=False,
        grad=None,
        hess=None,
        l=-1,
        r=-1,
        missing_dir=1,
        sample_num=0,
        is_left_node=False,
        sibling_nodeid=None,
        parent_nodeid=None,
        split_id=None,
    ):
        self.nid = nid
        self.sitename = sitename
        self.fid = fid
        self.bid = bid
        self.weight = weight
        self.is_leaf = is_leaf
        self.grad = grad
        self.hess = hess
        self.l = l
        self.r = r
        self.missing_dir = missing_dir
        self.sample_num = sample_num
        self.is_left_node = is_left_node
        self.sibling_nodeid = sibling_nodeid
        self.parent_nodeid = parent_nodeid
        self.split_id = split_id

    def to_dict(self):
        return {
            "nid": self.nid,
            "sitename": self.sitename,
            "fid": self.fid,
            "bid": self.bid,
            "weight": self.weight,
            "is_leaf": self.is_leaf,
            "grad": self.grad,
            "hess": self.hess,
            "l": self.l,
            "r": self.r,
            "missing_dir": self.missing_dir,
            "sample_num": self.sample_num,
            "is_left_node": self.is_left_node,
            "sibling_nodeid": self.sibling_nodeid,
            "parent_nodeid": self.parent_nodeid,
            "split_id": self.split_id,
        }

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return "(node_id {}: fid {} bid {} left {}, right {}, pid {}, is_leaf {}, sample_count {},  g {}, h {}, weight {}, sitename {})".format(
            self.nid,
            self.fid,
            self.bid,
            self.l,
            self.r,
            self.parent_nodeid,
            self.is_leaf,
            self.sample_num,
            self.grad,
            self.hess,
            self.weight,
            self.sitename,
        )


def _make_decision(feat_val, bid, missing_dir=None, use_missing=None, zero_as_missing=None, zero_val=0):
    # no missing val
    left, right = True, False
    direction = left if feat_val <= bid + FLOAT_ZERO else right
    return direction


def _update_sample_pos(s: pd.Series, cur_layer_node: List[Node], node_map: dict, sitename=None):
    node_id = s.iloc[-1]
    node = cur_layer_node[node_map[node_id]]
    if node.is_leaf:
        return -(node.nid + 1)  # use negative index to represent leaves, + 1 to avoid root node 0
    feat_val = s[node.fid]
    bid = node.bid
    dir_ = _make_decision(feat_val, bid)
    if dir_:  # go left
        ret_node = node.l
    else:
        ret_node = node.r
    return ret_node


def _get_sample_on_local_nodes(s: pd.Series, cur_layer_node: List[Node], node_map: dict, sitename):
    node_id = s.iloc[-1]
    node = cur_layer_node[node_map[node_id]]
    on_local_node = node.sitename == sitename
    return on_local_node


def _update_sample_pos_on_local_nodes(s: pd.Series, cur_layer_node: List[Node], node_map: dict, sitename):
    on_local_node = _get_sample_on_local_nodes(s, cur_layer_node, node_map, sitename)
    if not on_local_node:
        return False, -1
    else:
        return True, _update_sample_pos(s, cur_layer_node, node_map, sitename)


def _merge_sample_pos(s: pd.Series):
    if s["g_on_local"]:
        return s["g_on_local"], s["g_node_idx"]
    else:
        return s["h_on_local"], s["h_node_idx"]


def _convert_sample_pos_to_score(s: pd.Series, tree_nodes: List[Node]):
    node_idx = s.iloc[0]
    if node_idx < 0:
        node_idx = -(node_idx + 1)
    target_node = tree_nodes[node_idx]
    if not target_node.is_leaf:
        raise ValueError("this sample is not on a leaf node")
    return target_node.weight


class DecisionTree(object):
    def __init__(self, max_depth=3, use_missing=False, zero_as_missing=False, valid_features=None):
        """
        Initialize a DecisionTree instance.

        Parameters:
        -----------
        max_depth : int
            The maximum depth of the tree.
        use_missing : bool, optional
            Whether or not to use missing values (default is False).
        zero_as_missing : bool, optional
            Whether to treat zero as a missing value (default is False).
        valid_features: list of boolean, optional
            Valid features for training, default is None, which means all features are valid.
        """
        self.max_depth = max_depth
        self.use_missing = use_missing
        self.zero_as_missing = zero_as_missing

        # runtime variables
        self._nodes = []
        self._cur_layer_node = []
        self._cur_leaf_idx = -1
        self._feature_importance = {}
        self._predict_weights = None
        self._g_tensor, self._h_tensor = None, None
        self._sample_pos = None
        self._leaf_node_map = {}
        self._valid_feature = valid_features
        self._sample_on_leaves = None
        self._sample_weights = None

    def _init_sample_pos(self, train_data: DataFrame):
        sample_pos = train_data.create_frame()
        sample_pos["node_idx"] = 0  # position of current sample
        return sample_pos

    def _init_leaves_sample_table(self, sample_pos: DataFrame):
        return sample_pos.empty_frame()

    def _get_leaf_node_map(self):
        if len(self._nodes) >= len(self._leaf_node_map):
            for n in self._nodes:
                self._leaf_node_map[n.nid] = n.is_leaf

    def _convert_sample_pos_to_weight(self, sample_pos: DataFrame, tree_nodes: List[Node]):
        import functools

        map_func = functools.partial(_convert_sample_pos_to_score, tree_nodes=tree_nodes)
        sample_weight = sample_pos.apply_row(map_func, columns=["score"])
        return sample_weight

    def _convert_bin_idx_to_split_val(self, ctx: Context, tree_nodes: List[Node], binning_dict: dict, schema):
        columns = schema.columns
        sitename = ctx.local.name
        for node in tree_nodes:
            if node.sitename == sitename:
                if not node.is_leaf:
                    feat_name = node.fid
                    split_val = binning_dict[feat_name][node.bid]
                    node.bid = split_val
            else:
                continue

        return tree_nodes

    def _initialize_root_node(self, ctx: Context, train_df: DataFrame, gh: DataFrame = None):
        sitename = ctx.local.name
        if gh is None:
            sum_g, sum_h = 0, 0
        else:
            sum_gh = gh.sum()
            sum_g = float(sum_gh["g"])
            sum_h = float(sum_gh["h"])
        root_node = Node(nid=0, grad=sum_g, hess=sum_h, sitename=sitename, sample_num=len(train_df))

        return root_node

    def _update_feature_importance(self, ctx: Context, split_info: List[SplitInfo], data: DataFrame):
        sitename = ctx.local.name
        for info in split_info:
            if info is not None and info.sitename == sitename:
                feat_name = self._fid_to_feature_name(info.best_fid, data)
                if feat_name not in self._feature_importance:
                    self._feature_importance[feat_name] = FeatureImportance(gain=info.gain)
                else:
                    self._feature_importance[feat_name] = self._feature_importance[feat_name] + FeatureImportance(
                        gain=info.gain
                    )

    def _fid_to_feature_name(self, fid: int, dataframe: DataFrame):
        if fid is None:
            return None
        return dataframe.schema.columns[fid]

    def _update_tree(self, ctx: Context, cur_layer_nodes: List[Node], split_info: List[SplitInfo], data: DataFrame):
        assert len(cur_layer_nodes) == len(
            split_info
        ), "node num not match split info num, got {} node vs {} split info".format(
            len(cur_layer_nodes), len(split_info)
        )

        next_layer_node = []

        for idx in range(len(split_info)):
            node: Node = cur_layer_nodes[idx]

            if split_info[idx] is None:
                node.is_leaf = True
                node.sitename = ctx.guest.name  # leaf always belongs to guest
                self._nodes.append(node)
                logger.info("set node {} to leaf".format(node))
                continue

            sum_grad = node.grad
            sum_hess = node.hess
            sum_cnt = node.sample_num

            feat_name = self._fid_to_feature_name(split_info[idx].best_fid, data)
            node.fid = feat_name
            node.bid = split_info[idx].best_bid
            node.missing_dir = split_info[idx].missing_dir
            node.sitename = split_info[idx].sitename
            node.split_id = split_info[idx].split_id  # if not a local node, has split id

            p_id = node.nid
            l_id, r_id = self._tree_node_num + 1, self._tree_node_num + 2
            self._tree_node_num += 2
            node.l, node.r = l_id, r_id

            l_g, l_h = split_info[idx].sum_grad, split_info[idx].sum_hess
            l_cnt = split_info[idx].sample_count

            # logger.info("splitting node {}, split info is {}".format(node, split_info[idx]))

            # create new left node and new right node
            left_node = Node(
                nid=l_id,
                grad=float(l_g),
                hess=float(l_h),
                weight=float(self.splitter.node_weight(l_g, l_h)),
                parent_nodeid=p_id,
                sibling_nodeid=r_id,
                is_left_node=True,
                sample_num=l_cnt,
            )

            # not gonna happen
            assert sum_cnt > l_cnt, "sum cnt {} not greater than l cnt {}".format(sum_cnt, l_cnt)

            r_g = float(sum_grad - l_g)
            r_h = float(sum_hess - l_h)
            r_cnt = sum_cnt - l_cnt

            right_node = Node(
                nid=r_id,
                grad=r_g,
                hess=r_h,
                weight=float(self.splitter.node_weight(sum_grad - l_g, sum_hess - l_h)),
                parent_nodeid=p_id,
                sibling_nodeid=l_id,
                sample_num=r_cnt,
                is_left_node=False,
            )

            next_layer_node.append(left_node)
            next_layer_node.append(right_node)
            self._nodes.append(node)

        return next_layer_node

    def _drop_samples_on_leaves(self, new_sample_pos: DataFrame, data: DataFrame, grad_and_hess: DataFrame):
        assert len(new_sample_pos) == len(
            data
        ), "sample pos num not match data num, got {} sample pos vs {} data".format(len(new_sample_pos), len(data))
        x = new_sample_pos >= 0
        pack_data = DataFrame.hstack([data, new_sample_pos, grad_and_hess]).iloc(x)
        new_data = pack_data.create_frame(columns=data.schema.columns)
        update_pos = pack_data.create_frame(columns=new_sample_pos.schema.columns)
        grad_and_hess = pack_data.create_frame(columns=grad_and_hess.schema.columns)
        """
        new_data = data.iloc(x)
        update_pos = new_sample_pos.iloc(x)
        grad_and_hess = grad_and_hess.iloc(x)
        """
        logger.info(
            "drop leaf samples, new sample count is {}, {} samples dropped".format(
                len(new_sample_pos), len(data) - len(new_data)
            )
        )
        return new_data, update_pos, grad_and_hess

    def _get_samples_on_leaves(self, sample_pos: DataFrame):
        x = sample_pos < 0
        samples_on_leaves = sample_pos.iloc(x)
        return samples_on_leaves

    def _get_column_max_bin(self, result_dict):
        bin_len = {}
        for column, values in result_dict.items():
            bin_num = len(values)
            bin_len[column] = bin_num
        max_max_value = max(bin_len.values())
        return bin_len, max_max_value

    def fit(self, ctx: Context, train_data: DataFrame, grad_and_hess: DataFrame, encryptor):
        raise NotImplementedError("This method should be implemented by subclass")

    def get_feature_importance(self):
        return self._feature_importance

    def get_sample_predict_weights(self):
        return self._sample_weights

    def get_nodes(self):
        return self._nodes

    def print_tree(self):
        from anytree import Node as AnyNode, RenderTree

        nodes = self._nodes
        anytree_nodes = {}
        for node in nodes:
            if not node.is_leaf:
                anytree_nodes[node.nid] = AnyNode(
                    name=f"{node.nid}: fid {node.fid}, bid {node.bid}, sample num {node.sample_num}, on {node.sitename}"
                )
            else:
                anytree_nodes[node.nid] = AnyNode(
                    name=f"{node.nid}: weight {node.weight}, sample num {node.sample_num}, leaf"
                )
        for node in nodes:
            if node.l != -1:
                anytree_nodes[node.l].parent = anytree_nodes[node.nid]
            if node.r != -1:
                anytree_nodes[node.r].parent = anytree_nodes[node.nid]

        for pre, _, node in RenderTree(anytree_nodes[0]):
            print("%s%s" % (pre, node.name))

    @staticmethod
    def _recover_nodes(model_dict):
        nodes = []
        for node_dict in model_dict["nodes"]:
            node = Node(**node_dict)
            nodes.append(node)
        return nodes

    @staticmethod
    def _recover_feature_importance(model_dict):
        feature_importance = {}
        for k, v in model_dict["feature_importance"].items():
            feature_importance[k] = FeatureImportance.from_dict(v)
        return feature_importance

    @staticmethod
    def _from_model(model_dict, tree_class):
        nodes = DecisionTree._recover_nodes(model_dict)
        feature_importance = DecisionTree._recover_feature_importance(model_dict)
        param = model_dict["hyper_param"]
        tree = tree_class(**param)
        tree._nodes = nodes
        tree._feature_importance = feature_importance
        return tree

    def get_hyper_param(self):
        param = {"max_depth": self.max_depth, "use_missing": self.use_missing, "zero_as_missing": self.zero_as_missing}
        return param

    @staticmethod
    def from_model(model_dict):
        return DecisionTree._from_model(model_dict, DecisionTree)

    def get_model(self):
        model_dict = {}
        nodes = [n.to_dict() for n in self._nodes]
        feat_importance = {k: v.to_dict() for k, v in self._feature_importance.items()}
        param = self.get_hyper_param()
        model_dict["nodes"] = nodes
        model_dict["feature_importance"] = feat_importance
        model_dict["hyper_param"] = param

        return model_dict
