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
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import (
    DecisionTree,
    Node,
    _update_sample_pos_on_local_nodes,
    _merge_sample_pos,
)
from fate.ml.ensemble.learner.decision_tree.tree_core.hist import SBTHistogramBuilder
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import FedSBTSplitter
from fate.ml.ensemble.learner.decision_tree.tree_core.loss import get_task_info
from fate.ml.utils.predict_tools import BINARY, MULTI, REGRESSION
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from typing import List
import functools
import logging
import pandas as pd
import torch as t
import numpy as np
import math


logger = logging.getLogger(__name__)

FIX_POINT_PRECISION = 52


class HeteroDecisionTreeGuest(DecisionTree):
    def __init__(
        self,
        max_depth=3,
        valid_features=None,
        use_missing=False,
        zero_as_missing=False,
        goss=False,
        l1=0.1,
        l2=0,
        min_impurity_split=1e-2,
        min_sample_split=2,
        min_leaf_node=1,
        min_child_weight=1,
        objective=None,
        gh_pack=True,
        split_info_pack=True,
        hist_sub=True,
    ):
        super().__init__(
            max_depth, use_missing=use_missing, zero_as_missing=zero_as_missing, valid_features=valid_features
        )
        self.host_sitenames = None
        self._tree_node_num = 0
        self.hist_builder = None
        self.splitter = None

        # regularization
        self.l1 = l1
        self.l2 = l2
        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node
        self.min_child_weight = min_child_weight

        # goss
        self.goss = goss

        # other
        self._valid_features = valid_features
        self._hist_sub = hist_sub

        # homographic encryption
        self._encrypt_kit = None
        self._sk = None
        self._pk = None
        self._coder = None
        self._evaluator = None
        self._encryptor = None
        self._decryptor = None

        # for g, h packing
        self._en_key_length = None
        self._gh_pack = gh_pack
        self._split_info_pack = split_info_pack
        self._g_offset = 0
        self._g_abs_max = 0
        self._h_abs_max = 0
        self._objective = objective
        if gh_pack:
            if objective is None:
                raise ValueError("objective must be specified when gh_pack is True")
        self._pack_info = {}

    def set_encrypt_kit(self, kit):
        self._encrypt_kit = kit
        self._en_key_length = kit.key_size
        self._sk, self._pk, self._coder, self._evaluator, self._encryptor = (
            kit.sk,
            kit.pk,
            kit.coder,
            kit.evaluator,
            kit.get_tensor_encryptor(),
        )
        self._decryptor = kit.get_tensor_decryptor()
        logger.info("encrypt kit setup through setter")

    def _init_encrypt_kit(self, ctx: Context):
        kit = ctx.cipher.phe.setup(options={"kind": "paillier", "key_length": 1024})
        self._en_key_length = kit.key_size
        self._sk, self._pk, self._coder, self._evaluator, self._encryptor = (
            kit.sk,
            kit.pk,
            kit.coder,
            kit.evaluator,
            kit.get_tensor_encryptor(),
        )
        self._decryptor = kit.get_tensor_decryptor()
        logger.info("encrypt kit is not setup, auto initializing")

    def _get_column_max_bin(self, result_dict):
        bin_len = {}

        for column, values in result_dict.items():
            bin_num = len(values)
            bin_len[column] = bin_num

        max_max_value = max(bin_len.values())

        return bin_len, max_max_value

    def _update_sample_pos(
        self, ctx: Context, cur_layer_nodes: List[Node], sample_pos: DataFrame, data: DataFrame, node_map: dict
    ):
        sitename = ctx.local.name
        data_with_pos = DataFrame.hstack([data, sample_pos])
        map_func = functools.partial(
            _update_sample_pos_on_local_nodes, cur_layer_node=cur_layer_nodes, node_map=node_map, sitename=sitename
        )
        updated_sample_pos = data_with_pos.apply_row(map_func, columns=["g_on_local", "g_node_idx"])

        # synchronize sample pos
        host_update_sample_pos = ctx.hosts.get("updated_data")

        merge_func = functools.partial(_merge_sample_pos)
        for host_data in host_update_sample_pos:
            updated_sample_pos = DataFrame.hstack([updated_sample_pos, host_data]).apply_row(
                merge_func, columns=["g_on_local", "g_node_idx"]
            )

        new_sample_pos = updated_sample_pos.create_frame(columns=["g_node_idx"])
        new_sample_pos.rename(columns={"g_node_idx": "node_idx"})
        ctx.hosts.put("new_sample_pos", new_sample_pos)
        self.sample_pos = new_sample_pos

        return new_sample_pos

    def _g_h_process(self, grad_and_hess: DataFrame):
        en_grad_hess = grad_and_hess.create_frame()

        def make_long_tensor(s: pd.Series, coder, pk, offset, shift_bit, precision, encryptor, pack_num=2):
            pack_tensor = t.Tensor(s.values)
            pack_tensor[0] = pack_tensor[0] + offset
            pack_vec = coder.pack_floats(pack_tensor, shift_bit, pack_num, precision)
            en = pk.encrypt_encoded(pack_vec, obfuscate=True)
            ret = encryptor.lift(en, (len(en), 1), pack_tensor.dtype, pack_tensor.device)
            return ret

        def compute_offset_bit(sample_num, g_max, h_max):
            g_bit = int(math.log2(2**FIX_POINT_PRECISION * sample_num * g_max) + 1)  # add 1 more bit for safety
            h_bit = int(math.log2(2**FIX_POINT_PRECISION * sample_num * h_max) + 1)
            return max(g_bit, h_bit)

        if self._gh_pack:
            task_type = get_task_info(self._objective)

            if task_type == BINARY or task_type == MULTI:
                self._g_offset = 1
                self._g_abs_max = 2
                self._h_abs_max = 1

            elif task_type == REGRESSION:
                self._g_offset = abs(float(grad_and_hess["g"].min()["g"]))
                self._g_abs_max = abs(float(grad_and_hess["g"].max()["g"])) + self._g_offset
                self._h_abs_max = 2

            pack_num = 2
            shift_bit = compute_offset_bit(len(grad_and_hess), self._g_abs_max, self._h_abs_max)
            total_pack_num = (self._en_key_length - 2) // (shift_bit * pack_num)  # -2 in case overflow
            partial_func = functools.partial(
                make_long_tensor,
                coder=self._coder,
                offset=self._g_offset,
                pk=self._pk,
                shift_bit=shift_bit,
                pack_num=2,
                precision=FIX_POINT_PRECISION,
                encryptor=self._encryptor,
            )
            en_grad_hess["gh"] = grad_and_hess.apply_row(partial_func)

            # record pack info
            self._pack_info["g_offset"] = self._g_offset
            self._pack_info["shift_bit"] = shift_bit
            self._pack_info["precision"] = FIX_POINT_PRECISION
            self._pack_info["pack_num"] = pack_num
            self._pack_info["total_pack_num"] = total_pack_num
            self._pack_info["split_point_shift_bit"] = shift_bit * pack_num
            logger.info("gh are packed")
        else:
            en_grad_hess["g"] = self._encryptor.encrypt_tensor(grad_and_hess["g"].as_tensor())
            en_grad_hess["h"] = self._encryptor.encrypt_tensor(grad_and_hess["h"].as_tensor())
            logger.info("not using gh pack")

        return en_grad_hess

    def _send_gh(self, ctx: Context, grad_and_hess: DataFrame):
        # encrypt g & h
        en_grad_hess = self._g_h_process(grad_and_hess)
        ctx.hosts.put("en_gh", en_grad_hess)
        ctx.hosts.put("en_kit", [self._pk, self._evaluator])

    def _mask_node(self, ctx: Context, nodes: List[Node]):
        new_nodes = []
        for n in nodes:
            new_nodes.append(
                Node(
                    nid=n.nid,
                    is_leaf=n.is_leaf,
                    l=n.l,
                    r=n.r,
                    is_left_node=n.is_left_node,
                    split_id=n.split_id,
                    sitename=n.sitename,
                    sibling_nodeid=n.sibling_nodeid,
                    parent_nodeid=n.parent_nodeid,
                    sample_num=n.sample_num,
                )
            )
        return new_nodes

    def _check_assign_result(self, sample_pos: DataFrame, cur_layer_node: List[Node]):
        # debugging function
        sample_pos_df = sample_pos.as_pd_df()
        sample_pos_count = sample_pos_df.groupby("node_idx").count().to_dict()["sample_id"]
        for node in cur_layer_node:
            nid = node.nid
            sample_count_0 = node.sample_num
            sample_count_1 = sample_pos_count[nid]
            if sample_count_0 != sample_count_1:
                parent_nid = node.parent_nodeid
                for i in self._nodes:
                    if i.nid == parent_nid:
                        logger.info("parent node {}".format(i))
                raise ValueError(
                    "node {} sample count not match, {} vs {}, node details {}".format(
                        nid, sample_count_0, sample_count_1, node
                    )
                )

    def _sync_nodes(self, ctx: Context, cur_layer_nodes: List[Node], next_layer_nodes: List[Node]):
        mask_cur_layer = self._mask_node(ctx, cur_layer_nodes)
        mask_next_layer = self._mask_node(ctx, next_layer_nodes)
        ctx.hosts.put("sync_nodes", [mask_cur_layer, mask_next_layer])

    def booster_fit(self, ctx: Context, bin_train_data: DataFrame, grad_and_hess: DataFrame, binning_dict: dict):
        logger.info
        # Initialization
        train_df = bin_train_data
        sample_pos = self._init_sample_pos(train_df)
        self._sample_on_leaves = sample_pos.empty_frame()
        root_node = self._initialize_root_node(ctx, train_df, grad_and_hess)

        # initialize homographic encryption
        if self._encrypt_kit is None:
            self._init_encrypt_kit(ctx)
        # Send Encrypted Grad and Hess
        self._send_gh(ctx, grad_and_hess)

        # send pack info
        send_pack_info = (
            {
                "total_pack_num": self._pack_info["total_pack_num"],
                "split_point_shift_bit": self._pack_info["split_point_shift_bit"],
                "split_info_pack": self._split_info_pack,
            }
            if self._gh_pack
            else {}
        )
        ctx.hosts.put("pack_info", send_pack_info)

        # init histogram builder
        self.hist_builder = SBTHistogramBuilder(bin_train_data, binning_dict, None, None, hist_sub=self._hist_sub)

        # init splitter
        self.splitter = FedSBTSplitter(
            bin_train_data,
            binning_dict,
            l2=self.l2,
            l1=self.l1,
            min_sample_split=self.min_sample_split,
            min_impurity_split=self.min_impurity_split,
            min_child_weight=self.min_child_weight,
            min_leaf_node=self.min_leaf_node,
        )

        # Prepare for training
        node_map = {}
        cur_layer_node = [root_node]
        grad_and_hess["cnt"] = 1

        for cur_depth, sub_ctx in ctx.on_iterations.ctxs_range(self.max_depth):
            if len(cur_layer_node) == 0:
                logger.info("no nodes to split, stop training")
                break

            assert len(sample_pos) == len(train_df), "sample pos len not match train data len, {} vs {}".format(
                len(sample_pos), len(train_df)
            )

            # debug checking code
            # self._check_assign_result(sample_pos, cur_layer_node)
            # initialize node map
            node_map = {n.nid: idx for idx, n in enumerate(cur_layer_node)}
            # compute histogram
            hist_inst, statistic_result = self.hist_builder.compute_hist(
                sub_ctx, cur_layer_node, train_df, grad_and_hess, sample_pos, node_map
            )
            # compute best splits
            split_info = self.splitter.split(
                sub_ctx,
                statistic_result,
                cur_layer_node,
                node_map,
                self._sk,
                self._coder,
                self._gh_pack,
                self._pack_info,
            )
            # update tree with best splits
            next_layer_nodes = self._update_tree(sub_ctx, cur_layer_node, split_info, train_df)
            # update feature importance
            self._update_feature_importance(sub_ctx, split_info, train_df)
            # sync nodes
            self._sync_nodes(sub_ctx, cur_layer_node, next_layer_nodes)
            # update sample positions
            sample_pos = self._update_sample_pos(sub_ctx, cur_layer_node, sample_pos, train_df, node_map)
            # if sample reaches leaf nodes, drop them
            sample_on_leaves = self._get_samples_on_leaves(sample_pos)
            train_df, sample_pos, grad_and_hess = self._drop_samples_on_leaves(sample_pos, train_df, grad_and_hess)
            self._sample_on_leaves = DataFrame.vstack([self._sample_on_leaves, sample_on_leaves])
            # next layer nodes
            cur_layer_node = next_layer_nodes
            logger.info(
                "layer {} done: next layer will split {} nodes, active samples num {}".format(
                    cur_depth, len(cur_layer_node), len(sample_pos)
                )
            )
            self.next_layer_node = next_layer_nodes

        # handle final leaves
        if len(cur_layer_node) != 0:
            for node in cur_layer_node:
                node.is_leaf = True
                node.sitename = ctx.local.name  # leaf always on guest
                self._nodes.append(node)
            self._sample_on_leaves = DataFrame.vstack([self._sample_on_leaves, sample_pos])

        # when training is done, all samples must be on leaves
        assert len(self._sample_on_leaves) == len(bin_train_data), "sample on leaves num not match, {} vs {}".format(
            len(self._sample_on_leaves), len(bin_train_data)
        )
        # convert sample pos to weights
        self._sample_weights = self._convert_sample_pos_to_weight(self._sample_on_leaves, self._nodes)
        # convert bid to split value
        self._nodes = self._convert_bin_idx_to_split_val(ctx, self._nodes, binning_dict, bin_train_data.schema)

    def get_hyper_param(self):
        param = {
            "max_depth": self.max_depth,
            "valid_features": self._valid_features,
            "l1": self.l1,
            "l2": self.l2,
            "use_missing": self.use_missing,
            "zero_as_missing": self.zero_as_missing,
            "objective": self._objective,
        }
        return param

    @staticmethod
    def from_model(model_dict):
        return HeteroDecisionTreeGuest._from_model(model_dict, HeteroDecisionTreeGuest)
