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
    FeatureImportance,
)
from fate.ml.ensemble.learner.decision_tree.tree_core.hist import SBTHistogramBuilder, DistributedHistogram
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import FedSBTSplitter
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from typing import List
import functools
import logging


logger = logging.getLogger(__name__)


class HeteroDecisionTreeHost(DecisionTree):
    def __init__(
        self,
        max_depth=3,
        valid_features=None,
        use_missing=False,
        zero_as_missing=False,
        random_seed=None,
        global_random_seed=None,
        hist_sub=True,
    ):
        super().__init__(
            max_depth, use_missing=use_missing, zero_as_missing=zero_as_missing, valid_features=valid_features
        )
        self._tree_node_num = 0
        self.hist_builder = None
        self.splitter = None
        self._valid_features = valid_features
        self._random_seed = random_seed
        self._global_random_seed = global_random_seed
        self._pk = None
        self._evaluator = None
        self._gh_pack = True
        self._pack_info = None
        self._hist_sub = hist_sub

    def _convert_split_id(
        self,
        ctx: Context,
        cur_layer_nodes: List[Node],
        node_map: dict,
        hist_builder: SBTHistogramBuilder,
        statistic_histogram: DistributedHistogram,
        splitter: FedSBTSplitter,
        data: DataFrame,
    ):
        sitename = ctx.local.party[0] + "_" + ctx.local.party[1]
        to_recover = {}
        for idx, n in enumerate(cur_layer_nodes):
            if (not n.is_leaf) and n.sitename == sitename:
                node_id = n.nid
                split_id = n.split_id
                to_recover[node_id] = split_id

        if len(to_recover) != 0:
            if self._random_seed is None:
                for node_id, split_id in to_recover.items():
                    node = cur_layer_nodes[node_map[node_id]]
                    fid, bid = splitter.get_bucket(split_id)
                    node.fid = self._fid_to_feature_name(int(fid), data)
                    node.bid = int(bid)
            else:
                recover_rs = hist_builder.recover_feature_bins(statistic_histogram, to_recover, node_map)
                for node_id, split_tuple in recover_rs.items():
                    node = cur_layer_nodes[node_map[node_id]]
                    fid, bid = split_tuple
                    node.fid = self._fid_to_feature_name(int(fid), data)
                    node.bid = int(bid)

    def _update_host_feature_importance(self, ctx: Context, nodes: List[Node], train_df: DataFrame):
        sitename = ctx.local.party[0] + "_" + ctx.local.party[1]
        for n in nodes:
            if sitename == n.sitename:
                feat_name = n.fid
                if feat_name not in self._feature_importance:
                    self._feature_importance[feat_name] = FeatureImportance()
                else:
                    self._feature_importance[feat_name] = self._feature_importance[feat_name] + FeatureImportance()

    def _update_sample_pos(
        self, ctx, cur_layer_nodes: List[Node], sample_pos: DataFrame, data: DataFrame, node_map: dict
    ):
        sitename = ctx.local.party[0] + "_" + ctx.local.party[1]
        data_with_pos = DataFrame.hstack([data, sample_pos])
        map_func = functools.partial(
            _update_sample_pos_on_local_nodes, cur_layer_node=cur_layer_nodes, node_map=node_map, sitename=sitename
        )
        update_sample_pos = data_with_pos.apply_row(map_func, columns=["h_on_local", "h_node_idx"])

        ctx.guest.put("updated_data", update_sample_pos)
        new_sample_pos = ctx.guest.get("new_sample_pos")

        return new_sample_pos

    def _get_gh(self, ctx: Context):
        grad_and_hess: DataFrame = ctx.guest.get("en_gh")
        if len(grad_and_hess.columns) == 1:
            gh_pack = True
        elif len(grad_and_hess.columns) == 2:
            gh_pack = False
        else:
            raise ValueError("error columns, got {}".format(len(grad_and_hess.columns)))
        return grad_and_hess, gh_pack

    def _sync_nodes(self, ctx: Context):
        nodes = ctx.guest.get("sync_nodes")
        cur_layer_nodes, next_layer_nodes = nodes
        return cur_layer_nodes, next_layer_nodes

    def booster_fit(self, ctx: Context, bin_train_data: DataFrame, binning_dict: dict):
        train_df = bin_train_data
        feat_max_bin, max_bin = self._get_column_max_bin(binning_dict)
        sample_pos = self._init_sample_pos(train_df)

        # Get Encrypted Grad And Hess
        ret = self._get_gh(ctx)
        en_grad_and_hess: DataFrame = ret[0]
        self._gh_pack = ret[1]
        self._pk, self._evaluator = ctx.guest.get("en_kit")
        self._pack_info = ctx.guest.get("pack_info")
        split_info_pack = self._pack_info.get("split_info_pack", False)
        root_node = self._initialize_root_node(ctx, train_df)

        # init histogram builder
        self.hist_builder = SBTHistogramBuilder(
            bin_train_data,
            binning_dict,
            random_seed=self._random_seed,
            global_random_seed=self._global_random_seed,
            hist_sub=self._hist_sub,
        )
        # splitter
        self.splitter = FedSBTSplitter(bin_train_data, binning_dict)

        node_map = {}
        cur_layer_node = [root_node]
        en_grad_and_hess["cnt"] = 1
        for cur_depth, sub_ctx in ctx.on_iterations.ctxs_range(self.max_depth):
            if len(cur_layer_node) == 0:
                logger.info("no nodes to split, stop training")
                break

            node_map = {n.nid: idx for idx, n in enumerate(cur_layer_node)}
            # compute histogram with encrypted grad and hess
            hist_inst, statistic_histogram = self.hist_builder.compute_hist(
                sub_ctx,
                cur_layer_node,
                train_df,
                en_grad_and_hess,
                sample_pos,
                node_map,
                pk=self._pk,
                evaluator=self._evaluator,
                gh_pack=self._gh_pack,
            )

            if split_info_pack:
                logger.debug("packing split info")
                statistic_histogram.i_squeeze(
                    {"gh": (self._pack_info["total_pack_num"], self._pack_info["split_point_shift_bit"])}
                )

            self.splitter.split(sub_ctx, statistic_histogram, cur_layer_node, node_map)
            cur_layer_node, next_layer_nodes = self._sync_nodes(sub_ctx)
            self._convert_split_id(
                sub_ctx, cur_layer_node, node_map, self.hist_builder, statistic_histogram, self.splitter, train_df
            )
            self._update_host_feature_importance(sub_ctx, cur_layer_node, train_df)
            logger.info(
                "cur layer node num: {}, next layer node num: {}".format(len(cur_layer_node), len(next_layer_nodes))
            )
            sample_pos = self._update_sample_pos(sub_ctx, cur_layer_node, sample_pos, train_df, node_map)
            train_df, sample_pos, en_grad_and_hess = self._drop_samples_on_leaves(
                sample_pos, train_df, en_grad_and_hess
            )
            self._nodes += cur_layer_node
            cur_layer_node = next_layer_nodes
            logger.info(
                "layer {} done: next layer will split {} nodes, active samples num {}".format(
                    cur_depth, len(cur_layer_node), len(sample_pos)
                )
            )

        # sync complete tree
        if len(cur_layer_node) != 0:
            for node in cur_layer_node:
                node.is_leaf = True
                node.sitename = ctx.guest.party[0] + "_" + ctx.guest.party[1]
                self._nodes.append(node)

        # convert bid to split value
        self._nodes = self._convert_bin_idx_to_split_val(ctx, self._nodes, binning_dict, bin_train_data.schema)

    def get_hyper_param(self):
        param = {
            "max_depth": self.max_depth,
            "valid_features": self._valid_features,
            "use_missing": self.use_missing,
            "zero_as_missing": self.zero_as_missing,
        }
        return param

    @staticmethod
    def from_model(model_dict):
        return HeteroDecisionTreeHost._from_model(model_dict, HeteroDecisionTreeHost)
