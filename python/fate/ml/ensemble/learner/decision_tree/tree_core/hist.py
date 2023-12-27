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
import typing

import torch
from typing import Dict
from fate.arch.histogram import HistogramBuilder, DistributedHistogram
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import Node
from typing import List
import numpy as np
from fate.arch.dataframe import DataFrame
from fate.arch import Context
import logging


logger = logging.getLogger(__name__)


HIST_TYPE = ["distributed", "sklearn"]


class SklearnHistBuilder(object):
    def __init__(self, bin_data, bin_num, g, h) -> None:
        from sklearn.ensemble._hist_gradient_boosting.grower import HistogramBuilder

        try:
            hist_builder = HistogramBuilder(bin_data, bin_num, g, h, False)
        except TypeError as e:
            from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

            n_threads = _openmp_effective_n_threads(None)
            hist_builder = HistogramBuilder(bin_data, bin_num, g, h, False, n_threads)

        self.hist_builder = hist_builder

    def compute_hist(
        self, nodes: List[Node], bin_train_data=None, gh=None, sample_pos: DataFrame = None, node_map={}, debug=False
    ):
        grouped = sample_pos.as_pd_df().groupby("node_idx")["sample_id"].apply(np.array).apply(np.uint32)
        data_indices = [None for i in range(len(nodes))]
        inverse_node_map = {v: k for k, v in node_map.items()}
        for idx, node in enumerate(nodes):
            data_indices[idx] = grouped[inverse_node_map[idx]]

        hists = []
        idx = 0
        for node in nodes:
            hist = self.hist_builder.compute_histograms_brute(data_indices[idx])
            hist_arr = np.array(hist)
            g = hist_arr["sum_gradients"].cumsum(axis=1)
            h = hist_arr["sum_hessians"].cumsum(axis=1)
            count = hist_arr["count"].cumsum(axis=1)
            hists.append([g, h, count])
            idx += 1

        if debug:
            return hists, data_indices
        else:
            return hists


class SBTHistogramBuilder(object):
    def __init__(
        self, bin_train_data: DataFrame, bin_info: dict, random_seed=None, global_random_seed=None, hist_sub=True
    ) -> None:
        columns = bin_train_data.schema.columns
        self.random_seed = random_seed
        self.global_random_seed = global_random_seed
        self.feat_bin_num = [len(bin_info[feat]) for feat in columns]
        self._cache_parent_hist: typing.Optional[DistributedHistogram] = None
        self._last_layer_node_map = None
        self._hist_sub = hist_sub

    def _get_plain_text_schema(self, dtypes):
        return {
            "g": {"type": "plaintext", "stride": 1, "dtype": dtypes["g"]},
            "h": {"type": "plaintext", "stride": 1, "dtype": dtypes["h"]},
            "cnt": {"type": "plaintext", "stride": 1, "dtype": dtypes["cnt"]},
        }

    def _get_enc_hist_schema(self, pk, evaluator, dtypes):
        return {
            "g": {"type": "ciphertext", "stride": 1, "pk": pk, "evaluator": evaluator, "dtype": dtypes["g"]},
            "h": {"type": "ciphertext", "stride": 1, "pk": pk, "evaluator": evaluator, "dtype": dtypes["h"]},
            "cnt": {"type": "plaintext", "stride": 1, "dtype": dtypes["cnt"]},
        }

    def _get_pack_en_hist_schema(self, pk, evaluator, dtypes):
        return {
            "gh": {"type": "ciphertext", "stride": 1, "pk": pk, "evaluator": evaluator, "dtype": dtypes["gh"]},
            "cnt": {"type": "plaintext", "stride": 1, "dtype": dtypes["cnt"]},
        }

    def _prepare_hist_sub(self, nodes: List[Node], cur_layer_node_map: dict, parent_node_map: dict):
        weak_nodes_ids = []
        mapping = []
        n_map = {n.nid: n for n in nodes}
        new_node_map = {}
        hist_pos = 0
        for n in nodes:
            if n.nid == 0:
                # root node
                weak_nodes_ids.append(0)
                # root node, just return
                return set(weak_nodes_ids), None, mapping

            if n.is_left_node:
                sib = n_map[n.sibling_nodeid]

                if sib.sample_num < n.sample_num:
                    weak_node = sib
                else:
                    weak_node = n

                mapping_list = []
                parent_nid = weak_node.parent_nodeid
                weak_nodes_ids.append(weak_node.nid)
                mapping_list = (
                    parent_node_map[parent_nid],
                    hist_pos,
                    cur_layer_node_map[weak_node.nid],
                    cur_layer_node_map[weak_node.sibling_nodeid],
                )
                mapping.append(mapping_list)
                new_node_map[weak_node.nid] = hist_pos
                hist_pos += 1

            else:
                continue
        return set(weak_nodes_ids), new_node_map, mapping

    def _get_samples_on_weak_nodes(self, sample_pos: DataFrame, weak_nodes: set):
        # root node
        if 0 in weak_nodes:
            return sample_pos
        is_on_weak = sample_pos.apply_row(lambda s: s["node_idx"] in weak_nodes)
        weak_sample_pos = sample_pos.iloc(is_on_weak)
        return weak_sample_pos

    def _is_first_layer(self, nodes):
        if len(nodes) == 1 and nodes[0].nid == 0:
            return True
        else:
            return False

    def compute_hist(
        self,
        ctx: Context,
        nodes: List[Node],
        bin_train_data: DataFrame,
        gh: DataFrame,
        sample_pos: DataFrame = None,
        node_map={},
        pk=None,
        evaluator=None,
        gh_pack=False,
    ):
        node_num = len(nodes)
        is_first_layer = self._is_first_layer(nodes)
        need_hist_sub_process = (not is_first_layer) and self._hist_sub

        weak_nodes, new_node_map, mapping = None, None, None
        if need_hist_sub_process:
            weak_nodes, new_node_map, mapping = self._prepare_hist_sub(nodes, node_map, self._last_layer_node_map)
            node_num = len(weak_nodes)
            logger.debug("weak nodes {}, new_node_map {}, mapping {}".format(weak_nodes, new_node_map, mapping))

        if ctx.is_on_guest:
            schema = self._get_plain_text_schema(gh.dtypes)
        elif ctx.is_on_host:
            if pk is None or evaluator is None:
                schema = self._get_plain_text_schema(gh.dtypes)
            else:
                if gh_pack:
                    schema = self._get_pack_en_hist_schema(pk, evaluator, gh.dtypes)
                else:
                    schema = self._get_enc_hist_schema(pk, evaluator, gh.dtypes)
        else:
            raise ValueError("not support called on role: {}".format(ctx.local))

        if need_hist_sub_process:
            node_mapping = {node_map[k]: v for k, v in new_node_map.items()}
        else:
            node_mapping = None

        hist = HistogramBuilder(
            num_node=node_num,
            feature_bin_sizes=self.feat_bin_num,
            value_schemas=schema,
            global_seed=self.global_random_seed,
            seed=self.random_seed,
            node_mapping=node_mapping,
        )

        # if goss is enabled
        if len(sample_pos) > len(gh):
            sample_pos = sample_pos.loc(gh.get_indexer(target="sample_id"), preserve_order=True)
            map_sample_pos = sample_pos.apply_row(lambda x: node_map[x["node_idx"]])
            bin_train_data = bin_train_data.loc(gh.get_indexer(target="sample_id"), preserve_order=True)
        else:
            map_sample_pos = sample_pos.apply_row(lambda x: node_map[x["node_idx"]])

        stat_obj = bin_train_data.distributed_hist_stat(hist, map_sample_pos, gh)

        if need_hist_sub_process:
            stat_obj = self._cache_parent_hist.compute_child(stat_obj, mapping)

        if self._hist_sub:
            self._cache_parent_hist = stat_obj
            self._last_layer_node_map = node_map

        stat_obj = stat_obj.shuffle_splits()

        return hist, stat_obj

    def recover_feature_bins(
        self, statistic_histogram: DistributedHistogram, nid_split_id: Dict[int, int], node_map: dict
    ) -> Dict[int, int]:
        if self.random_seed is None:
            return nid_split_id  # randome seed has no shuffle, no need to recover
        else:
            reverse_node_map = {v: k for k, v in node_map.items()}
            nid_split_id_ = {node_map[k]: v for k, v in nid_split_id.items()}
            recover = statistic_histogram.recover_feature_bins(self.feat_bin_num, nid_split_id_)
            recover_rs = {reverse_node_map[k]: v for k, v in recover.items()}
            return recover_rs
