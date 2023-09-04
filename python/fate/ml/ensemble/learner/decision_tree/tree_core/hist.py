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
import torch
from typing import Dict
from sklearn.ensemble._hist_gradient_boosting.grower import HistogramBuilder
from fate.arch.histogram.histogram import DistributedHistogram, Histogram
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import Node
from typing import List
import numpy as np
import pandas as pd
from fate.arch.dataframe import DataFrame
from fate.arch import Context
import logging



HIST_TYPE = ['distributed', 'sklearn']

class SklearnHistBuilder(object):

    def __init__(self, bin_data, bin_num, g, h) -> None:
        
        try:
            hist_builder = HistogramBuilder(bin_data, bin_num, g, h, False)
        except TypeError as e:
            from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
            n_threads = _openmp_effective_n_threads(None)
            hist_builder = HistogramBuilder(bin_data, bin_num, g, h, False, n_threads)
    
        self.hist_builder = hist_builder

    
    def compute_hist(self, nodes: List[Node], bin_train_data=None, gh=None, sample_pos: DataFrame = None, node_map={}, debug=False):
        
        grouped = sample_pos.as_pd_df().groupby('node_idx')['sample_id'].apply(np.array).apply(np.uint32)
        data_indices = [None for i in range(len(nodes))]
        inverse_node_map = {v: k for k, v in node_map.items()}
        print('grouped is {}'.format(grouped.keys()))
        print('node map is {}'.format(node_map))
        for idx, node in enumerate(nodes):
            data_indices[idx] = grouped[inverse_node_map[idx]]
       
        hists = []
        idx = 0
        for node in nodes:
            hist = self.hist_builder.compute_histograms_brute(data_indices[idx])
            hist_arr = np.array(hist)
            g = hist_arr['sum_gradients'].cumsum(axis=1)
            h = hist_arr['sum_hessians'].cumsum(axis=1)
            count = hist_arr['count'].cumsum(axis=1)
            hists.append([g, h, count])
            idx += 1

        if debug:
            return hists, data_indices
        else:
            return hists
        

# def get_hist_builder(bin_train_data, grad_and_hess, root_node, max_bin, bin_info, hist_type='distributed'):
    
#     assert hist_type in HIST_TYPE, 'hist_type should be in {}'.format(HIST_TYPE)

#     if hist_type == 'distributed':
#         pass

#     if hist_type == 'sklearn':

#         if isinstance(bin_train_data, DataFrame):
#             data = bin_train_data.as_pd_df()
#         elif isinstance(bin_train_data, pd.DataFrame):
#             data = bin_train_data

#         if isinstance(grad_and_hess, DataFrame):
#             gh = grad_and_hess.as_pd_df()
#         elif isinstance(grad_and_hess, pd.DataFrame):
#             gh = grad_and_hess

#         data['sample_id'] = data['sample_id'].astype(np.uint32)
#         gh['sample_id'] = gh['sample_id'].astype(np.uint32)
#         collect_data = data.sort_values(by='sample_id')
#         collect_gh = gh.sort_values(by='sample_id')
#         if bin_train_data.schema.label_name is None:
#             feat_arr = collect_data.drop(columns=[bin_train_data.schema.sample_id_name, bin_train_data.schema.match_id_name]).values
#         else:
#             feat_arr = collect_data.drop(columns=[bin_train_data.schema.sample_id_name, bin_train_data.schema.label_name, bin_train_data.schema.match_id_name]).values
#         g = collect_gh['g'].values
#         h = collect_gh['h'].values
#         feat_arr = np.asfortranarray(feat_arr.astype(np.uint8))
#         return SklearnHistBuilder(feat_arr, max_bin, g, h)

class SBTHistogramBuilder(object):

    def __init__(self, bin_train_data: DataFrame, bin_info: dict, random_seed=None) -> None:
        
        columns = bin_train_data.schema.columns
        self.random_seed = random_seed
        self.feat_bin_num = [len(bin_info[feat]) for feat in columns]

    def _get_plain_text_schema(self):
        return {
                "g": {"type": "tensor", "stride": 1, "dtype": torch.float32},
                "h": {"type": "tensor", "stride": 1, "dtype": torch.float32},
                "cnt": {"type": "tensor", "stride": 1, "dtype": torch.int32},
            }
    
    def _get_enc_hist_schema(self, pk, evaluator):
        return {
                "g":{"type": "paillier", "stride": 1, "pk": pk, "evaluator": evaluator},
                "h":{"type": "paillier", "stride": 1, "pk": pk, "evaluator": evaluator},
                "cnt": {"type": "tensor", "stride": 1, "dtype": torch.int32},
            }
    
    def _get_pack_en_hist_schema(self, pk, evaluator):
            return {
                "gh":{"type": "paillier", "stride": 1, "pk": pk, "evaluator": evaluator},
                "cnt": {"type": "tensor", "stride": 1, "dtype": torch.int32},
            }

    def compute_hist(self, ctx: Context, nodes: List[Node], bin_train_data: DataFrame, gh: DataFrame, sample_pos: DataFrame = None, node_map={}, 
                     pk=None, evaluator=None, gh_pack=False):

        node_num = len(nodes)
        if ctx.is_on_guest:
            schema = self._get_plain_text_schema()
        elif ctx.is_on_host:
            if pk is None or evaluator is None:
                schema = self._get_plain_text_schema()
            else:
                if gh_pack:
                    schema = self._get_pack_en_hist_schema(pk, evaluator)
                else:
                    schema = self._get_enc_hist_schema(pk, evaluator)

        hist = DistributedHistogram(
            node_size=node_num,
            feature_bin_sizes=self.feat_bin_num,
            value_schemas=schema,
            seed=self.random_seed,
        )
        # indexer = bin_train_data.get_indexer('sample_id')
        # gh = gh.loc(indexer, preserve_order=True)
        # gh["cnt"] = 1
        # sample_pos = sample_pos.loc(indexer, preserve_order=True)
        # map_sample_pos = sample_pos.create_frame()
        map_sample_pos = sample_pos.apply_row(lambda x: node_map[x['node_idx']])

        stat_obj = bin_train_data.distributed_hist_stat(hist, map_sample_pos, gh)

        return hist, stat_obj

    def recover_feature_bins(self, hist: DistributedHistogram, nid_split_id: Dict[int, int], node_map: dict) -> Dict[int, int]:
        if self.random_seed is None:
            return nid_split_id  # randome seed has no shuffle, no need to recover
        else:
            reverse_node_map = {v: k for k, v in node_map.items()}
            nid_split_id_ = {node_map[k]: v for k, v in nid_split_id.items()}
            recover = hist.recover_feature_bins(self.random_seed, nid_split_id_)
            print('recover rs is', recover)
            recover_rs = {reverse_node_map[k]: v for k, v in recover.items()}
            return recover_rs
