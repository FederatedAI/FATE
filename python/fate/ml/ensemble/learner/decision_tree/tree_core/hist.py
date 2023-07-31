from sklearn.ensemble._hist_gradient_boosting.grower import HistogramBuilder
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import Node
from typing import List
import numpy as np
import pandas as pd
from fate.arch.dataframe import DataFrame


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
    

def get_hist_builder(bin_train_data, grad_and_hess, root_node, max_bin, hist_type='distributed'):
    
    assert hist_type in HIST_TYPE, 'hist_type should be in {}'.format(HIST_TYPE)

    if hist_type == 'sklearn':

        if isinstance(bin_train_data, DataFrame):
            data = bin_train_data.as_pd_df()
        elif isinstance(bin_train_data, pd.DataFrame):
            data = bin_train_data

        if isinstance(grad_and_hess, DataFrame):
            gh = grad_and_hess.as_pd_df()
        elif isinstance(grad_and_hess, pd.DataFrame):
            gh = grad_and_hess

        data['sample_id'] = data['sample_id'].astype(np.uint32)
        gh['sample_id'] = gh['sample_id'].astype(np.uint32)
        collect_data = data.sort_values(by='sample_id')
        collect_gh = gh.sort_values(by='sample_id')
        if bin_train_data.schema.label_name is None:
            feat_arr = collect_data.drop(columns=[bin_train_data.schema.sample_id_name, bin_train_data.schema.match_id_name]).values
        else:
            feat_arr = collect_data.drop(columns=[bin_train_data.schema.sample_id_name, bin_train_data.schema.label_name, bin_train_data.schema.match_id_name]).values
        g = collect_gh['g'].values
        h = collect_gh['h'].values
        feat_arr = np.asfortranarray(feat_arr.astype(np.uint8))
        return SklearnHistBuilder(feat_arr, max_bin, g, h)