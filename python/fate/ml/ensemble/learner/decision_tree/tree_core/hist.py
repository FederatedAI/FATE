from sklearn.ensemble._hist_gradient_boosting.grower import HistogramBuilder
from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import Node
from typing import List
import numpy as np


class SklearnHistBuilder(object):

    def __init__(self, bin_data, bin_num, g, h) -> None:
        
        try:
            hist_builder = HistogramBuilder(bin_data, bin_num, g, h, False)
        except TypeError as e:
            from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
            n_threads = _openmp_effective_n_threads(None)
            hist_builder = HistogramBuilder(bin_data, bin_num, g, h, False, n_threads)
    
        self.hist_builder = hist_builder

    
    def compute_hist(self, nodes: List[Node], gh=None, ):
        
        hists = []
        for node in nodes:
            hist = self.hist_builder.compute_histograms_brute(node.inst_indices)
            hist_arr = np.array(hist)
            g = hist_arr['sum_gradients'].cumsum(axis=1)
            h = hist_arr['sum_hessians'].cumsum(axis=1)
            count = hist_arr['count'].cumsum(axis=1)
            hists.append([g, h, count])

        return hists