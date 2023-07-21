from fate.ml.ensemble.learner.decision_tree.tree_core.decision_tree import DecisionTree, Node
from fate.ml.ensemble.learner.decision_tree.tree_core.hist import SklearnHistBuilder
from fate.ml.ensemble.learner.decision_tree.tree_core.splitter import SklearnSplitter
from fate.arch import Context
from fate.arch.dataframe import DataFrame
import numpy as np


class HeteroDecisionTreeGuest(DecisionTree):

    def __init__(self, max_depth=3, feature_importance_type='split', valid_features=None, max_split_nodes=1024):
        super().__init__(max_depth, use_missing=False, zero_as_missing=False, feature_importance_type=feature_importance_type, valid_features=valid_features)
        self.max_split_nodes = max_split_nodes
        self.hist_builder = None
        self.splitter = None

    def _compute_best_splits(self):
        pass

    def get_column_max_bin(self, result_dict):
        bin_len = {}
        
        for column, values in result_dict.items():
            bin_num = len(values)
            bin_len[column] = bin_num 
        
        max_max_value = max(bin_len.values())
        
        return bin_len, max_max_value
    
    def get_sklearn_hist_builder(self, bin_train_data, grad_and_hess, root_node, max_bin):
        collect_data = bin_train_data.as_pd_df().sort_values(by='sample_id')
        collect_gh = grad_and_hess.as_pd_df().sort_values(by='sample_id')
        root_node.set_inst_indices(collect_gh['sample_id'].values)
        feat_arr = collect_data.drop(columns=[bin_train_data.schema.sample_id_name, bin_train_data.schema.label_name, bin_train_data.schema.match_id_name]).values
        g = collect_gh['g'].values
        h = collect_gh['h'].values
        feat_arr = np.asfortranarray(feat_arr.astype(np.uint8))
        return SklearnHistBuilder(feat_arr, max_bin, g, h)
    
    def get_sklearn_splitter(self, bin_train_data, grad_and_hess, root_node, max_bin):
        pass

    def get_distribute_hist_builder(self, bin_train_data, grad_and_hess, root_node, max_bin):
        pass

    def booster_fit(self, ctx: Context, bin_train_data: DataFrame, grad_and_hess: DataFrame, bining_dict: dict):
        

        feat_max_bin, max_bin = self.get_column_max_bin(bining_dict)
        sample_pos = self._init_sample_pos(bin_train_data)
        root_node = self._initialize_root_node(grad_and_hess, ctx.guest.party[0] + '_' + ctx.guest.party[1])
        
        self._nodes.append(root_node)

        # init histogram builder
        self.hist_builder = self.get_sklearn_hist_builder(bin_train_data, grad_and_hess, root_node, max_bin)
        # init splitter
        self.splitter = SklearnSplitter(bining_dict)

        self.cur_layer_node = [root_node]
        for cur_depth in range(self.max_depth):

            # select samples on cur nodes

            # compute histogram
            hist = self.hist_builder.compute_hist(self.cur_layer_node, grad_and_hess)
            # compute best splits
            # update tree with best splits

            # update sample position

            # update cur_layer_node
            break


        return root_node, feat_max_bin, max_bin, hist, self.splitter

    def fit(self, ctx: Context, train_data: DataFrame):
        pass

    def predict(self, ctx: Context, data_inst: DataFrame):
        pass
    
