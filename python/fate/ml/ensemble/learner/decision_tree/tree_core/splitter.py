import copy
import numpy as np
import logging


logger = logging.getLogger(__name__)


class SplitInfo(object):
    def __init__(self, best_fid=None, best_bid=None,
                 sum_grad=0, sum_hess=0, gain=None, missing_dir=1, mask_id=None, sample_count=-1):
        
        self.best_fid = best_fid
        self.best_bid = best_bid
        self.sum_grad = sum_grad
        self.sum_hess = sum_hess
        self.gain = gain
        self.missing_dir = missing_dir
        self.mask_id = mask_id
        self.sample_count = sample_count

    def __str__(self):
        return '(fid {} bid {}, sum_grad {}, sum_hess {}, gain {}, missing dir {}, mask_id {}, ' \
               'sample_count {})\n'.format(
                   self.best_fid, self.best_bid, self.sum_grad, self.sum_hess, self.gain, self.missing_dir,
                   self.mask_id, self.sample_count)

    def __repr__(self):
        return self.__str__()


class Splitter(object):

    def __init__(self) -> None:
        pass


class SklearnSplitter(Splitter):

    def __init__(self, feature_binning_dict, min_impurity_split=1e-2, min_sample_split=2,
                 min_leaf_node=1, min_child_weight=1, l1=0, l2=0, valid_features=None) -> None:
        super().__init__()
        self.min_impurity_split = min_impurity_split
        self.min_sample_split = min_sample_split
        self.min_leaf_node = min_leaf_node
        self.min_child_weight = min_child_weight
        self.feature_binning_dict = feature_binning_dict
        self.hist_mask = self.generate_mask(feature_binning_dict)
        self.l1, self.l2 = l1, l2

    def generate_mask(self, feature_dict):
        split_counts = [len(split_point) for split_point in feature_dict.values()]
        max_bin = max(split_counts) 
        mask = np.zeros((len(feature_dict), max_bin)).astype(np.bool8)
        for i, bucket_count in enumerate(split_counts):
            mask[i, :bucket_count] = True  # valid split point
        
        return ~mask

    def node_gain(self, g, h):
        if isinstance(h, np.ndarray):
            h[h == 0] = np.nan
        score = g * g / (h + self.l2)
        return score

    def find_node_best_split(self, node_hist):

        l_g, l_h, l_cnt = node_hist
        cnt_sum = l_cnt[::, -1][0]
        g_sum = l_g[::, -1][0]
        h_sum = l_h[::, -1][0]
        
        if cnt_sum < self.min_sample_split:
            return None

        r_g, r_h = g_sum - l_g, h_sum - l_h
        r_cnt = cnt_sum - l_cnt
        
        # filter split
        # leaf count
        min_leaf_node_mask_l = l_cnt < self.min_leaf_node
        min_leaf_node_mask_r = r_cnt < self.min_leaf_node
        union_mask_0 = np.logical_or(min_leaf_node_mask_l, min_leaf_node_mask_r)
        # min child weight
        min_child_weight_mask_l = l_h < self.min_child_weight
        min_child_weight_mask_r = r_h < self.min_child_weight
        union_mask_1 = np.logical_or(min_child_weight_mask_l, min_child_weight_mask_r)
        mask = np.logical_or(union_mask_0, self.hist_mask)
        mask = np.logical_or(mask, union_mask_1)

        rs = self.node_gain(l_g, l_h) + \
            self.node_gain(r_g, r_h) - \
            self.node_gain(g_sum, h_sum)
        
        rs[np.isnan(rs)] = -np.inf
        rs[rs < self.min_impurity_split] = -np.inf
        rs[mask] = -np.inf

        # reduce
        feat_best_split = rs.argmax(axis=1)
        feat_best_gain = rs.max(axis=1)

        # best split
        best_split_idx = feat_best_gain.argmax()
        best_gain = feat_best_gain.max()
        
        if best_gain == -np.inf:
            # can not split
            logger.info('this node cannot be further split')
            return None

        feat_id = best_split_idx
        bin_id = feat_best_split[best_split_idx]

        split_info = SplitInfo(
            best_fid=feat_id,
            best_bid=bin_id,
            gain=best_gain,
            sum_grad=l_g[feat_id][bin_id],
            sum_hess=l_h[feat_id][bin_id],
            sample_count=l_cnt[feat_id][bin_id]
        )
        
        return split_info
    
    def split(self, histogram: list):
        
        splits = []
        for node_hist in histogram:
            split_info = self.find_node_best_split(node_hist)
            splits.append(split_info)
        return splits