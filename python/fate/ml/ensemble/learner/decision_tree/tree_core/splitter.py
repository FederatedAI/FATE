import copy
import numpy as np
import logging
from fate.arch.dataframe import DataFrame
from fate.arch import Context


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
    
    def node_weight(self, sum_grad, sum_hess):
        weight = -(sum_grad / (sum_hess + self.l2))
        return weight

    def _compute_min_leaf_mask(self, l_cnt, r_cnt):

        min_leaf_node_mask_l = l_cnt < self.min_leaf_node
        min_leaf_node_mask_r = r_cnt < self.min_leaf_node
        union_mask_0 = np.logical_or(min_leaf_node_mask_l, min_leaf_node_mask_r)
        return union_mask_0

    def find_node_best_split(self, node_hist, debug=False):

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
        union_mask_0 = self._compute_min_leaf_mask(l_cnt, r_cnt)
        # min child weight
        min_child_weight_mask_l = l_h < self.min_child_weight
        min_child_weight_mask_r = r_h < self.min_child_weight
        union_mask_1 = np.logical_or(min_child_weight_mask_l, min_child_weight_mask_r)
        mask = np.logical_or(union_mask_0, self.hist_mask)
        mask = np.logical_or(mask, union_mask_1)

        rs = self.node_gain(l_g, l_h) + \
            self.node_gain(r_g, r_h) - \
            self.node_gain(g_sum, h_sum)
        
        if debug:
            return rs

        rs[np.isnan(rs)] = -np.inf
        rs[rs < self.min_impurity_split] = -np.inf
        rs[mask] = -np.inf

        # reduce
        feat_best_split = rs.argmax(axis=1)
        feat_best_gain = rs.max(axis=1)

        print('best gain', feat_best_gain)
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
    
    def _split(self, histogram: list, cur_layer_node):
            
        splits = []
        logger.info('got {} hist'.format(len(histogram)))
        for node_hist in histogram:
            split_info = self.find_node_best_split(node_hist)
            splits.append(split_info)
        logger.info('split info is {}'.format(split_info))
        assert len(splits) == len(cur_layer_node), 'split info length {} != node length {}'.format(len(splits), len(cur_layer_node))
        return splits
    
    def split(self, ctx, histogram: list,  cur_layer_node):
        return self._split(histogram, cur_layer_node)


class FedSklearnSplitter(SklearnSplitter):

    def __init__(self, feature_binning_dict, min_impurity_split=1e-2, min_sample_split=2,
                 min_leaf_node=1, min_child_weight=1, l1=0, l2=0, valid_features=None, random_seed=42) -> None:
        super().__init__(feature_binning_dict, min_impurity_split, min_sample_split, min_leaf_node, min_child_weight, l1, l2, valid_features)
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def _get_host_splits(self, ctx, histogram, cur_layer_node):
        pass

    def _guest_split(self, ctx, histogram, cur_layer_node):
        
        guest_best_splits = self._split(ctx, histogram, cur_layer_node)
        host_best_splits = self._get_host_splits(ctx, histogram, cur_layer_node)

    def _host_prepare(self, histogram):
        to_send_hist = []
        pos_map = []
        # prepare host split points
        for node_hist in histogram:
            g, h, cnt = node_hist
            shape = g.shape
            pos_map_ = {}
            g[self.hist_mask] = np.nan
            h[self.hist_mask] = np.nan
            # cnt is int, cannot use np.nan as mask
            g, h, cnt = g.flatten(), h.flatten(), cnt.flatten()
            random_shuffle_idx = np.random.permutation(len(g))
            g = g[random_shuffle_idx]
            h = h[random_shuffle_idx]
            cnt = cnt[random_shuffle_idx]
            to_send_hist.append([g, h, cnt])
            for i in random_shuffle_idx:
                pos_map_[i] = (i // shape[0], i % shape[1])
            pos_map.append(pos_map_)
        return to_send_hist, pos_map

    def _host_split(self, ctx, histogram, cur_layer_node):
        to_send_hist, pos_map = self._host_prepare(histogram)
        return to_send_hist, pos_map

    def split(self, ctx: Context, histogram, cur_layer_node):
        
        if ctx.is_on_guest:
            return self._guest_split(ctx, histogram, cur_layer_node)
        elif ctx.is_on_host:
            return self._host_split(ctx, histogram, cur_layer_node)
        else:
            raise ValueError('illegal role {}'.format(ctx.role))