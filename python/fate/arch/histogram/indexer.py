from typing import List, Tuple

import numpy as np
import torch


class Shuffler:
    """
    Shuffler is used to shuffle the data in the same way for all partition.


    """

    def __init__(self, num_node: int, node_size: int, seed: int):
        self.num_node = num_node
        self.node_size = node_size
        self.perm_indexes = [
            torch.randperm(node_size, generator=torch.Generator().manual_seed(seed)) for _ in range(num_node)
        ]

    def get_global_perm_index(self):
        index = torch.hstack([index + (nid * self.node_size) for nid, index in enumerate(self.perm_indexes)])
        return index

    #
    # def reverse_index(self, index):
    #     return torch.argsort(self.perm_index)[index]

    def get_shuffle_index(self, step, reverse=False):
        """
        get chunk shuffle index
        """
        stepped = torch.arange(0, self.num_node * self.node_size * step).reshape(self.num_node * self.node_size, step)
        indexes = stepped[self.get_global_perm_index(), :].flatten()
        if reverse:
            indexes = torch.argsort(indexes)
        return indexes.detach().cpu().tolist()

    def get_reverse_indexes(self, step, indexes):
        mapping = self.get_shuffle_index(step, reverse=False)
        return [mapping[i] for i in indexes]


class HistogramIndexer:
    """
    HistogramIndexer is used to index the `VectorizedHistogram` data.

    The data is stored in a flatten array with the shape is (num_node * sum(feature_bin_sizes)) and the
    data is indexed by (node_id, feature_id, bin_id) in the flatten array.
    At logical level, the data is indexed by (node_id, feature_id, bin_id) likes:

    node_id_0:
        feature_id_0:
            bin_id_0: data
            bin_id_1: data
            ...
        feature_id_1:
            bin_id_0: data
            bin_id_1: data
            ...
        ...
    node_id_1:
        feature_id_0:
            bin_id_0: data
            bin_id_1: data
            ...
        feature_id_1:
            bin_id_0: data
            bin_id_1: data
            ...
        ...
    ...
    notice that the data is stored in the flatten array, so the index is calculated by:
        position = node_id * sum(feature_bin_sizes) + feature_bin_sizes[feature_id] + bin_id

    which means the bin_size of each feature is not necessary to be the same but the sum of all feature_bin_sizes
    should be the same.

    Notes: This class will be rewritten in rust in the future.
    """

    def __init__(self, node_size: int, feature_bin_sizes: List[int]):
        self.node_size = node_size
        self.feature_bin_sizes = feature_bin_sizes
        self.feature_size = len(feature_bin_sizes)
        self.feature_axis_stride = np.cumsum([0] + [feature_bin_sizes[i] for i in range(self.feature_size)])
        self.node_axis_stride = sum(feature_bin_sizes)

        self._shuffler = None

    def get_position(self, nid: int, fid: int, bid: int):
        """
        get data position by node_id, feature_id, bin_id
        Args:
            nid: node id
            fid: feature id
            bid: bin id

        Returns: data position
        """
        return nid * self.node_axis_stride + self.feature_axis_stride[fid] + bid

    def get_positions(self, nids: List[int], bids: List[List[int]]):
        """
        get data positions by node_ids and bin_ids
        Args:
            nids: node ids
            bids: bin ids

        Returns: data positions
        """
        positions = []
        for nid, bids in zip(nids, bids):
            positions.append([self.get_position(nid, fid, bid) for fid, bid in enumerate(bids)])
        return positions

    def get_reverse_position(self, position) -> Tuple[int, int, int]:
        """
        get node_id, feature_id, bin_id by data position
        Args:
            position: data position

        Returns: node_id, feature_id, bin_id
        """
        nid = position // self.node_axis_stride
        bid = position % self.node_axis_stride
        for fid in range(self.feature_size):
            if bid < self.feature_axis_stride[fid + 1]:
                return nid, fid, bid - self.feature_axis_stride[fid]

    def get_node_intervals(self):
        intervals = []
        for nid in range(self.node_size):
            intervals.append((nid * self.node_axis_stride, (nid + 1) * self.node_axis_stride))
        return intervals

    def get_feature_position_ranges(self):
        """
        get feature position and size
        Returns: list of (feature_position, feature_bin_size)
        """
        intervals = []
        for nid in range(self.node_size):
            node_stride = nid * self.node_axis_stride
            for fid in range(self.feature_size):
                intervals.append(
                    (node_stride + self.feature_axis_stride[fid], node_stride + self.feature_axis_stride[fid + 1])
                )
        return intervals

    def splits_into_k(self, k: int):
        n = self.node_axis_stride
        split_sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
        start = 0
        for pid, size in enumerate(split_sizes):
            end = start + size
            shift = self.node_axis_stride
            yield pid, (start, end), [(start + nid * shift, end + nid * shift) for nid in range(self.node_size)]
            start += size

    def total_data_size(self):
        return self.node_size * self.node_axis_stride

    def one_node_data_size(self):
        return self.node_axis_stride

    def global_flatten_bin_sizes(self):
        return self.feature_bin_sizes * self.node_size

    def flatten_in_node(self):
        return HistogramIndexer(self.node_size, [self.one_node_data_size()])

    def squeeze_bins(self):
        return HistogramIndexer(self.node_size, [1] * self.feature_size)

    def reshape(self, feature_bin_sizes):
        return HistogramIndexer(self.node_size, feature_bin_sizes)

    def get_shuffler(self, seed):
        if self._shuffler is None:
            self._shuffler = Shuffler(self.node_size, self.one_node_data_size(), seed)
        return self._shuffler

    def unflatten_indexes(self):
        indexes = {}
        for nid in range(self.node_size):
            indexes[nid] = {}
            for fid in range(self.feature_size):
                indexes[nid][fid] = []
                for bid in range(self.feature_bin_sizes[fid]):
                    indexes[nid][fid].append(self.get_position(nid, fid, bid))
        return indexes
