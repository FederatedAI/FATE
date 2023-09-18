import logging
import typing
from typing import List, MutableMapping, Tuple, Optional

import torch

from fate.arch.abc import CTableABC
from ._histogram_local import Histogram
from ._histogram_splits import HistogramSplits
from .indexer import HistogramIndexer, Shuffler

logger = logging.getLogger(__name__)


def _decrypt_func(sk_map, coder_map, squeezed, unpacker_map):
    def _decrypt(split: HistogramSplits):
        split = split.decrypt(sk_map)
        if unpacker_map is not None:
            split.i_unpack_decode(unpacker_map, squeezed)
            return split
        else:
            split.i_decode(coder_map)
            return split

    return _decrypt


class DistributedHistogram:
    def __init__(
        self,
        splits: CTableABC[int, HistogramSplits],
        k,
        node_size,
        node_data_size,
        global_seed,
        seed=None,
        squeezed=False,
        shuffled=False,
    ):
        self._splits = splits
        self._k = k
        self._node_size = node_size
        self._node_data_size = node_data_size
        self._squeezed = squeezed
        self._shuffled = shuffled
        self._seed = seed
        self._global_seed = global_seed

    def __getstate__(self):
        """
        Get the state for pickle.

        remove sensitive data before sending to other parties, such as:
            - global_seed
        """
        return self._splits, self._k, self._node_size, self._node_data_size, self._squeezed, self._shuffled

    def __setstate__(self, state):
        self._splits, self._k, self._node_size, self._node_data_size, self._squeezed, self._shuffled = state

    def i_squeeze(self, squeeze_map: MutableMapping[str, typing.Tuple[int, int]]):
        """
        Squeeze the histogram values.

        Args:
            squeeze_map: name -> (pack_num, offset_bit)
        """
        self._splits = self._splits.mapValues(lambda split: split.i_squeeze(squeeze_map))
        self._squeezed = True

    def i_shuffle_splits(self):
        """
        Shuffle the histogram splits values.
        """
        seed = self._seed
        if seed is None:
            return
        self._splits = self._splits.mapValues(lambda split: split.i_shuffle(seed, False))
        self._shuffled = True

    def shuffle_splits(self):
        """
        Shuffle the histogram splits values, return a new DistributedHistogram.
        """
        seed = self._seed
        if seed is None:
            return self
        splits = self._splits.mapValues(lambda split: split.shuffle(seed, False))
        return DistributedHistogram(
            splits, self._k, self._node_size, self._node_data_size, self._global_seed, self._seed, self._squeezed, True
        )

    def compute_child(self, weak_child: "DistributedHistogram", mapping: List[Tuple[int, int, int, int]]):
        """
        Compute the child histogram.

        Args:
            weak_child: the splits of one child
            mapping: the mapping from parent to child,
                the mapping is a list of (parent_pos, weak_child_pos, target_weak_child_pos, target_strong_child_pos)
                which means in logic:
                    output[target_weak_child_pos] = weak_child[weak_child_pos]
                    output[target_strong_child_pos] = self[parent_pos] - weak_child[weak_child_pos]
        Examples:
            # tree structure:
            #              -1
            #        0            1
            #    2       3     4      5           <-- parent nodes, node #4 is leaf node
            #  6  *7  *8  9        *10 11         <-- child nodes, node #7, #8, #10 are weak child nodes
            #
            # pos  parent_node  weak_child_node  output_node
            #  0   #2           #7               #6
            #  1   #3           #8               #7
            #  2   #4           #10              #8
            #  3   #5                            #9
            #  4                                 #10
            #  5                                 #11
            >>> parent = DistributedHistogram(...) # data for nodes stored in order [#2, #3, #4, #5]
            >>> weak_child = DistributedHistogram(...) # data for nodes stored in order [#7, #8, #10]
            >>> mapping = [
            >>>     (0, 0, 1, 0),  # pos for (#2, #7, #7, #6)
            >>>     (1, 1, 2, 3),  # pos for (#3, #8, #8, #9)
            >>>     (3, 2, 4, 5)   # pos for (#5, #10, #10, #11)
            >>> ]
            >>> child = parent.compute_child(weak_child, mapping) # data for nodes stored in order [#6, #7, #8, #9, #10, #11]
        """
        # assert self._node_size == weak_child._node_size, 'node size not match, {} != {}'.format(
        #     self._node_size, weak_child._node_size
        # )
        assert self._node_data_size == weak_child._node_data_size
        splits = self._splits.join(weak_child._splits, lambda x, y: x.compute_child_splits(y, mapping))
        return DistributedHistogram(
            splits,
            weak_child._k,
            len(mapping) * 2,
            weak_child._node_data_size,
            weak_child._global_seed,
            weak_child._seed,
        )

    def i_sub_on_key(self, from_key: str, to_key: str):
        """
        Subtract the histogram splits values on the given key.

        Args:
            from_key: the start key
            to_key: the end key
        """
        self._splits = self._splits.mapValues(lambda split: split.i_sub_on_key(from_key, to_key))

    def recover_feature_bins(
        self, feature_bin_sizes, split_points: typing.Dict[int, int]
    ) -> typing.Dict[int, typing.Tuple[int, int]]:
        """
        Recover the feature bins from the split points.

        Args:
            feature_bin_sizes: the feature bin sizes
            split_points: nid -> split data index

        Returns:
            nid -> (fid, bid)
        """

        if self._shuffled:
            split_points = self._recover_from_split_shuffle(split_points)

        split_points = self._recover_from_global_shuffle(split_points)
        return self._recover_histogram_position(split_points, feature_bin_sizes)

    def _recover_histogram_position(
        self, split_points: typing.Dict[int, int], feature_bin_sizes
    ) -> typing.Dict[int, typing.Tuple[int, int]]:
        fid_bid = {}
        indexer = HistogramIndexer(self._node_size, feature_bin_sizes)
        for nid, index in split_points.items():
            _, fid, bid = indexer.get_reverse_position(index)
            fid_bid[nid] = (fid, bid)
        return fid_bid

    def _recover_from_global_shuffle(self, split_points: MutableMapping[int, int]):
        if self._global_seed is None:
            return split_points
        shuffler = Shuffler(self._node_size, self._node_data_size, self._global_seed)
        points = list(split_points.items())
        real_indexes = shuffler.get_reverse_indexes(step=1, indexes=[p[1] for p in points])
        out = {}
        for (nid, _), index in zip(points, real_indexes):
            out[nid] = index
        return out

    def _recover_from_split_shuffle(self, split_points):
        splits_info = list(self._splits_into_k(self._node_data_size, self._k))
        _size_mapping = {}
        out = {}
        for nid in split_points:
            index = split_points[nid]
            for split_info in splits_info:
                if split_info[0] <= index < split_info[1]:
                    if split_info[1] - split_info[0] not in _size_mapping:
                        _size_mapping[split_info[1] - split_info[0]] = [(split_info, nid, index - split_info[0])]
                    else:
                        _size_mapping[split_info[1] - split_info[0]].append((split_info, nid, index - split_info[0]))
        for size in _size_mapping:
            shuffler = Shuffler(self._node_size, size, self._seed)
            fixed_size_splits = _size_mapping[size]
            for (split_info, nid, _), i in zip(
                fixed_size_splits, shuffler.get_reverse_indexes(step=1, indexes=[p[2] for p in fixed_size_splits])
            ):
                out[nid] = split_info[0] + i
        return out

    @staticmethod
    def _splits_into_k(n, k: int):
        d, r = divmod(n, k)
        start = 0
        for _ in range(k):
            end = start + d + (r > 0)
            yield start, end
            start = end
            r -= 1

    def decrypt(
        self,
        sk_map: MutableMapping[str, typing.Any],
        coder_map: MutableMapping[str, typing.Tuple[typing.Any, torch.dtype]],
        unpacker_map: Optional[MutableMapping[str, typing.Tuple[typing.Any, int, int, int, int, int]]] = None,
    ):
        out = list(self._splits.mapValues(_decrypt_func(sk_map, coder_map, self._squeezed, unpacker_map)).collect())
        out.sort(key=lambda x: x[0])
        data = HistogramSplits.cat([split for _, split in out])
        return Histogram(HistogramIndexer(self._node_size, [self._node_data_size]), data)

    # def decrypt_(self, sk_map: MutableMapping[str, typing.Any]):
    #     """
    #     Decrypt the histogram values.
    #
    #     Args:
    #         sk_map: name -> sk
    #     """
    #     table = self._table.mapValues(lambda split: split.i_decrypt(sk_map))
    #     return DistributedHistogram(table, self._node_size, self._node_data_size, self._squeezed)
    #
    # def unpack_decode(self, coder_map: MutableMapping[str, typing.Tuple[typing.Any, int, int, int, int]]):
    #     """
    #     Unpack and decode the histogram values.
    #
    #     Args:
    #         coder_map: name -> (coder, pack_num, offset_bit, precision, total_num)
    #     """
    #     table = self._table.mapValues(lambda split: split.i_unpack_decode(coder_map, self._squeezed))
    #     return DistributedHistogram(table, self._node_size, self._node_data_size)
    #
    # def decode(self, coder_map: MutableMapping[str, typing.Tuple[typing.Any, torch.dtype]]):
    #     """
    #     Decode the histogram values.
    #
    #     Args:
    #         coder_map: name -> (coder, dtype)
    #     """
    #     table = self._table.mapValues(lambda split: split.i_decode(coder_map))
    #     return DistributedHistogram(table, self._node_size, self._node_data_size)
    #
    # def union(self) -> Histogram:
    #     """
    #     Union the splits into one histogram.
    #     """
    #     out = list(self._table.collect())
    #     out.sort(key=lambda x: x[0])
    #     return self.cat([split for _, split in out])
    # def cat(self, hists: typing.List["HistogramSplits"]) -> "Histogram":
    #     data = HistogramSplits.cat(hists)
    #     return Histogram(HistogramIndexer(self._node_size, [self._node_data_size]), data)
