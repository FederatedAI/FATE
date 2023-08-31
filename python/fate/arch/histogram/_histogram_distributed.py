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
        split.i_decrypt(sk_map)
        if unpacker_map is not None:
            split.i_unpack_decode(unpacker_map, squeezed)
            return split
        else:
            split.i_decode(coder_map)
            return split

    return _decrypt


class DistributedHistogram:
    def __init__(self, table: CTableABC[int, HistogramSplits], node_size, node_data_size, squeezed=False):
        self._table = table
        self._node_size = node_size
        self._node_data_size = node_data_size
        self._squeezed = squeezed

    def i_squeeze(self, squeeze_map: MutableMapping[str, typing.Tuple[int, int]]):
        """
        Squeeze the histogram values.

        Args:
            squeeze_map: name -> (pack_num, offset_bit)
        """
        self._table = self._table.mapValues(lambda split: split.i_squeeze(squeeze_map))
        self._squeezed = True

    def i_shuffle(self, seed, reverse=False):
        """
        Shuffle the histogram values.

        Args:
            seed: random seed, if None, do not shuffle
            reverse: if reverse the shuffle
        """
        if seed is None:
            return
        self._table = self._table.mapValues(lambda split: split.i_shuffle(seed, reverse))

    def decrypt(
            self,
            sk_map: MutableMapping[str, typing.Any],
            coder_map: MutableMapping[str, typing.Tuple[typing.Any, torch.dtype]],
            unpacker_map: Optional[MutableMapping[str, typing.Tuple[typing.Any, int, int, int, int, int]]] = None,
    ):
        out = list(self._table.mapValues(_decrypt_func(sk_map, coder_map, self._squeezed, unpacker_map)).collect())
        out.sort(key=lambda x: x[0])
        data = HistogramSplits.cat([split for _, split in out])
        return Histogram(HistogramIndexer(self._node_size, [self._node_data_size]), data)

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
        assert self._node_size == weak_child._node_size
        assert self._node_data_size == weak_child._node_data_size
        splits = self._table.join(weak_child._table, lambda x, y: x.compute_child_splits(y, mapping))
        return DistributedHistogram(splits, self._node_size * 2, self._node_data_size)

    def recover_feature_bins(
            self, feature_bin_sizes, seed, split_points: typing.Dict[int, int]
    ) -> typing.Dict[int, typing.Tuple[int, int]]:
        """
        Recover the feature bins from the split points.

        Args:
            seed: random seed
            split_points: nid -> split data index

        Returns:
            nid -> (fid, bid)
        """
        indexer = HistogramIndexer(self._node_size, feature_bin_sizes)
        points = list(split_points.items())
        # reverse index from split shuffle
        shuffler = Shuffler(self._node_size, self._node_data_size, seed)

        real_indexes = shuffler.get_reverse_indexes(step=1, indexes=[p[1] for p in points])
        real_fid_bid = {}
        for (nid, _), index in zip(points, real_indexes):
            _, fid, bid = indexer.get_reverse_position(index)
            real_fid_bid[nid] = (fid, bid)
        return real_fid_bid

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
