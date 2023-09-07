import logging
import typing
from typing import List, Tuple

from ..indexer import Shuffler

logger = logging.getLogger(__name__)


class HistogramValues:
    def iadd_slice(self, value, sa, sb, size):
        raise NotImplementedError

    def i_update(self, value, positions):
        raise NotImplementedError

    def i_update_with_masks(self, value, positions, masks):
        raise NotImplementedError

    def iadd(self, other):
        raise NotImplementedError

    def chunking_sum(self, intervals: typing.List[typing.Tuple[int, int]]):
        raise NotImplementedError

    def compute_child(self, weak_child: "HistogramValues", positions: List[Tuple[int, int, int, int, int, int]], size):
        ...

    def intervals_slice(self, intervals: typing.List[typing.Tuple[int, int]]):
        raise NotImplementedError

    def i_shuffle(self, shuffler: "Shuffler", reverse=False):
        raise NotImplementedError

    def shuffle(self, shuffler: "Shuffler", reverse=False):
        raise NotImplementedError

    def slice(self, start, end):
        raise NotImplementedError

    def decrypt(self, sk):
        raise NotImplementedError

    def squeeze(self, pack_num, offset_bit):
        raise NotImplementedError

    def unpack(self, coder, pack_num, offset_bit, precision, total_num, stride):
        raise NotImplementedError

    def i_chunking_cumsum(self, chunk_sizes: typing.List[int]):
        raise NotImplementedError

    def decode(self, coder, dtype):
        raise NotImplementedError

    def cat(self, chunks_info, values):
        raise NotImplementedError(f"{self.__class__.__name__}.cat")

    def extract_node_data(self, node_data_size, node_size):
        raise NotImplementedError
