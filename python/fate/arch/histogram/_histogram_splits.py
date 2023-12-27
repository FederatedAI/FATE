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

import logging
import typing
from typing import List, Tuple

from .values import HistogramValuesContainer
from .indexer import Shuffler

logger = logging.getLogger(__name__)


class HistogramSplits:
    def __init__(self, sid, num_node, start, end, data):
        self.sid = sid
        self.num_node = num_node
        self.start = start
        self.end = end
        self._data: HistogramValuesContainer = data

    def __str__(self):
        result = f"{self.__class__.__name__}(start={self.start}, end={self.end}):\n"
        result += str(self._data)
        return result

    def __repr__(self):
        return self.__str__()

    def iadd(self, other: "HistogramSplits"):
        self._data.iadd(other._data)
        return self

    def i_sub_on_key(self, from_key, to_key):
        self._data.i_sub_on_key(from_key, to_key)
        return self

    def compute_child_splits(
        self: "HistogramSplits", weak_child_splits: "HistogramSplits", mapping: List[Tuple[int, int, int, int]]
    ):
        assert len(mapping) == weak_child_splits.num_node
        assert self.end == weak_child_splits.end
        assert self.start == weak_child_splits.start
        assert self.sid == weak_child_splits.sid
        size = self.end - self.start
        positions = []
        for parent_pos, weak_child_pos, target_weak_child_pos, target_strong_child_pos in mapping:
            target_weak_child_start = target_weak_child_pos * size
            target_weak_child_end = (target_weak_child_pos + 1) * size
            target_strong_child_start = target_strong_child_pos * size
            target_strong_child_end = (target_strong_child_pos + 1) * size
            parent_data_start = parent_pos * size
            parent_data_end = (parent_pos + 1) * size
            weak_child_data_start = weak_child_pos * size
            weak_child_data_end = (weak_child_pos + 1) * size
            positions.append(
                (
                    target_weak_child_start,
                    target_weak_child_end,
                    target_strong_child_start,
                    target_strong_child_end,
                    parent_data_start,
                    parent_data_end,
                    weak_child_data_start,
                    weak_child_data_end,
                )
            )
        data = self._data.compute_child(weak_child_splits._data, positions, size * len(mapping) * 2)
        return HistogramSplits(self.sid, 2 * weak_child_splits.num_node, self.start, self.end, data)

    def i_decrypt(self, sk_map):
        self._data = self._data.decrypt(sk_map)
        return self

    def decrypt(self, sk_map):
        data = self._data.decrypt(sk_map)
        return HistogramSplits(self.sid, self.num_node, self.start, self.end, data)

    def i_decode(self, coder_map):
        self._data = self._data.decode(coder_map)
        return self

    def i_unpack_decode(self, coder_map, squeezed):
        unpacker_map = {}
        for name, (coder, gh_pack_num, offset_bit, precision, squeeze_num) in coder_map.items():
            if squeezed:
                pack_num = gh_pack_num * squeeze_num
            else:
                pack_num = gh_pack_num
            total_num = (self.end - self.start) * self.num_node * gh_pack_num
            unpacker_map[name] = (coder, pack_num, offset_bit, precision, total_num, gh_pack_num)
        self._data = self._data.unpack_decode(unpacker_map)
        return self

    def i_squeeze(self, squeeze_map):
        self._data.i_squeeze(squeeze_map)
        return self

    def i_shuffle(self, seed, reverse=False):
        shuffler = Shuffler(self.num_node, self.end - self.start, seed)
        self._data.i_shuffle(shuffler, reverse=reverse)
        return self

    def shuffle(self, seed, reverse=False):
        shuffler = Shuffler(self.num_node, self.end - self.start, seed)
        data = self._data.shuffle(shuffler, reverse=reverse)
        return HistogramSplits(self.sid, self.num_node, self.start, self.end, data)

    @classmethod
    def cat(cls, splits: typing.List["HistogramSplits"]) -> "HistogramValuesContainer":
        chunks_info = []
        chunks_values = []
        for split in splits:
            chunks_info.append((split.num_node, split.end - split.start))
            chunks_values.append(split._data)
        data = HistogramValuesContainer.cat(chunks_info, chunks_values)
        return data
