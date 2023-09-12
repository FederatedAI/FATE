import logging
import typing
from typing import List, Tuple

import torch

from ._value import HistogramValues
from ..indexer import Shuffler

logger = logging.getLogger(__name__)


class HistogramPlainValues(HistogramValues):
    def __init__(self, data, dtype: torch.dtype, size: int, stride: int):
        self.data = data
        self.dtype = dtype
        self.size = size
        self.stride = stride

    def __str__(self):
        return f"<Plain: {self.data.reshape(-1, self.stride)}>"

    def __repr__(self):
        return str(self)

    @classmethod
    def zeros(cls, size, stride, dtype=torch.float64):
        return cls(torch.zeros(size * stride, dtype=dtype), dtype, size, stride)

    def intervals_slice(self, intervals: typing.List[typing.Tuple[int, int]]):
        size = sum(e - s for s, e in intervals)
        result = torch.zeros(size * self.stride, dtype=self.data.dtype)
        start = 0
        for s, e in intervals:
            end = start + (e - s) * self.stride
            result[start:end] = self.data[s * self.stride : e * self.stride]
            start = end
        return HistogramPlainValues(result, self.dtype, size, self.stride)

    def iadd_slice(self, value, sa, sb, size):
        size = size * self.stride
        value = value.view(-1)
        self.data[sa : sa + size] += value[sb : sb + size]

    def slice(self, start, end):
        return HistogramPlainValues(
            self.data[start * self.stride : end * self.stride], self.dtype, end - start, self.stride
        )

    def iadd(self, other):
        self.data += other.data

    def i_update(self, value, positions):
        if self.stride == 1:
            index = torch.LongTensor(positions)
            value = value.view(-1, 1).expand(-1, index.shape[1]).flatten()
            index = index.flatten()
            data = self.data
        else:
            index = torch.LongTensor(positions)
            data = self.data.view(-1, self.stride)
            value = (
                value.view(-1, self.stride)
                .unsqueeze(1)
                .expand(-1, index.shape[1], self.stride)
                .reshape(-1, self.stride)
            )
            index = index.flatten().unsqueeze(1).expand(-1, self.stride)
        if self.data.dtype != value.dtype:
            logger.warning(f"update value dtype {value.dtype} is not equal to data dtype {self.data.dtype}")
            value = value.to(data.dtype)
        data.scatter_add_(0, index, value)

    def i_update_with_masks(self, value, positions, masks):
        if self.stride == 1:
            value = value[masks]
            index = torch.LongTensor(positions)
            value = value.view(-1, 1).expand(-1, index.shape[1]).flatten()
            index = index.flatten()
            data = self.data
        else:
            index = torch.LongTensor(positions)
            data = self.data.view(-1, self.stride)
            value = value.view(-1, self.stride)[masks]
            value = value.unsqueeze(1).expand(-1, index.shape[1], self.stride).reshape(-1, self.stride)
            index = index.flatten().unsqueeze(1).expand(-1, self.stride)
        if self.data.dtype != value.dtype:
            logger.warning(f"update value dtype {value.dtype} is not equal to data dtype {self.data.dtype}")
            value = value.to(data.dtype)
        data.scatter_add_(0, index, value)

    def i_shuffle(self, shuffler: "Shuffler", reverse=False):
        indices = shuffler.get_shuffle_index(step=self.stride, reverse=reverse)
        self.data = self.data[indices]

    def shuffle(self, shuffler: "Shuffler", reverse=False):
        indices = shuffler.get_shuffle_index(step=self.stride, reverse=reverse)
        data = self.data[indices]
        return HistogramPlainValues(data, self.dtype, self.size, self.stride)

    def i_chunking_cumsum(self, chunk_sizes: typing.List[int]):
        data_view = self.data.view(-1, self.stride)
        start = 0
        for num in chunk_sizes:
            data_view[start : start + num, :] = data_view[start : start + num, :].cumsum(dim=0)
            start += num

    def chunking_sum(self, intervals: typing.List[typing.Tuple[int, int]]):
        size = len(intervals)
        result = torch.zeros(size * self.stride, dtype=self.data.dtype)
        data_view = self.data.view(-1, self.stride)
        for i, (start, end) in enumerate(intervals):
            result[i * self.stride : (i + 1) * self.stride] = data_view[start:end, :].sum(dim=0)
        return HistogramPlainValues(result, self.dtype, size, self.stride)

    def compute_child(
        self, weak_child: "HistogramPlainValues", positions: List[Tuple[int, int, int, int, int, int, int, int]], size
    ):
        data = torch.zeros(size * self.stride, dtype=self.data.dtype)
        data_view = data.view(-1, self.stride)

        parent_data_view = self.data.view(-1, self.stride)
        weak_child_data_view = weak_child.data.view(-1, self.stride)

        for (
            target_weak_child_start,
            target_weak_child_end,
            target_strong_child_start,
            target_strong_child_end,
            parent_data_start,
            parent_data_end,
            weak_child_data_start,
            weak_child_data_end,
        ) in positions:
            # copy data from weak child to correct position
            data_view[target_weak_child_start:target_weak_child_end] = weak_child_data_view[
                weak_child_data_start:weak_child_data_end
            ]
            # compute strong child data
            data_view[target_strong_child_start:target_strong_child_end] = (
                parent_data_view[parent_data_start:parent_data_end]
                - weak_child_data_view[weak_child_data_start:weak_child_data_end]
            )
        return HistogramPlainValues(data, self.dtype, size, self.stride)

    @classmethod
    def cat(cls, chunks_info: List[Tuple[int, int]], values: List["HistogramPlainValues"]):
        data = []
        for (num_chunk, chunk_size), value in zip(chunks_info, values):
            data.append(value.data.reshape(num_chunk, chunk_size, value.stride))
        data = torch.cat(data, dim=1)
        size = data.shape[0]
        dtype = data.dtype
        data = data.flatten()
        return cls(data, dtype, size, values[0].stride)

    def extract_node_data(self, node_data_size, node_size):
        return list(self.data.reshape(node_size, node_data_size, self.stride))
