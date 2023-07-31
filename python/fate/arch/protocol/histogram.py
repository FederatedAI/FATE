import typing

import numpy as np
import torch

from .paillier import PK, SK, Coder, ops


class HistogramIndexer:
    def __init__(self, feature_bin_sizes):
        feature_size = len(feature_bin_sizes)
        self.feature_axis_stride = np.cumsum([0] + [feature_bin_sizes[i] for i in range(feature_size)])
        self.node_axis_stride = sum(feature_bin_sizes)

    def get_position(self, nid, fid, bid):
        return nid * self.node_axis_stride + self.feature_axis_stride[fid] + bid


class HistogramValues:
    @classmethod
    def zeros(cls, pk, size: int, stride: int = 1):
        raise NotImplementedError

    def iadd_slice(self, index, value):
        raise NotImplementedError

    def iadd(self, other):
        raise NotImplementedError

    def get_stride(self, index):
        raise NotImplementedError

    def chunking_sum(self, intervals: typing.List[typing.Tuple[int, int]]):
        raise NotImplementedError

    def decrypt(self, sk):
        raise NotImplementedError

    def i_chunking_cumsum(self, chunk_sizes: typing.List[int]):
        raise NotImplementedError

    def decode(self, coder, dtype):
        raise NotImplementedError


class HistogramEncryptedValues(HistogramValues):
    def __init__(self, pk: PK, data, stride=1):
        self.stride = stride
        self.data = data
        self.pk = pk

    @classmethod
    def zeros(cls, pk, size: int, stride: int = 1):
        return cls(pk, ops.zeros(size * stride), stride)

    def iadd_slice(self, index, value):
        ops.i_add(self.pk, self.data, value, index * self.stride)
        return self

    def iadd(self, other):
        ops.i_add(self.pk, self.data, other.data)
        return self

    def get_stride(self, index):
        return ops.slice(self.data, index * self.stride, self.stride)

    def chunking_sum(self, intervals: typing.List[typing.Tuple[int, int]]):
        """
        sum bins in the given logical intervals
        """
        intervals = [(start * self.stride, end * self.stride) for start, end in intervals]
        data = ops.intervals_sum_with_step(self.pk, self.data, intervals, self.stride)
        return HistogramEncryptedValues(self.pk, data, self.stride)

    def decrypt(self, sk: "SK"):
        data = sk.decrypt_to_encoded(self.data)
        return HistogramEncodedValues(data, self.stride)

    def i_chunking_cumsum(self, chunk_sizes: typing.List[int]):
        chunk_sizes = [num * self.stride for num in chunk_sizes]
        ops.chunking_cumsum_with_step(self.pk, self.data, chunk_sizes, self.stride)
        return self

    def __str__(self):
        return str(self.data)


class HistogramEncodedValues(HistogramValues):
    def __init__(self, data, stride=1):
        self.data = data
        self.stride = stride

    def decode_f64(self, coder: "Coder"):
        return HistogramPlainValues(coder.decode_f64_vec(self.data), self.stride)

    def decode_i64(self, coder: "Coder"):
        return HistogramPlainValues(coder.decode_i64_vec(self.data), self.stride)

    def decode_f32(self, coder: "Coder"):
        return HistogramPlainValues(coder.decode_f32_vec(self.data), self.stride)

    def decode_i32(self, coder: "Coder"):
        return HistogramPlainValues(coder.decode_i32_vec(self.data), self.stride)

    def decode(self, coder: "Coder", dtype):
        if dtype == torch.float64:
            return self.decode_f64(coder)
        elif dtype == torch.float32:
            return self.decode_f32(coder)
        elif dtype == torch.int64:
            return self.decode_i64(coder)
        elif dtype == torch.int32:
            return self.decode_i32(coder)
        else:
            raise NotImplementedError

    def get_stride(self, index):
        return self.data.get_stride(index, self.stride)


class HistogramPlainValues(HistogramValues):
    def __init__(self, data, stride=1):
        self.data = data
        self.stride = stride

    @classmethod
    def zeros(cls, size, stride, dtype=torch.float64):
        return cls(torch.zeros(size * stride, dtype=dtype), stride)

    def get_stride(self, index):
        return self.data[index * self.stride : index * self.stride + self.stride]

    def iadd_slice(self, index, value):
        start = index * self.stride
        end = index * self.stride + len(value)
        self.data[start:end] += value

    def iadd(self, other):
        self.data += other.data

    def i_chunking_cumsum(self, chunk_sizes: typing.List[int]):
        data_view = self.data.view(-1, self.stride)
        start = 0
        for num in chunk_sizes:
            data_view[start : start + num, :] = data_view[start : start + num, :].cumsum(dim=0)
            start += num

    def chunking_sum(self, intervals: typing.List[typing.Tuple[int, int]]):
        result = torch.zeros(len(intervals) * self.stride, dtype=self.data.dtype)
        data_view = self.data.view(-1, self.stride)
        for i, (start, end) in enumerate(intervals):
            result[i * self.stride : (i + 1) * self.stride] = data_view[start:end, :].sum(dim=0)
        return HistogramPlainValues(result, self.stride)

    def __str__(self):
        return str(self.data)


class Histogram:
    def __init__(self, node_size, feature_bin_sizes):
        self.node_size = node_size
        self.feature_bin_sizes = feature_bin_sizes
        self._indexer = HistogramIndexer(feature_bin_sizes)
        self._num_data_unit = self.node_size * self._indexer.node_axis_stride

        self._values_mapping: typing.MutableMapping[str, HistogramValues] = {}

    def set_value_schema(self, schema: dict):
        for name, items in schema.items():
            stride = items.get("stride", 1)
            if items["type"] == "paillier":
                pk = items["pk"]
                self._values_mapping[name] = HistogramEncryptedValues.zeros(pk, self._num_data_unit, stride)
            elif items["type"] == "tensor":
                dtype = items.get("dtype", torch.float64)
                self._values_mapping[name] = HistogramPlainValues.zeros(
                    self._num_data_unit, stride=stride, dtype=dtype
                )
            else:
                raise NotImplementedError

    def update(self, nids, fids, targets):
        for nid, bins, target in zip(nids, fids, targets):
            for fid, bid in enumerate(bins):
                index = self._indexer.get_position(nid, fid, bid)
                for name, value in target.items():
                    self._values_mapping[name].iadd_slice(index, value)
        return self

    def merge(self, hist: "Histogram"):
        for name, value_container in hist._values_mapping.items():
            if name in self._values_mapping:
                self._values_mapping[name].iadd(value_container)
            else:
                self._values_mapping[name] = value_container
        return self

    def decrypt(self, sk_map: dict):
        result = Histogram(self.node_size, self.feature_bin_sizes)
        for name, value_container in self._values_mapping.items():
            if name in sk_map:
                result._values_mapping[name] = value_container.decrypt(sk_map[name])
            else:
                result._values_mapping[name] = value_container
        return result

    def decode(self, coder_map: dict):
        result = Histogram(self.node_size, self.feature_bin_sizes)
        for name, value_container in self._values_mapping.items():
            if name in coder_map:
                coder, dtype = coder_map[name]
                result._values_mapping[name] = value_container.decode(coder, dtype)
            else:
                result._values_mapping[name] = value_container
        return result

    def __str__(self):
        result = ""
        for nid in range(self.node_size):
            result += f"node-{nid}:\n"
            for fid in range(len(self.feature_bin_sizes)):
                result += f"\tfeature-{fid}:\n"
                for bid in range(self.feature_bin_sizes[fid]):
                    for name, value_container in self._values_mapping.items():
                        values = value_container.get_stride(self._indexer.get_position(nid, fid, bid))
                        result += f"\t\t{name}: {values}\t"
                    result += "\n"
        return result

    def flatten_all_feature_bins(self):
        """
        flatten all feature bins into one bin

        Note: this method will change the histogram data itself, any inplace ops on the original histogram will
        affect the result of this method and vice versa.
        :return:
        """
        result = Histogram(self.node_size, [sum(self.feature_bin_sizes)])
        for name, value_container in self._values_mapping.items():
            result._values_mapping[name] = value_container
        return result

    def cumsum_bins(self):
        feature_bin_nums = []
        for nid in range(self.node_size):
            feature_bin_nums.extend(self.feature_bin_sizes)
        for name, value_container in self._values_mapping.items():
            value_container.i_chunking_cumsum(feature_bin_nums)

    def sum_bins(self):
        result = Histogram(self.node_size, [1] * len(self.feature_bin_sizes))
        intervals = []
        for nid in range(self.node_size):
            for fid in range(len(self.feature_bin_sizes)):
                intervals.append(
                    (
                        self._indexer.get_position(nid, fid, 0),
                        self._indexer.get_position(nid, fid, self.feature_bin_sizes[fid]),
                    )
                )
        for name, value_container in self._values_mapping.items():
            result._values_mapping[name] = value_container.chunking_sum(intervals)
        return result
