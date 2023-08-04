import typing
from typing import List, MutableMapping, Tuple

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
        return indexes

    def get_reverse_indexes(self, step, indexes):
        mapping = self.get_shuffle_index(step, reverse=True)
        return [mapping[i] for i in indexes]


class HistogramIndexer:
    def __init__(self, node_size: int, feature_bin_sizes: List[int]):
        self.node_size = node_size
        self.feature_bin_size = feature_bin_sizes
        self.feature_size = len(feature_bin_sizes)
        self.feature_axis_stride = np.cumsum([0] + [feature_bin_sizes[i] for i in range(self.feature_size)])
        self.node_axis_stride = sum(feature_bin_sizes)

        self._shuffler = None

    def get_position(self, nid, fid, bid):
        return nid * self.node_axis_stride + self.feature_axis_stride[fid] + bid

    def get_reverse_position(self, position):
        nid = position // self.node_axis_stride
        bid = position % self.node_axis_stride
        for fid in range(self.feature_size):
            if bid < self.feature_axis_stride[fid + 1]:
                return nid, fid, bid - self.feature_axis_stride[fid]

    def get_bin_num(self, fid):
        return self.feature_bin_size[fid]

    def get_bin_interval(self, nid, fid):
        node_stride = nid * self.node_axis_stride
        return node_stride + self.feature_axis_stride[fid], node_stride + self.feature_axis_stride[fid + 1]

    def get_node_intervals(self):
        intervals = []
        for nid in range(self.node_size):
            intervals.append((nid * self.node_axis_stride, (nid + 1) * self.node_axis_stride))
        return intervals

    def get_global_feature_intervals(self):
        intervals = []
        for nid in range(self.node_size):
            for fid in range(self.feature_size):
                intervals.append(self.get_bin_interval(nid, fid))
        return intervals

    def splits_into_k(self, k):
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
        return self.feature_bin_size * self.node_size

    def flatten_in_node(self):
        return HistogramIndexer(self.node_size, [self.one_node_data_size()])

    def squeeze_bins(self):
        return HistogramIndexer(self.node_size, [1] * self.feature_size)

    def get_shuffler(self, seed):
        if self._shuffler is None:
            self._shuffler = Shuffler(self.node_size, self.one_node_data_size(), seed)
        return self._shuffler

    def reshape(self, feature_bin_sizes):
        return HistogramIndexer(self.node_size, feature_bin_sizes)


class HistogramValues:
    def iadd_slice(self, index, value):
        raise NotImplementedError

    def iadd(self, other):
        raise NotImplementedError

    def chunking_sum(self, intervals: typing.List[typing.Tuple[int, int]]):
        raise NotImplementedError

    def intervals_slice(self, intervals: typing.List[typing.Tuple[int, int]]):
        raise NotImplementedError

    def i_shuffle(self, shuffler: "Shuffler", reverse=False):
        raise NotImplementedError

    def slice(self, start, end):
        raise NotImplementedError

    def decrypt(self, sk):
        raise NotImplementedError

    def i_chunking_cumsum(self, chunk_sizes: typing.List[int]):
        raise NotImplementedError

    def decode(self, coder, dtype):
        raise NotImplementedError

    def cat(self, chunks_info, values):
        raise NotImplementedError

    def extract_node_data(self, node_data_size, node_size):
        raise NotImplementedError


class HistogramEncryptedValues(HistogramValues):
    def __init__(self, pk, evaluator, data, stride=1):
        self.stride = stride
        self.data = data
        self.pk = pk
        self.evaluator = evaluator

    @classmethod
    def zeros(cls, pk, evaluator, size: int, stride: int = 1):
        return cls(pk, evaluator, evaluator.zeros(size * stride), stride)

    def iadd_slice(self, index, value):
        from fate.arch.tensor.phe import PHETensor

        if isinstance(value, PHETensor):
            value = value.data
        self.evaluator.i_add(self.pk, self.data, value, index * self.stride)
        return self

    def iadd(self, other):
        self.evaluator.i_add(self.pk, self.data, other.data)
        return self

    def slice(self, start, end):
        return HistogramEncryptedValues(
            self.pk,
            self.evaluator,
            self.evaluator.slice(self.data, start * self.stride, (end - start) * self.stride),
            self.stride,
        )

    def intervals_slice(self, intervals: typing.List[typing.Tuple[int, int]]) -> "HistogramEncryptedValues":
        intervals = [(start * self.stride, end * self.stride) for start, end in intervals]
        data = self.evaluator.intervals_slice(self.data, intervals)
        return HistogramEncryptedValues(self.pk, self.evaluator, data, self.stride)

    def i_shuffle(self, shuffler: "Shuffler", reverse=False):
        indices = shuffler.get_shuffle_index(step=self.stride, reverse=reverse)
        self.evaluator.i_shuffle(self.pk, self.data, indices)
        return self

    def chunking_sum(self, intervals: typing.List[typing.Tuple[int, int]]):
        """
        sum bins in the given logical intervals
        """
        intervals = [(start * self.stride, end * self.stride) for start, end in intervals]
        data = self.evaluator.intervals_sum_with_step(self.pk, self.data, intervals, self.stride)
        return HistogramEncryptedValues(self.pk, self.evaluator, data, self.stride)

    def decrypt(self, sk):
        data = sk.decrypt_to_encoded(self.data)
        return HistogramEncodedValues(data, self.stride)

    def i_chunking_cumsum(self, chunk_sizes: typing.List[int]):
        chunk_sizes = [num * self.stride for num in chunk_sizes]
        self.evaluator.chunking_cumsum_with_step(self.pk, self.data, chunk_sizes, self.stride)
        return self

    def __str__(self):
        return f"<HistogramEncryptedValues stride={self.stride}, data={self.data}>"

    def extract_node_data(self, node_data_size, node_size):
        raise NotImplementedError


class HistogramEncodedValues(HistogramValues):
    def __init__(self, data, stride=1):
        self.data = data
        self.stride = stride

    def decode_f64(self, coder):
        return HistogramPlainValues(coder.decode_f64_vec(self.data), self.stride)

    def decode_i64(self, coder):
        return HistogramPlainValues(coder.decode_i64_vec(self.data), self.stride)

    def decode_f32(self, coder):
        return HistogramPlainValues(coder.decode_f32_vec(self.data), self.stride)

    def decode_i32(self, coder):
        return HistogramPlainValues(coder.decode_i32_vec(self.data), self.stride)

    def decode(self, coder, dtype):
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

    def slice(self, start, end):
        if hasattr(self.data, "slice"):
            return self.data.slice(start * self.stride, end * self.stride)
        else:
            return "<Encoded data=unknown>"


class HistogramPlainValues(HistogramValues):
    def __init__(self, data, stride=1):
        self.data = data
        self.stride = stride

    def __str__(self):
        return f"<Plain: {self.data.reshape(-1, self.stride)}>"

    def __repr__(self):
        return str(self)

    @classmethod
    def zeros(cls, size, stride, dtype=torch.float64):
        return cls(torch.zeros(size * stride, dtype=dtype), stride)

    def intervals_slice(self, intervals: typing.List[typing.Tuple[int, int]]):
        result = torch.zeros(sum(e - s for s, e in intervals) * self.stride, dtype=self.data.dtype)
        start = 0
        for s, e in intervals:
            end = start + (e - s) * self.stride
            result[start:end] = self.data[s * self.stride : e * self.stride]
            start = end
        return HistogramPlainValues(result, self.stride)

    def iadd_slice(self, index, value):
        start = index * self.stride
        end = index * self.stride + len(value)
        self.data[start:end] += value

    def slice(self, start, end):
        return HistogramPlainValues(self.data[start * self.stride : end * self.stride], self.stride)

    def iadd(self, other):
        self.data += other.data

    def i_shuffle(self, shuffler: "Shuffler", reverse=False):
        indices = shuffler.get_shuffle_index(step=self.stride, reverse=reverse)
        self.data = self.data[indices]

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

    @classmethod
    def cat(cls, chunks_info: List[Tuple[int, int]], values: List["HistogramPlainValues"]):
        data = []
        for (num_chunk, chunk_size), value in zip(chunks_info, values):
            data.append(value.data.reshape(num_chunk, chunk_size, value.stride))
        data = torch.cat(data, dim=1).flatten()
        return cls(data, values[0].stride)

    def extract_node_data(self, node_data_size, node_size):
        return list(self.data.reshape(node_size, node_data_size * self.stride))


class Histogram:
    def __init__(self, indexer: "HistogramIndexer", values: MutableMapping[str, "HistogramValues"]):
        self._indexer = indexer
        self._values_mapping = values

    @classmethod
    def create(cls, node_size, feature_bin_sizes, values_schema: dict):
        indexer = HistogramIndexer(node_size, feature_bin_sizes)
        values_mapping = {}
        for name, items in values_schema.items():
            stride = items.get("stride", 1)
            if items["type"] == "paillier":
                pk = items["pk"]
                evaluator = items["evaluator"]
                values_mapping[name] = HistogramEncryptedValues.zeros(pk, evaluator, indexer.total_data_size(), stride)
            elif items["type"] == "tensor":
                dtype = items.get("dtype", torch.float64)
                values_mapping[name] = HistogramPlainValues.zeros(
                    indexer.total_data_size(), stride=stride, dtype=dtype
                )
            else:
                raise NotImplementedError
        return cls(indexer, values_mapping)

    def i_update(self, fids, nids, targets):
        for i in range(fids.shape[0]):
            nid = nids[i][0]
            bins = fids[i]
            for fid, bid in enumerate(bins):
                index = self._indexer.get_position(nid, fid, bid)
                for name, value in targets.items():
                    self._values_mapping[name].iadd_slice(index, value[i])
        return self

    def iadd(self, hist: "Histogram"):
        for name, value_container in hist._values_mapping.items():
            if name in self._values_mapping:
                self._values_mapping[name].iadd(value_container)
            else:
                self._values_mapping[name] = value_container
        return self

    def decrypt(self, sk_map: dict):
        values_mapping = {}
        for name, value_container in self._values_mapping.items():
            if name in sk_map:
                values_mapping[name] = value_container.decrypt(sk_map[name])
            else:
                values_mapping[name] = value_container
        return Histogram(self._indexer, values_mapping)

    def decode(self, coder_map: dict):
        values_mapping = {}
        for name, value_container in self._values_mapping.items():
            if name in coder_map:
                coder, dtype = coder_map[name]
                values_mapping[name] = value_container.decode(coder, dtype)
            else:
                values_mapping[name] = value_container
        return Histogram(self._indexer, values_mapping)

    def i_shuffle(self, seed, reverse=False):
        shuffler = self._indexer.get_shuffler(seed)
        for name, value_container in self._values_mapping.items():
            value_container.i_shuffle(shuffler, reverse=reverse)

    def __str__(self):
        result = ""
        for nid in range(self._indexer.node_size):
            result += f"node-{nid}:\n"
            for fid in range(self._indexer.feature_size):
                result += f"\tfeature-{fid}:\n"
                for bid in range(self._indexer.get_bin_num(fid)):
                    for name, value_container in self._values_mapping.items():
                        start = self._indexer.get_position(nid, fid, bid)
                        values = value_container.slice(start, start + 1)
                        result += f"\t\t{name}: {values}"
                    result += "\n"
        return result

    def flatten_all_feature_bins(self):
        """
        flatten all feature bins into one bin

        Note: this method will change the histogram data itself, any inplace ops on the original histogram will
        affect the result of this method and vice versa.
        :return:
        """
        indexer = self._indexer.flatten_in_node()
        values = {name: value_container for name, value_container in self._values_mapping.items()}
        return Histogram(indexer, values)

    def i_cumsum_bins(self):
        for name, value_container in self._values_mapping.items():
            value_container.i_chunking_cumsum(self._indexer.global_flatten_bin_sizes())

    def sum_bins(self):
        indexer = self._indexer.squeeze_bins()
        values_mapping = {}
        intervals = self._indexer.get_global_feature_intervals()
        for name, value_container in self._values_mapping.items():
            values_mapping[name] = value_container.chunking_sum(intervals)
        return Histogram(indexer, values_mapping)

    def to_splits(self, k) -> typing.Iterator[typing.Tuple[(int, "HistogramSplits")]]:
        for pid, (start, end), indexes in self._indexer.splits_into_k(k):
            data = {}
            for name, value_container in self._values_mapping.items():
                data[name] = value_container.intervals_slice(indexes)
            yield pid, HistogramSplits(self._indexer.node_size, start, end, data)

    def reshape(self, feature_bin_sizes):
        indexer = self._indexer.reshape(feature_bin_sizes)
        return Histogram(indexer, self._values_mapping)

    def extract_data(self):
        data = {}
        for name, value_container in self._values_mapping.items():
            node_data_list = value_container.extract_node_data(self._indexer.node_axis_stride, self._indexer.node_size)
            for nid, node_data in enumerate(node_data_list):
                if nid not in data:
                    data[nid] = {}
                data[nid][name] = node_data
        return data


class HistogramSplits:
    def __init__(self, num_node, start, end, data):
        self.num_node = num_node
        self.start = start
        self.end = end
        self._data: typing.MutableMapping[str, HistogramValues] = data

    def __str__(self):
        result = f"HistogramSplits(start={self.start}, end={self.end}):\n"
        for name, value in self._data.items():
            result += f"{name}: {value}\n"
        return result

    def __repr__(self):
        return self.__str__()

    def iadd(self, other: "HistogramSplits"):
        for name, value in other._data.items():
            self._data[name].iadd(value)
        return self

    def i_decrypt(self, sk_map):
        for name, value in self._data.items():
            if name in sk_map:
                self._data[name] = value.decrypt(sk_map[name])
        return self

    def i_decode(self, coder_map):
        for name, value in self._data.items():
            if name in coder_map:
                coder, dtype = coder_map[name]
                self._data[name] = value.decode(coder, dtype)
        return self

    def decrypt(
        self,
        sk_map: MutableMapping[str, typing.Any],
        coder_map: MutableMapping[str, typing.Tuple[typing.Any, torch.dtype]],
    ):
        self.i_decrypt(sk_map)
        self.i_decode(coder_map)
        return self

    @classmethod
    def cat(cls, splits: typing.List["HistogramSplits"]):
        data = {}
        chunks_info = []
        for split in splits:
            chunks_info.append((split.num_node, split.end - split.start))
            for name, value in split._data.items():
                if name not in data:
                    data[name] = [value]
                else:
                    data[name].append(value)
        for name, values in data.items():
            data[name] = values[0].cat(chunks_info, values)
        return data


class DistributedHistogram:
    def __init__(self, node_size, feature_bin_sizes, value_schemas, seed):
        self._node_size = node_size
        self._feature_bin_sizes = feature_bin_sizes
        self._node_data_size = sum(feature_bin_sizes)
        self._value_schemas = value_schemas
        self._seed = seed

    def i_update(self, data, k=None):
        if k is None:
            k = data.count()
        mapper = get_partition_hist_build_mapper(
            self._node_size, self._feature_bin_sizes, self._value_schemas, self._seed, k
        )
        table = data.mapReducePartitions(mapper, lambda x, y: x.iadd(y))
        return ShuffledHistogram(table, self._node_size, self._node_data_size)

    def recover_feature_bins(self, seed, split_points: typing.Dict[int, int]) -> typing.Dict[int, int]:
        indexer = HistogramIndexer(self._node_size, self._feature_bin_sizes)
        points = list(split_points.items())
        real_indexes = indexer.get_shuffler(seed).get_reverse_indexes(1, [p[1] for p in points])
        return {nid: indexer.get_reverse_position(index) for (nid, _), index in zip(points, real_indexes)}


class ShuffledHistogram:
    def __init__(self, table, node_size, node_data_size):
        self._table = table
        self._node_size = node_size
        self._node_data_size = node_data_size

    def decrypt(
        self,
        sk_map: MutableMapping[str, typing.Any],
        coder_map: MutableMapping[str, typing.Tuple[typing.Any, torch.dtype]],
    ):
        out = list(self._table.map(lambda pid, split: (pid, split.decrypt(sk_map, coder_map))).collect())
        out.sort(key=lambda x: x[0])
        return self.cat([split for _, split in out])

    def cat(self, hists: typing.List["HistogramSplits"]) -> "Histogram":
        data = HistogramSplits.cat(hists)
        return Histogram(HistogramIndexer(self._node_size, [self._node_data_size]), data)


def argmax_reducer(
    max1: typing.Dict[int, typing.Tuple[int, int, float]], max2: typing.Dict[int, typing.Tuple[int, int, float]]
):
    for nid, (pid, index, gain) in max2.items():
        if nid in max1:
            if gain > max1[nid][2]:
                max1[nid] = (pid, index, gain)
    return max1


def get_partition_hist_build_mapper(node_size, feature_bin_sizes, value_schemas, seed, k):
    def _partition_hist_build_mapper(part):
        hist = Histogram.create(node_size, feature_bin_sizes, value_schemas)
        for _, raw in part:
            nids, fids, targets = raw
            hist.i_update(nids, fids, targets)
        hist.i_cumsum_bins()
        if seed is not None:
            hist.i_shuffle(seed)
        splits = hist.to_splits(k)
        return list(splits)

    return _partition_hist_build_mapper
