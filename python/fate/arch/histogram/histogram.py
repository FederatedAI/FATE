import typing
import logging
from typing import List, MutableMapping, Tuple
from fate.arch.abc._table import CTableABC

import torch

# from fate_utils.histogram import HistogramIndexer, Shuffler
from .indexer import HistogramIndexer, Shuffler

logger = logging.getLogger(__name__)


class HistogramValues:
    def iadd_slice(self, value, sa, sb, size):
        raise NotImplementedError

    def i_update(self, value, positions):
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

    def squeeze(self, pack_num, offset_bit):
        raise NotImplementedError

    def unpack(self, coder, pack_num, offset_bit, precision, total_num):
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

    def i_update(self, value, positions):
        from fate.arch.tensor.phe import PHETensor
        if isinstance(value, PHETensor):
            value = value.data

        if hasattr(self.evaluator, "i_update"):
            return self.evaluator.i_update(self.pk, self.data, value, positions, self.stride)
        else:
            for i, feature_positions in enumerate(positions):
                for pos in feature_positions:
                    self.evaluator.i_add(self.pk, self.data, value, pos * self.stride, i * self.stride, self.stride)

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

    def squeeze(self, pack_num, offset_bit):
        data = self.evaluator.squeeze(pack_num, offset_bit, self.pk)
        return HistogramEncryptedValues(self.pk, self.evaluator, data, self.stride)

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

    def unpack(self, coder, pack_num, offset_bit, precision, total_num):
        return HistogramPlainValues(coder.unpack_floats(self.data, offset_bit, pack_num, precision, total_num),
                                    self.stride)

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
            result[start:end] = self.data[s * self.stride: e * self.stride]
            start = end
        return HistogramPlainValues(result, self.stride)

    def iadd_slice(self, value, sa, sb, size):
        size = size * self.stride
        value = value.view(-1)
        self.data[sa: sa + size] += value[sb: sb + size]

    def slice(self, start, end):
        return HistogramPlainValues(self.data[start * self.stride: end * self.stride], self.stride)

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
            value = value.view(-1, self.stride).unsqueeze(1).expand(-1, index.shape[1], self.stride).reshape(-1,
                                                                                                             self.stride)
            index = index.flatten().unsqueeze(1).expand(-1, self.stride)
        if self.data.dtype != value.dtype:
            logger.warning(f"update value dtype {value.dtype} is not equal to data dtype {self.data.dtype}")
            value = value.to(data.dtype)
        data.scatter_add_(0, index, value)

    def i_shuffle(self, shuffler: "Shuffler", reverse=False):
        indices = shuffler.get_shuffle_index(step=self.stride, reverse=reverse)
        self.data = self.data[indices]

    def i_chunking_cumsum(self, chunk_sizes: typing.List[int]):
        data_view = self.data.view(-1, self.stride)
        start = 0
        for num in chunk_sizes:
            data_view[start: start + num, :] = data_view[start: start + num, :].cumsum(dim=0)
            start += num

    def chunking_sum(self, intervals: typing.List[typing.Tuple[int, int]]):
        result = torch.zeros(len(intervals) * self.stride, dtype=self.data.dtype)
        data_view = self.data.view(-1, self.stride)
        for i, (start, end) in enumerate(intervals):
            result[i * self.stride: (i + 1) * self.stride] = data_view[start:end, :].sum(dim=0)
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

    def maybe_create_shuffler(self, seed):
        if seed is None:
            return None
        return self._indexer.get_shuffler(seed)

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
        positions = self._indexer.get_positions(
            nids.flatten().detach().numpy().tolist(), fids.detach().numpy().tolist()
        )
        for name, value in targets.items():
            self._values_mapping[name].i_update(value, positions)
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

    def i_shuffle(self, shuffler, reverse=False):
        if shuffler is not None:
            for name, value_container in self._values_mapping.items():
                value_container.i_shuffle(shuffler, reverse=reverse)

    def __str__(self):
        result = ""
        indexes = self._indexer.unflatten_indexes()
        for nid, fids in indexes.items():
            result += f"node-{nid}:\n"
            for fid, bids in fids.items():
                result += f"\tfeature-{fid}:\n"
                for start in bids:
                    for name, value_container in self._values_mapping.items():
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
        intervals = self._indexer.get_feature_position_ranges()
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

    def i_squeeze(self, squeeze_map):
        for name, value in self._data.items():
            if name in squeeze_map:
                pack_num, offset_bit = squeeze_map[name]
                self._data[name] = value.squeeze(pack_num, offset_bit)
        return self

    def i_unpack_decode(self, coder_map):
        for name, value in self._data.items():
            if name in coder_map:
                coder, pack_num, offset_bit, precision, total_num = coder_map[name]
                self._data[name] = value.unpack(coder, pack_num, offset_bit, precision, total_num)
        return self

    def i_decode(self, coder_map):
        for name, value in self._data.items():
            if name in coder_map:
                coder, dtype = coder_map[name]
                self._data[name] = value.decode(coder, dtype)
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

    def pack(self, coder_map):
        for name, value in self._data.items():
            if name in coder_map:
                pack_num, offset_bit = coder_map[name]
                self._data[name] = value.squeeze(pack_num=pack_num, offset_bit=offset_bit)
        return self


class DistributedHistogram:
    def __init__(self, node_size, feature_bin_sizes, value_schemas, seed):
        self._node_size = node_size
        self._feature_bin_sizes = feature_bin_sizes
        self._node_data_size = sum(feature_bin_sizes)
        self._value_schemas = value_schemas
        self._seed = seed

    def i_update(self, data, k=None) -> "ShuffledHistogram":
        """
        Update the histogram with the data.
        Args:
            data: table with the following schema:
            k: number of output splits of the histogram
        Returns:
            ShuffledHistogram, the shuffled(if seed is not None) histogram
        """
        if k is None:
            k = data.partitions
        mapper = get_partition_hist_build_mapper(
            self._node_size, self._feature_bin_sizes, self._value_schemas, self._seed, k
        )
        table = data.mapReducePartitions(mapper, lambda x, y: x.iadd(y))
        return ShuffledHistogram(table, self._node_size, self._node_data_size)

    def recover_feature_bins(
            self, seed, split_points: typing.Dict[int, int]
    ) -> typing.Dict[int, typing.Tuple[int, int]]:
        """
        Recover the feature bins from the split points.

        Args:
            seed: random seed
            split_points: nid -> split data index

        Returns:
            nid -> (fid, bid)
        """
        indexer = HistogramIndexer(self._node_size, self._feature_bin_sizes)
        points = list(split_points.items())
        shuffler = indexer.get_shuffler(seed)
        real_indexes = shuffler.get_reverse_indexes(1, [p[1] for p in points])
        real_fid_bid = {}
        for (nid, _), index in zip(points, real_indexes):
            _, fid, bid = indexer.get_reverse_position(index)
            real_fid_bid[nid] = (fid, bid)
        return real_fid_bid


class ShuffledHistogram:
    def __init__(self, table: CTableABC[int, HistogramSplits], node_size, node_data_size, squeezed=False):
        self._table = table
        self._node_size = node_size
        self._node_data_size = node_data_size
        self._squeezed = squeezed

    def squeeze(self, squeeze_map: MutableMapping[str, typing.Tuple[int, int]]):
        """
        Squeeze the histogram values.

        Args:
            squeeze_map: name -> (pack_num, offset_bit)
        """
        table = self._table.mapValues(lambda split: split.i_squeeze(squeeze_map))
        return ShuffledHistogram(table, self._node_size, self._node_data_size, True)

    def decrypt_(self, sk_map: MutableMapping[str, typing.Any]):
        """
        Decrypt the histogram values.

        Args:
            sk_map: name -> sk
        """
        table = self._table.mapValues(lambda split: split.i_decrypt(sk_map))
        return ShuffledHistogram(table, self._node_size, self._node_data_size)

    def unpack_decode(self, coder_map: MutableMapping[str, typing.Tuple[typing.Any, int, int, int, int]]):
        """
        Unpack and decode the histogram values.

        Args:
            coder_map: name -> (coder, pack_num, offset_bit, precision, total_num)
        """
        table = self._table.mapValues(lambda split: split.i_unpack_decode(coder_map))
        return ShuffledHistogram(table, self._node_size, self._node_data_size)

    def decode(self, coder_map: MutableMapping[str, typing.Tuple[typing.Any, torch.dtype]]):
        """
        Decode the histogram values.

        Args:
            coder_map: name -> (coder, dtype)
        """
        table = self._table.mapValues(lambda split: split.i_decode(coder_map))
        return ShuffledHistogram(table, self._node_size, self._node_data_size)

    def union(self) -> Histogram:
        """
        Union the splits into one histogram.
        """
        out = list(self._table.collect())
        out.sort(key=lambda x: x[0])
        return self.cat([split for _, split in out])

    def decrypt(
            self,
            sk_map: MutableMapping[str, typing.Any],
            coder_map: MutableMapping[str, typing.Tuple[typing.Any, torch.dtype]],
    ):
        out = list(self._table.mapValues(_decrypt_func(sk_map, coder_map, self._squeezed)).collect())
        out.sort(key=lambda x: x[0])
        return self.cat([split for _, split in out])

    def cat(self, hists: typing.List["HistogramSplits"]) -> "Histogram":
        data = HistogramSplits.cat(hists)
        return Histogram(HistogramIndexer(self._node_size, [self._node_data_size]), data)


def _decrypt_func(sk_map, coder_map, squeezed):
    def _decrypt(split: HistogramSplits):
        split.i_decrypt(sk_map)
        if squeezed:
            split.i_unpack_decode(coder_map)
            return split
        else:
            split.i_decode(coder_map)
            return split

    return _decrypt


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
        shuffle = hist.maybe_create_shuffler(seed)
        for _, raw in part:
            fids, nids, targets = raw
            hist.i_update(fids, nids, targets)
        hist.i_cumsum_bins()
        hist.i_shuffle(shuffle)
        splits = hist.to_splits(k)
        return splits

    return _partition_hist_build_mapper
