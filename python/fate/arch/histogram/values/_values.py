import typing
from typing import MutableMapping
from ._value import HistogramValues
from ._plain import HistogramPlainValues
from ._cipher import HistogramEncryptedValues

if typing.TYPE_CHECKING:
    from ..indexer import HistogramIndexer


class HistogramValuesContainer(object):
    def __init__(self, data: MutableMapping[str, HistogramValues]):
        self._data = data

    @classmethod
    def create(cls, values_schema: dict, size):
        values_mapping = {}
        for name, items in values_schema.items():
            stride = items.get("stride", 1)
            if items["type"] == "ciphertext":
                pk = items["pk"]
                evaluator = items["evaluator"]
                coder = items.get("coder")
                dtype = items.get("dtype")
                values_mapping[name] = HistogramEncryptedValues.zeros(pk, evaluator, size, coder, dtype, stride)
            elif items["type"] == "plaintext":
                import torch

                dtype = items.get("dtype", torch.float64)
                values_mapping[name] = HistogramPlainValues.zeros(size, stride=stride, dtype=dtype)
            else:
                raise NotImplementedError
        return HistogramValuesContainer(values_mapping)

    def __str__(self):
        result = ""
        for name, value in self._data.items():
            result += f"{name}: {value}\n"
        return result

    def i_update(self, targets, positions):
        for name, value in targets.items():
            self._data[name].i_update(value, positions)
        return self

    def i_update_with_masks(self, targets, positions, masks):
        for name, value in targets.items():
            self._data[name].i_update_with_masks(value, positions, masks)
        return self

    def iadd(self, other: "HistogramValuesContainer"):
        for name, values in other._data.items():
            if name in self._data:
                self._data[name].iadd(values)
            else:
                self._data[name] = values
        return self

    def i_sub_on_key(self, from_key, to_key):
        left_value = self._data[from_key]
        right_value = self._data[to_key]
        if isinstance(left_value, HistogramEncryptedValues):
            if isinstance(right_value, HistogramEncryptedValues):
                assert left_value.stride == right_value.stride
                left_value.data = left_value.evaluator.sub(left_value.pk, left_value.data, right_value.data)
            elif isinstance(right_value, HistogramPlainValues):
                assert left_value.stride == right_value.stride
                if left_value.coder is None:
                    raise ValueError(f"coder is None, please set coder for i_sub_on_key({from_key}, {to_key})")
                left_value.data = left_value.evaluator.sub_plain(
                    left_value.data, right_value.data, left_value.pk, left_value.coder
                )
            else:
                raise NotImplementedError
        elif isinstance(left_value, HistogramPlainValues):
            if isinstance(right_value, HistogramEncryptedValues):
                assert left_value.stride == right_value.stride
                if right_value.coder is None:
                    raise ValueError(f"coder is None, please set coder for i_sub_on_key({from_key}, {to_key})")
                data = right_value.evaluator.rsub_plain(
                    right_value.data, left_value.data, right_value.pk, right_value.coder
                )
                self._data[from_key] = HistogramEncryptedValues(
                    right_value.pk,
                    right_value.evaluator,
                    data,
                    right_value.coder,
                    right_value.dtype,
                    right_value.size,
                    right_value.stride,
                )
            elif isinstance(right_value, HistogramPlainValues):
                assert left_value.stride == right_value.stride
                left_value.data = left_value.data - right_value.data
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def decrypt(self, sk_map: dict):
        values_mapping = {}
        for name, values in self._data.items():
            if name in sk_map:
                values_mapping[name] = values.decrypt(sk_map[name])
            else:
                values_mapping[name] = values
        return HistogramValuesContainer(values_mapping)

    def decode(self, coder_map: dict):
        values_mapping = {}
        for name, values in self._data.items():
            if name in coder_map:
                coder, dtype = coder_map[name]
                values_mapping[name] = values.decode(coder, dtype)
            else:
                values_mapping[name] = values
        return HistogramValuesContainer(values_mapping)

    def unpack_decode(self, unpacker_map: dict):
        values_mapping = {}
        for name, values in self._data.items():
            if name in unpacker_map:
                unpacker, pack_num, offset_bit, precision, total_num, stride = unpacker_map[name]
                values_mapping[name] = values.unpack(unpacker, pack_num, offset_bit, precision, total_num, stride)
            else:
                values_mapping[name] = values
        return HistogramValuesContainer(values_mapping)

    def i_squeeze(self, squeeze_map):
        for name, value in self._data.items():
            if name in squeeze_map:
                pack_num, offset_bit = squeeze_map[name]
                self._data[name] = value.squeeze(pack_num, offset_bit)

    def i_shuffle(self, shuffler, reverse=False):
        for name, values in self._data.items():
            values.i_shuffle(shuffler, reverse=reverse)

    def shuffle(self, shuffler, reverse=False):
        data = {}
        for name, values in self._data.items():
            data[name] = values.shuffle(shuffler, reverse=reverse)
        return HistogramValuesContainer(data)

    def i_cumsum_bins(self, intervals: list):
        for name, values in self._data.items():
            values.i_chunking_cumsum(intervals)

    def intervals_slice(self, intervals: list):
        result = {}
        for name, values in self._data.items():
            result[name] = values.intervals_slice(intervals)
        return HistogramValuesContainer(result)

    def extract_data(self, indexer: "HistogramIndexer"):
        data = {}
        for name, value_container in self._data.items():
            node_data_list = value_container.extract_node_data(indexer.node_axis_stride, indexer.node_size)
            for nid, node_data in enumerate(node_data_list):
                if nid not in data:
                    data[nid] = {}
                data[nid][name] = node_data
        return data

    def compute_child(self, weak_child: "HistogramValuesContainer", positions: list, size):
        result = {}
        for name, values in self._data.items():
            result[name] = values.compute_child(weak_child._data[name], positions, size)
        return HistogramValuesContainer(result)

    @classmethod
    def cat(cls, chunks_info, chunks_values: typing.List["HistogramValuesContainer"]) -> "HistogramValuesContainer":
        data = {}
        for chunk_values in chunks_values:
            for name, value in chunk_values._data.items():
                if name not in data:
                    data[name] = [value]
                else:
                    data[name].append(value)
        for name, values in data.items():
            data[name] = values[0].cat(chunks_info, values)
        return HistogramValuesContainer(data)

    def show(self, indexer):
        result = ""
        indexes = indexer.unflatten_indexes()
        for nid, fids in indexes.items():
            result += f"node-{nid}:\n"
            for fid, bids in fids.items():
                result += f"\tfeature-{fid}:\n"
                for start in bids:
                    for name, value_container in self._data.items():
                        values = value_container.slice(start, start + 1)
                        result += f"\t\t{name}: {values}"
                    result += "\n"
        return result

    def to_structured_dict(self, indexer):
        indexes = indexer.unflatten_indexes()
        result = {}
        for nid, fids in indexes.items():
            result[nid] = {}
            for name, value_container in self._data.items():
                result[nid][name] = {}
                for fid, bids in fids.items():
                    result[nid][name][fid] = {}
                    for bid, start in enumerate(bids):
                        values = value_container.slice(start, start + 1)
                        result[nid][name][fid][bid] = values
        return result
