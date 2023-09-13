import logging

import torch

from ._plain import HistogramPlainValues
from ._value import HistogramValues

logger = logging.getLogger(__name__)


class HistogramEncodedValues(HistogramValues):
    def __init__(self, data, size: int, dtype: torch.dtype, stride: int):
        self.data = data
        self.size = size
        self.dtype = dtype
        self.stride = stride

    def decode_f64(self, coder):
        return HistogramPlainValues(coder.decode_f64_vec(self.data), self.dtype, self.size, self.stride)

    def decode_i64(self, coder):
        return HistogramPlainValues(coder.decode_i64_vec(self.data), self.dtype, self.size, self.stride)

    def decode_f32(self, coder):
        return HistogramPlainValues(coder.decode_f32_vec(self.data), self.dtype, self.size, self.stride)

    def decode_i32(self, coder):
        return HistogramPlainValues(coder.decode_i32_vec(self.data), self.dtype, self.size, self.stride)

    def decode(self, coder, dtype):
        if dtype is None:
            dtype = self.dtype
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

    def unpack(self, coder, pack_num, offset_bit, precision, total_num, stride):
        data = coder.unpack_floats(self.data, offset_bit, pack_num, precision, total_num)
        return HistogramPlainValues(data, self.dtype, self.size, stride)

    def slice(self, start, end):
        if hasattr(self.data, "slice"):
            return self.data.slice(start * self.stride, end * self.stride)
        else:
            return "<Encoded data=unknown>"
