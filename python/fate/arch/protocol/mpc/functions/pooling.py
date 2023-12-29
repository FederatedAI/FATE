#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fate.arch.protocol import mpc

__all__ = [
    "_max_pool2d_backward",
    "adaptive_max_pool2d",
    "adaptive_avg_pool2d",
    "max_pool2d",
]


def max_pool2d(
    self,
    kernel_size,
    padding=0,
    stride=None,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """Applies a 2D max pooling over an input signal composed of several
    input planes.
    """
    max_input = self.clone()
    max_input.data, output_size = _pool2d_reshape(
        self.data,
        kernel_size,
        padding=padding,
        stride=stride,
        dilation=dilation,
        ceil_mode=ceil_mode,
        # padding with extremely negative values to avoid choosing pads.
        # The magnitude of this value should not be too large because
        # multiplication can otherwise fail.
        pad_value=(-(2**24)),
        # TODO: Find a better solution for padding with max_pooling
    )
    max_vals, argmax_vals = max_input.max(dim=-1, one_hot=True)
    max_vals = max_vals.view(output_size)
    if return_indices:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        argmax_vals = argmax_vals.view(output_size + kernel_size)
        return max_vals, argmax_vals
    return max_vals


def _max_pool2d_backward(
    self,
    indices,
    kernel_size,
    padding=None,
    stride=None,
    dilation=1,
    ceil_mode=False,
    output_size=None,
):
    """Implements the backwards for a `max_pool2d` call."""
    # Setup padding
    if padding is None:
        padding = 0
    if isinstance(padding, int):
        padding = padding, padding
    assert isinstance(padding, tuple), "padding must be a int, tuple, or None"
    p0, p1 = padding

    # Setup stride
    if stride is None:
        stride = kernel_size
    if isinstance(stride, int):
        stride = stride, stride
    assert isinstance(stride, tuple), "stride must be a int, tuple, or None"
    s0, s1 = stride

    # Setup dilation
    if isinstance(stride, int):
        dilation = dilation, dilation
    assert isinstance(dilation, tuple), "dilation must be a int, tuple, or None"
    d0, d1 = dilation

    # Setup kernel_size
    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size
    assert isinstance(padding, tuple), "padding must be a int or tuple"
    k0, k1 = kernel_size

    assert self.dim() == 4, "Input to _max_pool2d_backward must have 4 dimensions"
    assert indices.dim() == 6, "Indices input for _max_pool2d_backward must have 6 dimensions"

    # Computes one-hot gradient blocks from each output variable that
    # has non-zero value corresponding to the argmax of the corresponding
    # block of the max_pool2d input.
    kernels = self.view(self.size() + (1, 1)) * indices

    # Use minimal size if output_size is not specified.
    if output_size is None:
        output_size = (
            self.size(0),
            self.size(1),
            s0 * self.size(2) - 2 * p0,
            s1 * self.size(3) - 2 * p1,
        )

    # Account for input padding
    result_size = list(output_size)
    result_size[-2] += 2 * p0
    result_size[-1] += 2 * p1

    # Account for input padding implied by ceil_mode
    if ceil_mode:
        c0 = self.size(-1) * s1 + (k1 - 1) * d1 - output_size[-1]
        c1 = self.size(-2) * s0 + (k0 - 1) * d0 - output_size[-2]
        result_size[-2] += c0
        result_size[-1] += c1

    # Sum the one-hot gradient blocks at corresponding index locations.
    result = self.new(torch.zeros(result_size, device=kernels.device))
    for i in range(self.size(2)):
        for j in range(self.size(3)):
            left_ind = s0 * i
            top_ind = s1 * j

            result[
                :,
                :,
                left_ind : left_ind + k0 * d0 : d0,
                top_ind : top_ind + k1 * d1 : d1,
            ] += kernels[:, :, i, j]

    # Remove input padding
    if ceil_mode:
        result = result[:, :, : result.size(2) - c0, : result.size(3) - c1]
    result = result[:, :, p0 : result.size(2) - p0, p1 : result.size(3) - p1]

    return result


def adaptive_avg_pool2d(self, output_size):
    r"""
    Applies a 2D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
    """
    if output_size is None or output_size[0] is None:
        output_size = self.shape[-2:]

    if self.shape[-2:] == output_size:
        return self.clone()

    resized_input, args, kwargs = _adaptive_pool2d_helper(self, output_size, reduction="mean")
    return resized_input.avg_pool2d(*args, **kwargs)


def adaptive_max_pool2d(self, output_size, return_indices=False):
    r"""Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if output_size is None or output_size[0] is None:
        output_size = self.shape[-2:]

    if self.shape[-2:] == output_size:
        if return_indices:
            return self.clone(), self.new(torch.ones(self.size() + torch.Size(output_size)))
        return self.clone()

    resized_input, args, kwargs = _adaptive_pool2d_helper(self, output_size, reduction="max")
    return resized_input.max_pool2d(*args, **kwargs, return_indices=return_indices)


# Helper functions
def _adaptive_pool2d_helper(input, output_size, reduction="mean"):
    r"""
    Provides a helper that adapts the input size and provides input
    args / kwargs to allow pool2d functions to emulate adaptive pool2d
    functions.

    This function computes the kernel_size, stride, and padding for
    pool2d functions and inserts rows along each dimension so that
    a constant stride can be used.
    """
    input = input.clone()

    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    assert len(output_size) == 2, "output_size must be 2-dimensional."

    output_size = list(output_size)
    for i in range(2):
        if output_size[i] is None:
            output_size[i] = input.size(i - 2)

    # Compute the start_index and end_index for kernels
    def compute_kernels(in_size, out_size):
        step = in_size / out_size

        starts = []
        ends = []
        max_kernel_size = 0
        for j in range(out_size):
            # Compute local kernel size
            start_index = int(j * step)
            end_index = int(math.ceil((j + 1) * step))
            k = end_index - start_index

            # Update global kernel size
            max_kernel_size = k if k > max_kernel_size else max_kernel_size

            # Store local kernels
            starts.append(start_index)
            ends.append(end_index)

        return starts, ends, max_kernel_size

    # Repeats a row `ind` of `tensor` at dimension `dim` for overlapping kernels
    def repeat_row(tensor, dim, ind):
        device = tensor.device
        x = tensor.index_select(dim, torch.arange(ind, device=device))
        y = tensor.index_select(dim, torch.arange(ind, tensor.size(dim), device=device))
        repeated_row = tensor.index_select(dim, torch.tensor(ind - 1, device=device))
        return mpc.cat([x, repeated_row, y], dim=dim)

    # Extends a row where a kernel is smaller than the maximum kernel size
    def extend_row(tensor, dim, start_ind, end_ind):
        device = tensor.device
        if reduction == "mean":
            extended_value = tensor.index_select(dim, torch.arange(start_ind, end_ind, device=device))
            extended_value = extended_value.mean(dim, keepdim=True)
        elif reduction == "max":
            extended_value = tensor.index_select(dim, torch.tensor(start_ind, device=device))
        else:
            raise ValueError(f"Invalid reduction {reduction} for adaptive pooling.")

        if start_ind == 0:
            return mpc.cat([extended_value, tensor], dim=dim)

        x = tensor.index_select(dim, torch.arange(start_ind, device=device))
        y = tensor.index_select(dim, torch.arange(start_ind, tensor.size(dim), device=device))
        return mpc.cat([x, extended_value, y], dim=dim)

    strides = []
    for i in range(2):
        dim = i - 2 + input.dim()
        in_size = input.size(dim)
        out_size = output_size[i] if output_size[i] is not None else in_size

        # Compute repeats
        if out_size > 1:
            starts, ends, stride = compute_kernels(in_size, out_size)

            added_rows = 0
            for i in range(out_size):
                start_ind = starts[i]
                end_ind = ends[i]

                # Extend kernel so all kernels have the same size
                k = end_ind - start_ind
                for _ in range(k, stride):
                    input = extend_row(input, dim, start_ind + added_rows, end_ind + added_rows)
                    added_rows += 1

                if i == out_size - 1:
                    break

                # Repeat overlapping rows so stride can be equal to the kernel size
                if end_ind > starts[i + 1]:
                    input = repeat_row(input, dim, end_ind + added_rows)
                    added_rows += 1
        else:
            stride = in_size

        strides.append(stride)

    strides = tuple(strides)
    kernel_sizes = strides

    args = (kernel_sizes,)
    kwargs = {"stride": strides}

    return input, args, kwargs


def _pooling_output_shape(input_size, kernel_size, pad_l, pad_r, stride, dilation, ceil_mode):
    """
    Generates output shape along a single dimension following conventions here:
    https://github.com/pytorch/pytorch/blob/b0424a895c878cb865947164cb0ce9ce3c2e73ef/aten/src/ATen/native/Pool.h#L24-L38
    """
    numerator = input_size + pad_l + pad_r - dilation * (kernel_size - 1) - 1
    if ceil_mode:
        numerator += stride - 1

    output_size = numerator // stride + 1

    # ensure that the last pooling starts inside the image
    # needed to avoid problems in ceil mode
    if ceil_mode and (output_size - 1) * stride >= input_size + pad_l:
        output_size -= 1

    return output_size


def _pool2d_reshape(
    input,
    kernel_size,
    padding=None,
    stride=None,
    dilation=1,
    ceil_mode=False,
    pad_value=0,
):
    """Rearrange a 4-d tensor so that each kernel is represented by each row"""

    # Setup kernel / stride / dilation values
    k = kernel_size
    if isinstance(k, int):
        k = (k, k)

    s = stride
    if s is None:
        s = k
    elif isinstance(s, int):
        s = (s, s)

    d = dilation
    if isinstance(d, int):
        d = (d, d)

    # Assert input parameters are correct type / size
    assert isinstance(k, tuple), "kernel_size must be an int or tuple"
    assert isinstance(s, tuple), "stride must be and int, a tuple, or None"
    assert len(k) == 2, "kernel_size must be an int or tuple pair"
    assert len(s) == 2, "stride must be an int or tuple pair"
    assert isinstance(pad_value, int), "pad_value must be an integer"
    assert input.dim() >= 2, "Pooling input dimension should be at least 2"

    # Apply padding if necessary
    if padding is not None:
        padding = (padding, padding) if isinstance(padding, int) else padding
        assert len(padding) == 2, "Padding must be an integer or a pair"
        padding = (padding[0], padding[0], padding[1], padding[1])
    else:
        padding = (0, 0, 0, 0)

    # Compute output size based on parameters
    n = input.size()[:-2]
    h = _pooling_output_shape(input.size(-2), k[0], padding[0], padding[1], s[0], d[0], ceil_mode)
    w = _pooling_output_shape(input.size(-1), k[1], padding[2], padding[3], s[1], d[1], ceil_mode)

    out_size = tuple(n + (h, w))

    input = torch.nn.functional.pad(input, padding, value=pad_value)
    if ceil_mode:
        update_pad = [0, 0, 0, 0]
        update_pad[3] = h * s[0] + (k[0] - 1) * d[0] - input.size(-2)
        update_pad[1] = w * s[1] + (k[1] - 1) * d[1] - input.size(-1)
        input = torch.nn.functional.pad(input, tuple(update_pad), value=pad_value)

    # Reshape input to arrange kernels to be represented by rows
    kernel_indices = torch.tensor(range(0, k[1] * d[1], d[1]), device=input.device)
    kernel_indices = torch.cat([kernel_indices + i * input.size(-1) for i in range(0, k[0] * d[0], d[0])])
    kernel_indices = torch.stack([kernel_indices + i * s[1] for i in range(w)])

    offset = input.size(-1)
    kernel_indices = torch.cat([kernel_indices + i * s[0] * offset for i in range(h)])

    for dim in range(2, input.dim()):
        offset *= input.size(-dim)
        kernel_indices = torch.stack([kernel_indices + i * offset for i in range(input.size(-dim - 1))])

    output = input.take(kernel_indices)
    return output, out_size
