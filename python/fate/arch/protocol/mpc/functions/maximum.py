#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fate.arch.protocol import mpc
from fate.arch.protocol.mpc.config import cfg

__all__ = [
    "argmax",
    "argmin",
    "max",
    "min",
]


def argmax(self, dim=None, keepdim=False, one_hot=True):
    """Returns the indices of the maximum value of all elements in the
    `input` tensor.
    """
    method = cfg.safety.mpc.functions.max_method

    if self.dim() == 0:
        result = (
            self.new(torch.ones((), device=self.device)) if one_hot else self.new(torch.zeros((), device=self.device))
        )
        return result

    result = _argmax_helper(self, dim, one_hot, method, _return_max=False)

    if not one_hot:
        result = _one_hot_to_index(result, dim, keepdim, self.device)
    return result


def argmin(self, dim=None, keepdim=False, one_hot=True):
    """Returns the indices of the minimum value of all elements in the
    `input` tensor.
    """
    return (-self).argmax(dim=dim, keepdim=keepdim, one_hot=one_hot)


def max(self, dim=None, keepdim=False, one_hot=True):
    """Returns the maximum value of all elements in the input tensor."""
    method = cfg.safety.mpc.functions.max_method
    if dim is None:
        if method in ["log_reduction", "double_log_reduction"]:
            # max_result can be obtained directly
            max_result = _max_helper_all_tree_reductions(self, method=method)
        else:
            # max_result needs to be obtained through argmax
            with cfg.temp_override({"functions.max_method": method}):
                argmax_result = self.argmax(one_hot=True)
            max_result = self.mul(argmax_result).sum()
        return max_result
    else:
        argmax_result, max_result = _argmax_helper(self, dim=dim, one_hot=True, method=method, _return_max=True)
        if max_result is None:
            max_result = (self * argmax_result).sum(dim=dim, keepdim=keepdim)
        if keepdim:
            max_result = max_result.unsqueeze(dim) if max_result.dim() < self.dim() else max_result
        if one_hot:
            return max_result, argmax_result
        else:
            return (
                max_result,
                _one_hot_to_index(argmax_result, dim, keepdim, self.device),
            )


def min(self, dim=None, keepdim=False, one_hot=True):
    """Returns the minimum value of all elements in the input tensor."""
    result = (-self).max(dim=dim, keepdim=keepdim, one_hot=one_hot)
    if dim is None:
        return -result
    else:
        return -result[0], result[1]


# Helper functions
def _argmax_helper_pairwise(enc_tensor, dim=None):
    """Returns 1 for all elements that have the highest value in the appropriate
    dimension of the tensor. Uses O(n^2) comparisons and a constant number of
    rounds of communication
    """
    dim = -1 if dim is None else dim
    row_length = enc_tensor.size(dim) if enc_tensor.size(dim) > 1 else 2

    # Copy each row (length - 1) times to compare to each other row
    a = enc_tensor.expand(row_length - 1, *enc_tensor.size())

    # Generate cyclic permutations for each row
    b = mpc.stack([enc_tensor.roll(i + 1, dims=dim) for i in range(row_length - 1)])

    # Use either prod or sum & comparison depending on size
    if row_length - 1 < torch.iinfo(torch.long).bits * 2:
        pairwise_comparisons = a.ge(b)
        result = pairwise_comparisons.prod(0)
    else:
        # Sum of columns with all 1s will have value equal to (length - 1).
        # Using ge() since it is slightly faster than eq()
        pairwise_comparisons = a.ge(b)
        result = pairwise_comparisons.sum(0).ge(row_length - 1)
    return result, None


def _compute_pairwise_comparisons_for_steps(input_tensor, dim, steps):
    """
    Helper function that does pairwise comparisons by splitting input
    tensor for `steps` number of steps along dimension `dim`.
    """
    enc_tensor_reduced = input_tensor.clone()
    for _ in range(steps):
        m = enc_tensor_reduced.size(dim)
        x, y, remainder = enc_tensor_reduced.split([m // 2, m // 2, m % 2], dim=dim)
        pairwise_max = mpc.where(x >= y, x, y)
        enc_tensor_reduced = mpc.cat([pairwise_max, remainder], dim=dim)
    return enc_tensor_reduced


def _max_helper_log_reduction(enc_tensor, dim=None):
    """Returns max along dim `dim` using the log_reduction algorithm"""
    if enc_tensor.dim() == 0:
        return enc_tensor
    input, dim_used = enc_tensor, dim
    if dim is None:
        dim_used = 0
        input = enc_tensor.flatten()
    n = input.size(dim_used)  # number of items in the dimension
    steps = int(math.log(n))
    enc_tensor_reduced = _compute_pairwise_comparisons_for_steps(input, dim_used, steps)

    # compute max over the resulting reduced tensor with n^2 algorithm
    # note that the resulting one-hot vector we get here finds maxes only
    # over the reduced vector in enc_tensor_reduced, so we won't use it
    with cfg.temp_override({"functions.max_method": "pairwise"}):
        enc_max_vec, enc_one_hot_reduced = enc_tensor_reduced.max(dim=dim_used)
    return enc_max_vec


def _max_helper_double_log_recursive(enc_tensor, dim):
    """Recursive subroutine for computing max via double log reduction algorithm"""
    n = enc_tensor.size(dim)
    # compute integral sqrt(n) and the integer number of sqrt(n) size
    # vectors that can be extracted from n
    sqrt_n = int(math.sqrt(n))
    count_sqrt_n = n // sqrt_n
    # base case for recursion: no further splits along dimension dim
    if n == 1:
        return enc_tensor
    else:
        # split into tensors that can be broken into vectors of size sqrt(n)
        # and the remainder of the tensor
        size_arr = [sqrt_n * count_sqrt_n, n % sqrt_n]
        split_enc_tensor, remainder = enc_tensor.split(size_arr, dim=dim)

        # reshape such that dim holds sqrt_n and dim+1 holds count_sqrt_n
        updated_enc_tensor_size = [sqrt_n, enc_tensor.size(dim + 1) * count_sqrt_n]
        size_arr = [enc_tensor.size(i) for i in range(enc_tensor.dim())]
        size_arr[dim], size_arr[dim + 1] = updated_enc_tensor_size
        split_enc_tensor = split_enc_tensor.reshape(size_arr)

        # recursive call on reshaped tensor
        split_enc_max = _max_helper_double_log_recursive(split_enc_tensor, dim)

        # reshape the result to have the (dim+1)th dimension as before
        # and concatenate the previously computed remainder
        size_arr[dim], size_arr[dim + 1] = [count_sqrt_n, enc_tensor.size(dim + 1)]
        enc_max_tensor = split_enc_max.reshape(size_arr)
        full_max_tensor = mpc.cat([enc_max_tensor, remainder], dim=dim)

        # call the max function on dimension dim
        with cfg.temp_override({"functions.max_method": "pairwise"}):
            enc_max, enc_arg_max = full_max_tensor.max(dim=dim, keepdim=True)
        # compute max over the resulting reduced tensor with n^2 algorithm
        # note that the resulting one-hot vector we get here finds maxes only
        # over the reduced vector in enc_tensor_reduced, so we won't use it
        return enc_max


def _max_helper_double_log_reduction(enc_tensor, dim=None):
    """Returns max along dim `dim` using the double_log_reduction algorithm"""
    if enc_tensor.dim() == 0:
        return enc_tensor
    input, dim_used, size_arr = enc_tensor, dim, ()
    if dim is None:
        dim_used = 0
        input = enc_tensor.flatten()
    # turn dim_used into a positive number
    dim_used = dim_used + input.dim() if dim_used < 0 else dim_used
    if input.dim() > 1:
        size_arr = [input.size(i) for i in range(input.dim()) if i != dim_used]
    # add another dimension to vectorize double log reductions
    input = input.unsqueeze(dim_used + 1)
    enc_max_val = _max_helper_double_log_recursive(input, dim_used)
    enc_max_val = enc_max_val.squeeze(dim_used + 1)
    enc_max_val = enc_max_val.reshape(size_arr)
    return enc_max_val


def _max_helper_accelerated_cascade(enc_tensor, dim=None):
    """Returns max along dimension `dim` using the accelerated cascading algorithm"""
    if enc_tensor.dim() == 0:
        return enc_tensor
    input, dim_used = enc_tensor, dim
    if dim is None:
        dim_used = 0
        input = enc_tensor.flatten()
    n = input.size(dim_used)  # number of items in the dimension
    if n < 3:
        with cfg.temp_override({"functions.max_method": "pairwise"}):
            enc_max, enc_argmax = enc_tensor.max(dim=dim_used)
            return enc_max
    steps = int(math.log(math.log(math.log(n)))) + 1
    enc_tensor_reduced = _compute_pairwise_comparisons_for_steps(enc_tensor, dim_used, steps)
    enc_max = _max_helper_double_log_reduction(enc_tensor_reduced, dim=dim_used)
    return enc_max


def _max_helper_all_tree_reductions(enc_tensor, dim=None, method="log_reduction"):
    """
    Finds the max along `dim` using the specified reduction method. `method`
    can be one of [`log_reduction`, `double_log_reduction`, 'accelerated_cascade`]
    `log_reduction`: Uses O(n) comparisons and O(log n) rounds of communication
    `double_log_reduction`: Uses O(n loglog n) comparisons and O(loglog n) rounds
    of communication (Section 2.6.2 in https://folk.idi.ntnu.no/mlh/algkon/jaja.pdf)
    `accelerated_cascade`: Uses O(n) comparisons and O(loglog n) rounds of
    communication. (See Section 2.6.3 of https://folk.idi.ntnu.no/mlh/algkon/jaja.pdf)
    """
    if method == "log_reduction":
        return _max_helper_log_reduction(enc_tensor, dim)
    elif method == "double_log_reduction":
        return _max_helper_double_log_reduction(enc_tensor, dim)
    elif method == "accelerated_cascade":
        return _max_helper_accelerated_cascade(enc_tensor, dim)
    else:
        raise RuntimeError("Unknown max method")


def _argmax_helper_all_tree_reductions(enc_tensor, dim=None, method="log_reduction"):
    """
    Returns 1 for all elements that have the highest value in the appropriate
    dimension of the tensor. `method` can be one of [`log_reduction`,
    `double_log_reduction`, `accelerated_cascade`].
    `log_reduction`: Uses O(n) comparisons and O(log n) rounds of communication
    `double_log_reduction`: Uses O(n loglog n) comparisons and O(loglog n) rounds
    of communication (Section 2.6.2 in https://folk.idi.ntnu.no/mlh/algkon/jaja.pdf)
    `accelerated_cascade`: Uses O(n) comparisons and O(loglog n) rounds of
    communication. (See Section 2.6.3 of https://folk.idi.ntnu.no/mlh/algkon/jaja.pdf)
    """
    enc_max_vec = _max_helper_all_tree_reductions(enc_tensor, dim=dim, method=method)
    # reshape back to the original size
    enc_max_vec_orig = enc_max_vec
    if dim is not None:
        enc_max_vec_orig = enc_max_vec.unsqueeze(dim)
    # compute the one-hot vector over the entire tensor
    enc_one_hot_vec = enc_tensor.eq(enc_max_vec_orig)
    return enc_one_hot_vec, enc_max_vec


def _argmax_helper(enc_tensor, dim=None, one_hot=True, method="pairwise", _return_max=False):
    """
    Returns 1 for one randomly chosen element among all the elements that have
    the highest value in the appropriate dimension of the tensor. Sets up the CrypTensor
    appropriately, and then chooses among the different argmax algorithms.
    """
    if enc_tensor.dim() == 0:
        result = enc_tensor.new(torch.ones(())) if one_hot else enc_tensor.new(torch.zeros(()))
        if _return_max:
            return result, None
        return result

    updated_enc_tensor = enc_tensor.flatten() if dim is None else enc_tensor

    if method == "pairwise":
        result_args, result_val = _argmax_helper_pairwise(updated_enc_tensor, dim)
    elif method in ["log_reduction", "double_log_reduction", "accelerated_cascade"]:
        result_args, result_val = _argmax_helper_all_tree_reductions(updated_enc_tensor, dim, method)
    else:
        raise RuntimeError("Unknown argmax method")

    # Break ties by using a uniform weighted sample among tied indices
    result_args = result_args.weighted_index(dim)
    result_args = result_args.view(enc_tensor.size()) if dim is None else result_args

    if _return_max:
        return result_args, result_val
    else:
        return result_args


def _one_hot_to_index(tensor, dim, keepdim, device=None):
    """
    Converts a one-hot tensor output from an argmax / argmin function to a
    tensor containing indices from the input tensor from which the result of the
    argmax / argmin was obtained.
    """
    if dim is None:
        result = tensor.flatten()
        result = result * torch.tensor(list(range(tensor.nelement())), device=device)
        return result.sum()
    else:
        size = [1] * tensor.dim()
        size[dim] = tensor.size(dim)
        result = tensor * torch.tensor(list(range(tensor.size(dim))), device=device).view(size)
        return result.sum(dim, keepdim=keepdim)
