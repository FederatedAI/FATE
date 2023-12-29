#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from functools import reduce

import torch

from .common.util import _grad_input_padding

# registry that maps function names to AutogradFunctions:
FUNCTION_REGISTRY = {}


def register_function(name):
    """Decorator that registers a new autograd function."""

    def register_function_cls(cls):
        """Function performing the actual registration."""
        if name in FUNCTION_REGISTRY:
            raise ValueError("Cannot register duplicate function ({})".format(name))
        if not issubclass(cls, AutogradFunction):
            raise ValueError("Function (%s: %s) must extend AutogradFunction" % (name, cls.__name__))
        cls.name = name
        FUNCTION_REGISTRY[name] = cls
        return cls

    return register_function_cls


def get_grad_fn(name):
    """
    Returns gradient function for the CrypTen function with the specified name.
    """
    if name in FUNCTION_REGISTRY:
        return FUNCTION_REGISTRY[name]
    return None


def _ensure_tensor(input):
    """
    Converts scalars in inputs to correct tensor type.
    """
    if isinstance(input, (int, float)):
        input = torch.tensor(input)
    return input


def _inverse_broadcast(grad_output, input_size):
    """
    Performs the inverse operation of a broadcast.
    """

    # special case where input was a scalar:
    if input_size == torch.Size():
        return grad_output.sum()

    # remove leading dimensions:
    while grad_output.dim() > len(input_size):
        grad_output = grad_output.sum(0, keepdim=False)
    assert grad_output.dim() == len(input_size), "cannot perform inverse broadcast"

    # perform accumulation across broadcast dimensions:
    for dim in range(grad_output.dim()):
        if input_size[dim] == 1 and grad_output.size(dim) > 1:
            grad_output = grad_output.sum(dim, keepdim=True)
    return grad_output


class BaseAutogradContext(object):
    """
    Base implementation for AutogradContext, which saves context information
    for AutogradFunctions. Base implementation contains no-ops for all functions.
    """

    def __init__(self):
        pass

    def reset(self):
        pass

    def save_for_backward(self, value):
        pass

    def save_multiple_for_backward(self, values):
        pass

    def mark_non_differentiable(self, non_differentiable):
        pass

    def is_differentiable(self, tensor):
        raise RuntimeError("Cannot check differentiability in BaseAutogradContext.")

    @property
    def saved_tensors(self):
        raise RuntimeError("Cannot check saved_tensors in BaseAutogradContext.")


class AutogradContext(BaseAutogradContext):
    """
    Object that can be used by AutogradFunction for saving context information.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.context = []
        self.non_differentiable = []

    def save_for_backward(self, value):
        self.context.append(value)

    def save_multiple_for_backward(self, values):
        for value in values:
            self.save_for_backward(value)

    def mark_non_differentiable(self, non_differentiable):
        if not isinstance(non_differentiable, list):
            non_differentiable = [non_differentiable]
        self.non_differentiable.extend(id(x) for x in non_differentiable)

    def is_differentiable(self, tensor):
        return id(tensor) not in self.non_differentiable

    @property
    def saved_tensors(self):
        return self.context


class AutogradFunction(object):
    """
    Base implementation of a function that supports autograd.
    """

    @staticmethod
    def forward(ctx, input):
        raise NotImplementedError("Forward function not implemented.")

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward function not implemented.")

    def __str__(self):
        if hasattr(self, "name"):
            return self.name


@register_function("t")
class AutogradT(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.t()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.t()


@register_function("transpose")
class AutogradTranspose(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim1, dim2):
        ctx.save_multiple_for_backward((dim1, dim2))
        return input.transpose(dim1, dim2)

    @staticmethod
    def backward(ctx, grad_output):
        dim1, dim2 = ctx.saved_tensors
        return grad_output.transpose(dim2, dim1)


@register_function("permute")
class AutogradPermute(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dims):
        ctx.save_for_backward(dims)
        return input.permute(dims)

    @staticmethod
    def backward(ctx, grad_output):
        (dims,) = ctx.saved_tensors
        inds = [dims.index(x) for x in range(len(dims))]
        return grad_output.permute(inds)


@register_function("flip")
class AutogradFlip(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dims):
        ctx.save_for_backward(dims)
        return input.flip(dims)

    @staticmethod
    def backward(ctx, grad_output):
        (dims,) = ctx.saved_tensors
        return grad_output.flip(dims)


@register_function("clone")
class AutogradClone(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


@register_function("cat")
class AutogradCat(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=0):
        split_sections = [t.size(dim) for t in input]
        ctx.save_multiple_for_backward((dim, split_sections))
        return crypten.cat(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        dim, split_sections = ctx.saved_tensors
        return grad_output.split(split_sections, dim=dim)


@register_function("stack")
class AutogradStack(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim=0):
        ctx.save_for_backward(dim)
        return crypten.stack(input, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        (dim,) = ctx.saved_tensors
        return grad_output.unbind(dim=dim)


@register_function("view")
class AutogradView(AutogradFunction):
    @staticmethod
    def forward(ctx, input, *size):
        ctx.save_for_backward(input.size())
        return input.view(*size)

    @staticmethod
    def backward(ctx, grad_output):
        (input_size,) = ctx.saved_tensors
        return grad_output.view(input_size)


@register_function("reshape")
class AutogradReshape(AutogradFunction):
    @staticmethod
    def forward(ctx, input, *shape):
        ctx.save_for_backward(input.size())
        return input.reshape(*shape)

    @staticmethod
    def backward(ctx, grad_output):
        (size,) = ctx.saved_tensors
        return grad_output.reshape(size)


@register_function("flatten")
class AutogradFlatten(AutogradFunction):
    @staticmethod
    def forward(ctx, input, start_dim=0, end_dim=-1):
        ctx.save_for_backward(input.size())
        return input.flatten(start_dim=start_dim, end_dim=end_dim)

    @staticmethod
    def backward(ctx, grad_output):
        (size,) = ctx.saved_tensors
        return grad_output.reshape(size)


@register_function("narrow")
class AutogradNarrow(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim, start, length):
        ctx.save_multiple_for_backward((input.size(dim), dim, start, length))
        return input.narrow(dim, start, length)

    @staticmethod
    def backward(ctx, grad_output):
        size, dim, start, length = ctx.saved_tensors

        # pad is applied to dimensions in reverse order
        dim = grad_output.dim() - 1 - dim

        # pad is applied in pairs that denote the pads at the beginning and end
        # of the tensor along the given dimension
        pad = [0] * 2 * grad_output.dim()
        pad[2 * dim] = start
        pad[2 * dim + 1] = size - length - start
        return grad_output.pad(pad)


@register_function("take")
class AutogradTake(AutogradFunction):
    @staticmethod
    def forward(ctx, input, index, dim=None):
        ctx.save_multiple_for_backward((input.size(), index, dim))
        return input.take(index, dim)

    @staticmethod
    def backward(ctx, grad_output):
        size, index, dim = ctx.saved_tensors
        grad = grad_output.new(torch.zeros(size))
        if dim is None:
            grad_flat = grad.flatten()
            flat_index = index.flatten()
            grad_output_flat = grad_output.flatten()
            grad_flat[flat_index] = grad_output_flat
            grad = grad_flat.reshape(size)
        else:
            flat_index = index.flatten()
            grad_output_flat = grad_output.flatten(start_dim=dim, end_dim=(dim + index.dim() - 1))
            grad.index_add_(dim, flat_index, grad_output_flat)
        return grad


@register_function("index_select")
class AutogradIndexSelect(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim, index):
        ctx.save_multiple_for_backward([input.size(), dim, index])
        return input.index_select(dim, index)

    @staticmethod
    def backward(ctx, grad_output):
        size, dim, index = ctx.saved_tensors
        index = index.unsqueeze(0) if index.dim() == 0 else index
        return grad_output.new(torch.zeros(size)).index_add_(dim, index, grad_output)


@register_function("gather")
class AutogradGather(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim, index):
        ctx.save_multiple_for_backward([input.size(), dim, index])
        return input.gather(dim, index)

    @staticmethod
    def backward(ctx, grad_output):
        size, dim, index = ctx.saved_tensors
        return grad_output.new(torch.zeros(size)).scatter_add_(dim, index, grad_output)


@register_function("scatter")
class AutogradScatter(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim, index, src):
        output = input.scatter(dim, index, src)
        ctx.save_multiple_for_backward([dim, index])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        dim, index = ctx.saved_tensors
        size = grad_output.size()
        mask = torch.ones(size).scatter(dim, index, torch.zeros(size)).long()
        input_grad = grad_output.mul(mask)
        src_grad = grad_output.gather(dim, index)
        return (input_grad, src_grad)


@register_function("roll")
class AutogradRoll(AutogradFunction):
    @staticmethod
    def forward(ctx, input, shifts, dims=None):
        ctx.save_multiple_for_backward((shifts, dims))
        return input.roll(shifts, dims=dims)

    @staticmethod
    def backward(ctx, grad_output):
        shifts, dims = ctx.saved_tensors

        # Reverse and negate shifts
        if isinstance(shifts, (tuple, list)):
            shifts = list(shifts)
            for i, shift in enumerate(shifts):
                shifts[i] = -shift
            shifts.reverse()
        else:
            shifts = -shifts

        # Reverse dims
        if isinstance(dims, (tuple, list)):
            dims = list(dims)
            dims.reverse()

        return grad_output.roll(shifts, dims)


@register_function("squeeze")
class AutogradSqueeze(AutogradFunction):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        # preprocess inputs:
        assert len(args) >= 1
        if len(args) == 1:
            (input,) = args  # no dimension to squeeze in args
            dim = kwargs.get("dim", None)
        else:
            assert len(args) == 2
            assert "dim" not in kwargs
            input, dim = args  # dimension to squeeze in args

        # perform the actual squeeze:
        output = input.squeeze() if dim is None else input.squeeze(dim)

        # keep correct dimensions for backward pass:
        if dim is None:
            dims = [idx for idx, sz in enumerate(input.size()) if sz == 1]
        else:
            # Squeezeing non singleton dimensions is a no-op:
            dims = [dim] if input.size(dim) == 1 else []
        ctx.save_for_backward(dims)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (dims,) = ctx.saved_tensors
        grad_input = grad_output
        for dim in dims:
            grad_input = grad_input.unsqueeze(dim)
        return grad_input


@register_function("unsqueeze")
class AutogradUnsqueeze(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.save_for_backward(dim)
        return input.unsqueeze(dim)

    @staticmethod
    def backward(ctx, grad_output):
        (dim,) = ctx.saved_tensors
        return grad_output.squeeze(dim)


@register_function("__getitem__")
class AutogradGetItem(AutogradFunction):
    @staticmethod
    def forward(ctx, input, index):
        ctx.save_multiple_for_backward([input.size(), index])
        return input[index]

    @staticmethod
    def backward(ctx, grad_output):
        size, index = ctx.saved_tensors
        grad = grad_output.new(torch.zeros(size))
        grad[index] = grad_output
        return grad


@register_function("neg")
class AutogradNeg(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.neg()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


@register_function("relu")
class AutogradReLU(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        mask = input.gt(0.0)
        ctx.save_for_backward(mask)
        return input.mul(mask)

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        return grad_output.mul(mask)


@register_function("dropout")
class AutogradDropout(AutogradFunction):
    @staticmethod
    def forward(ctx, input, p=0.5, training=True, inplace=False):
        if training and inplace:
            logging.warning("CrypTen dropout does not support inplace computation during training.")

        if not training:
            if inplace:
                return input
            else:
                return input.clone()

        # training mode:
        generator = crypten.generators["global"][input.device]
        random_tensor = torch.rand(input.size(), generator=generator, device=input.device)
        boolean_mask = (random_tensor > p).to(input.device, dtype=torch.float)
        if inplace:
            result = input.mul_(boolean_mask.div(1.0 - p))
        else:
            result = input.mul(boolean_mask.div(1.0 - p))
        ctx.save_multiple_for_backward([boolean_mask, p])
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) == 0:
            return grad_output  # forward pass was run in eval mode
        boolean_mask, p = ctx.saved_tensors
        return grad_output.mul(boolean_mask.div(1.0 - p))


@register_function("_feature_dropout")
class AutogradFeatureDropout(AutogradFunction):
    @staticmethod
    def forward(ctx, input, p=0.5, training=True, inplace=False):
        if training and inplace:
            logging.warning("CrypTen _feature_dropout does not support inplace computation during training.")

        # inference mode:
        if not training:
            if inplace:
                return input
            else:
                return input.clone()

        # training mode:
        feature_dropout_size = input.size()[0:2]
        generator = crypten.generators["global"][input.device]
        random_tensor = torch.rand(feature_dropout_size, generator=generator)
        boolean_mask = (random_tensor > p).to(dtype=torch.float)
        for i in range(2, input.dim()):
            boolean_mask = boolean_mask.unsqueeze(i)
        boolean_mask, _ = torch.broadcast_tensors(boolean_mask, input.data)
        if inplace:
            result = input.mul_(boolean_mask.div(1.0 - p))
        else:
            result = input.mul(boolean_mask.div(1.0 - p))
        ctx.save_multiple_for_backward([boolean_mask, p])
        return result

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) == 0:
            return grad_output  # forward pass was run in eval mode
        boolean_mask, p = ctx.saved_tensors
        return grad_output.mul(boolean_mask.div(1.0 - p))


@register_function("tanh")
class AutogradTanh(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        activations = input.tanh()
        ctx.save_for_backward(activations)
        return activations

    @staticmethod
    def backward(ctx, grad_output):
        (activations,) = ctx.saved_tensors
        return grad_output.mul(activations.square().neg().add(1.0))


@register_function("hardtanh")
class AutogradHardtanh(AutogradFunction):
    @staticmethod
    def forward(ctx, input, min_val=-1, max_val=1):
        assert isinstance(min_val, (int, float)), "hardtanh min_val must be an int or float"
        assert isinstance(max_val, (int, float)), "hardtanh max_val must be an int or float"
        if min_val == max_val:
            grad = input.new(torch.zeros(input.size()))
            ctx.save_for_backward(grad)
            return grad + min_val

        intermediate = crypten.stack([input - min_val, max_val - input]).gt(0)
        grad = intermediate.sum(0).sub(1)
        ctx.save_for_backward(grad)

        result = grad.mul(input)
        result += (1 - intermediate[0]).mul(min_val)
        result += (1 - intermediate[1]).mul(max_val)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad.mul(grad_output)


@register_function("erf")
class AutogradErf(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.erf()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad = input.pos_pow(2).neg().exp().mul(2.0 / math.sqrt(math.pi))
        return grad_output.mul(grad)


@register_function("relu6")
class AutogradReLU6(AutogradFunction):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)
    """

    @staticmethod
    def relu6(self):
        return self.hardtanh(min_value=0, max_value=6)

    @staticmethod
    def forward(ctx, input):
        intermediate = crypten.stack([input, 6 - input]).gt(0)
        grad = intermediate.sum(0).sub(1)
        ctx.save_for_backward(grad)

        result = grad.mul(input)
        result += (1 - intermediate[1]).mul(6)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (grad,) = ctx.saved_tensors
        return grad.mul(grad_output)


@register_function("add")
class AutogradAdd(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        input = _ensure_tensor(input)
        other = _ensure_tensor(other)
        ctx.save_multiple_for_backward([input.size(), other.size()])

        return input.add(other)

    @staticmethod
    def backward(ctx, grad_output):
        input_size1, input_size2 = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.clone(), input_size1),
            _inverse_broadcast(grad_output.clone(), input_size2),
        )


@register_function("sub")
class AutogradSub(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        input = _ensure_tensor(input)
        other = _ensure_tensor(other)
        ctx.save_multiple_for_backward([input.size(), other.size()])
        return input.sub(other)

    @staticmethod
    def backward(ctx, grad_output):
        input_size1, input_size2 = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.clone(), input_size1),
            _inverse_broadcast(grad_output.clone(), input_size2).neg(),
        )


@register_function("__rsub__")
class AutogradRSub(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        input = _ensure_tensor(input)
        other = _ensure_tensor(other)
        ctx.save_multiple_for_backward([input.size(), other.size()])
        return (-input).add(other)

    @staticmethod
    def backward(ctx, grad_output):
        input_size1, input_size2 = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.clone(), input_size1).neg(),
            _inverse_broadcast(grad_output.clone(), input_size2),
        )


@register_function("mul")
class AutogradMul(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        input = _ensure_tensor(input)
        other = _ensure_tensor(other)
        ctx.save_multiple_for_backward([input, other])
        return input.mul(other)

    @staticmethod
    def backward(ctx, grad_output):
        self_, other = ctx.saved_tensors
        return (
            _inverse_broadcast(grad_output.mul(other), self_.size()),
            _inverse_broadcast(grad_output.mul(self_), other.size()),
        )


@register_function("matmul")
class AutogradMatMul(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_multiple_for_backward([input, other])
        return input.matmul(other)

    @staticmethod
    def backward(ctx, grad_output):
        self_, other = ctx.saved_tensors

        # Cache sizes for inverse_broadcast
        self_size = self_.size()
        other_size = other.size()

        # Deal with vectors that are represented by a
        # < 2 dimensional tensor
        if self_.dim() < 2:
            self_ = self_.unsqueeze(0)
            grad_output = grad_output.unsqueeze(0)

        if other.dim() < 2:
            other = other.unsqueeze(1)
            grad_output = grad_output.unsqueeze(1)

        # Compute gradients
        self_grad = grad_output.matmul(other.transpose(-2, -1))
        other_grad = self_.transpose(-2, -1).matmul(grad_output)

        # Fix gradient sizes for vector inputs
        if len(self_size) < 2:
            self_grad = self_grad.squeeze()
            if self_grad.dim() < 1:
                self_grad = self_grad.unsqueeze(0)

        if len(other_size) < 2:
            other_grad = other_grad.squeeze()
            if other_grad.dim() < 1:
                other_grad = other_grad.unsqueeze(0)

        return (
            _inverse_broadcast(self_grad, self_size),
            _inverse_broadcast(other_grad, other_size),
        )


@register_function("div")
class AutogradDiv(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        if crypten.is_encrypted_tensor(other):
            other_reciprocal = other.reciprocal()
            ctx.save_multiple_for_backward([input, other_reciprocal])
            return input.mul(other_reciprocal)
        else:
            ctx.save_multiple_for_backward([input.size(), other])
            return input.div(other)

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors

        # saved is a list of [input, other_reciprocal]
        if crypten.is_encrypted_tensor(saved[1]):
            input, other_reciprocal = saved
            grad_input = other_reciprocal.mul(grad_output)
            grad_other = other_reciprocal.square().mul(input).mul(grad_output).neg()
            return (
                _inverse_broadcast(grad_input, input.size()),
                _inverse_broadcast(grad_other, other_reciprocal.size()),
            )
        # saved is a public tensor or scalar
        else:
            input_size, other = saved
            grad_input = grad_output.div(other)
            if torch.is_tensor(other):
                return _inverse_broadcast(grad_input, input_size)
            else:
                return grad_input


@register_function("__rtruediv__")
class AutogradRDiv(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        reciprocal = input.reciprocal()
        ctx.save_multiple_for_backward([reciprocal, other])
        return reciprocal.mul(other)

    @staticmethod
    def backward(ctx, grad_output):
        reciprocal, other = ctx.saved_tensors
        grad_input = reciprocal.square().mul(other).mul(grad_output).neg()
        grad_input = _inverse_broadcast(grad_input, reciprocal.size())

        if torch.is_tensor(other) or crypten.is_encrypted_tensor(other):
            grad_other = reciprocal.mul(grad_output)
            grad_other = _inverse_broadcast(grad_other, other.size())
            return (grad_input, grad_other)
        else:
            return grad_input


@register_function("polynomial")
class AutogradPolynomial(AutogradFunction):
    @staticmethod
    def forward(ctx, input, coeffs, func="mul"):
        ctx.mark_non_differentiable(coeffs)
        if isinstance(coeffs, (list, tuple)):
            coeffs = torch.tensor(coeffs)
        ctx.save_multiple_for_backward([input, coeffs, func])
        return input.polynomial(coeffs, func)

    @staticmethod
    def backward(ctx, grad_output):
        input, coeffs, func = ctx.saved_tensors
        coeffs *= torch.arange(coeffs.size(0)).add(1)
        return input.polynomial(coeffs[1:], func).add(coeffs[0]).mul_(grad_output)


@register_function("pow")
class AutogradPow(AutogradFunction):
    @staticmethod
    def forward(ctx, input, power):
        grad_pow = input.pow(power - 1)
        grad = grad_pow.mul(power)
        ctx.save_multiple_for_backward([input, grad])
        return grad_pow.mul(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, grad = ctx.saved_tensors
        return grad.mul_(grad_output)


@register_function("pos_pow")
class AutogradPosPow(AutogradFunction):
    @staticmethod
    def forward(ctx, input, power):
        if isinstance(power, int) or (isinstance(power, float) and int(power) == power):
            ctx.save_multiple_for_backward([input, power])
            return input.pow(power)
        else:
            log_input = input.log()
            ctx.save_multiple_for_backward([log_input, power])
            return log_input.mul(power).exp()

    @staticmethod
    def backward(ctx, grad_output):
        input, power = ctx.saved_tensors
        if isinstance(power, int) or (isinstance(power, float) and int(power) == power):
            return input.pow(power - 1.0).mul_(power).mul_(grad_output)
        else:
            return input.mul(power - 1.0).mul_(power).exp().mul(grad_output)


@register_function("square")
class AutogradSquare(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.square()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output.mul(input.mul(2.0))


@register_function("sqrt")
class AutogradSqrt(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        inv_sqrt = input.inv_sqrt()
        ctx.save_for_backward(inv_sqrt)
        return inv_sqrt.mul(input)

    @staticmethod
    def backward(ctx, grad_output):
        (inv_sqrt,) = ctx.saved_tensors
        return inv_sqrt.div_(2).mul_(grad_output)


@register_function("exp")
class AutogradExp(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        output = input.exp()
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        return output.mul(grad_output)


@register_function("log")
class AutogradLog(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.log()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output.div(input)


@register_function("reciprocal")
class AutogradReciprocal(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        reciprocal = input.reciprocal()
        ctx.save_for_backward(reciprocal)
        return reciprocal

    @staticmethod
    def backward(ctx, grad_output):
        (reciprocal,) = ctx.saved_tensors
        return grad_output.neg().mul_(reciprocal.square())


@register_function("dot")
class AutogradDot(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_multiple_for_backward([input, other])
        return input.dot(other)

    @staticmethod
    def backward(ctx, grad_output):
        self_, other = ctx.saved_tensors
        return (grad_output.mul(other), grad_output.mul(self_))


@register_function("ger")
class AutogradGer(AutogradFunction):
    @staticmethod
    def forward(ctx, input, other):
        ctx.save_multiple_for_backward([input, other])
        return input.ger(other)

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        return (grad_output.matmul(other), input.matmul(grad_output))


@register_function("sin")
class AutogradSin(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        cossin = input.cossin()
        ctx.save_for_backward(cossin[0])
        return cossin[1]

    @staticmethod
    def backward(ctx, grad_output):
        (cos,) = ctx.saved_tensors
        return grad_output.mul(cos)


@register_function("cos")
class AutogradCos(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        cossin = input.cossin()
        ctx.save_for_backward(cossin[1])
        return cossin[0]

    @staticmethod
    def backward(ctx, grad_output):
        (sin,) = ctx.saved_tensors
        return grad_output.mul(sin.neg_())


@register_function("cosine_similarity")
class AutogradCosineSimilarity(AutogradFunction):
    @staticmethod
    def forward(ctx, x1, x2, dim=1, eps=None):
        assert x1.size() == x2.size(), "cosine_similarity sizes must match"

        # Handle 0-d case
        zero_dim = x1.dim() == 0
        if zero_dim:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)

        # Handle 1-d vectors
        if x1.size(dim) == 1:
            ctx.save_multiple_for_backward([dim, zero_dim])
            return x1.mul(x2).sign().squeeze(dim)

        if not isinstance(x2, crypten.CrypTensor):
            x2 = x1.new(x2)

        xy = crypten.stack([x1, x2], dim=0)  # [x, y]
        norm_sq = xy.square().sum(dim=(dim + 1))  # [||x||^2, ||y||^2]
        inv_norms = norm_sq.inv_sqrt()  # [1 / ||x||, 1 / ||y||]

        ctx.save_multiple_for_backward((xy, inv_norms, dim))

        inv_norm = inv_norms.prod(0)  # 1 / ||x||||y||
        dot = xy.prod(0).sum(dim)  # x . y
        return dot.mul(inv_norm)

    @staticmethod
    def backward(ctx, grad_output):
        # Handle 1-d vectors
        if len(ctx.saved_tensors) == 2:
            (dim, zero_dim) = ctx.saved_tensors
            zeros = torch.zeros(grad_output.size()).unsqueeze(dim)
            result = grad_output.new(zeros, device=grad_output.device)
            if zero_dim:
                result = result.squeeze()
            return result, result.clone()

        xy, inv_norms, dim = ctx.saved_tensors

        dot = xy.prod(0).sum(dim, keepdim=True)
        inv_norms = inv_norms.unsqueeze(dim + 1)
        sq_inv_norms = inv_norms.square()

        xy_normalized = xy.mul(sq_inv_norms)
        yx = xy.roll(1, 0)

        grads = yx.sub(dot.mul(xy_normalized)).mul(inv_norms.prod(0))
        grads = grads.mul(grad_output.unsqueeze(dim))

        x_grad, y_grad = grads
        return x_grad, y_grad


@register_function("abs")
class AutogradAbs(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        sign = input.sign()
        ctx.save_for_backward(sign)
        return input.mul(sign)

    @staticmethod
    def backward(ctx, grad_output):
        (sign,) = ctx.saved_tensors
        return grad_output.mul(sign)


@register_function("sign")
class AutogradSign(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.sub(grad_output)


@register_function("norm")
class AutogradNorm(AutogradFunction):
    @staticmethod
    def forward(ctx, input, p="fro", dim=None, keepdim=False):
        if p == float("inf"):
            sign = input.sign()
            if dim is None:
                input = input.mul(sign)
                argmax = input.argmax(one_hot=True)
                max = input.mul(argmax).sum()
            else:
                max, argmax = input.mul(sign).max(dim, keepdim=keepdim, one_hot=True)

            ctx.save_multiple_for_backward((sign, argmax, p, dim, keepdim))
            return max
        else:
            if dim is None:
                norm = input.norm(p=p)
            else:
                norm = input.norm(p=p, dim=dim, keepdim=keepdim)
            ctx.save_multiple_for_backward((input, norm, p, dim, keepdim))
            return norm

    @staticmethod
    def backward(ctx, grad_output):
        input, norm, p, dim, keepdim = ctx.saved_tensors
        if not keepdim and dim is not None:
            grad_output.unsqueeze(dim)

        if p == 2 or p == "fro":
            return grad_output.mul(input.div(norm))
        elif p == float("inf"):
            sign, argmax = input, norm
            return grad_output.mul(argmax).mul(sign)
        else:
            sign = input.sign()
            abs = input.mul(sign)
            return grad_output.mul(abs.div(norm).pos_pow(p - 1).mul(sign))


@register_function("sum")
class AutogradSum(AutogradFunction):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        # preprocess inputs:
        assert len(args) >= 1
        if len(args) == 1:
            (input,) = args  # no dimension to sum over in args
            dim = kwargs.get("dim", None)
        else:
            assert len(args) == 2
            assert "dim" not in kwargs
            input, dim = args  # dimension to sum over in args
        keepdim = kwargs.get("keepdim", False)

        # compute sum:
        ctx.save_multiple_for_backward((input.size(), dim, keepdim))
        return input.sum(dim, keepdim=keepdim) if dim is not None else input.sum()

    @staticmethod
    def backward(ctx, grad_output):
        input_size, dim, keepdim = ctx.saved_tensors

        # Handle special case where input is 0-dimensional
        if len(input_size) == 0:
            return grad_output

        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(torch.ones(input_size))


@register_function("cumsum")
class AutogradCumsum(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.save_for_backward(dim)
        return input.cumsum(dim)

    @staticmethod
    def backward(ctx, grad_output):
        (dim,) = ctx.saved_tensors
        return grad_output.flip(dim).cumsum(dim).flip(dim)


@register_function("trace")
class AutogradTrace(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.size()[0])
        return input.trace()

    @staticmethod
    def backward(ctx, grad_output):
        (size,) = ctx.saved_tensors
        return grad_output.new(torch.eye(size)).mul_(grad_output)


@register_function("mean")
class AutogradMean(AutogradFunction):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        # preprocess inputs:
        assert len(args) >= 1
        if len(args) == 1:
            (input,) = args  # no dimension to average over in args
            dim = kwargs.get("dim", None)
        else:
            assert len(args) == 2
            assert "dim" not in kwargs
            input, dim = args  # dimension to average over in args
        keepdim = kwargs.get("keepdim", False)

        # compute mean:
        ctx.save_multiple_for_backward((input.size(), dim, keepdim))
        return input.mean(dim, keepdim=keepdim) if dim is not None else input.mean()

    @staticmethod
    def backward(ctx, grad_output):
        input_size, dim, keepdim = ctx.saved_tensors

        # Handle special case where input is 0-dimensional
        if len(input_size) == 0:
            return grad_output

        nelement = float(reduce(lambda x, y: x * y, input_size) if dim is None else input_size[dim])
        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(torch.ones(input_size, device=grad_output.device).div_(nelement))


@register_function("var")
class AutogradVariance(AutogradFunction):
    @staticmethod
    def forward(ctx, self, *args, **kwargs):
        # preprocess inputs:
        if len(args) == 0:
            dim = None
            unbiased = kwargs.get("unbiased", False)
            keepdim = False
            mean = self.mean()
        elif len(args) == 1:
            dim = args[0]
            unbiased = kwargs.get("unbiased", False)
            keepdim = kwargs.get("keepdim", False)
        elif len(args) == 2:
            dim, unbiased = args[0], args[1]
            keepdim = kwargs.get("keepdim", False)
        else:
            dim, unbiased, keepdim = args[0], args[1], args[2]

        if dim is not None:  # dimension is specified
            mean = self.mean(dim, keepdim=True)

        # Compute square error
        result = (self - mean).square()
        if dim is None:
            result = result.sum()
        else:
            result = result.sum(dim, keepdim=keepdim)

        # Determine divisor
        divisor = self.nelement() // result.nelement()
        if not unbiased:
            divisor -= 1

        # compute variance:
        ctx.save_multiple_for_backward((self, mean, divisor, dim, keepdim))
        return self.var(dim, unbiased=unbiased, keepdim=keepdim) if dim is not None else self.var(unbiased=unbiased)

    @staticmethod
    def backward(ctx, grad_output):
        input, mean, divisor, dim, keepdim = ctx.saved_tensors

        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)

        numerator = input.sub(mean).mul(2).mul(grad_output)
        if divisor == 0:
            return numerator
        return numerator.div(divisor)


@register_function("min")
class AutogradMin(AutogradFunction):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        # preprocess inputs:
        assert len(args) >= 1
        if len(args) == 1:
            (input,) = args  # no dimension to min over in args
            dim = kwargs.pop("dim", None)  # remove dim from kwargs after obtaining it
        else:
            assert len(args) == 2
            assert "dim" not in kwargs
            input, dim = args  # dimension to min over in args
        keepdim = kwargs.get("keepdim", False)
        one_hot = kwargs.get("one_hot", True)

        # find minimum value (and corresponding argmin):
        if dim is None:
            argmin = input.argmin(one_hot=one_hot)
            min = input.mul(argmin).sum()
        else:
            min, argmin = input.min(dim, **kwargs)

        # save context and return:
        ctx.save_multiple_for_backward((dim, keepdim, argmin, one_hot))
        if dim is None:
            return min
        else:
            ctx.mark_non_differentiable(argmin)
            return min, argmin

    @staticmethod
    def backward(ctx, grad_output):
        dim, keepdim, argmin, one_hot = ctx.saved_tensors
        assert one_hot, (
            "cannot backpropagate through min layer that does not"
            "use one-hot representation because private indexing is unsupported"
        )
        # Handle special case where input is 0-dimensional
        if len(argmin.size()) == 0:
            return grad_output

        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(argmin)


@register_function("max")
class AutogradMax(AutogradFunction):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        # preprocess inputs:
        assert len(args) >= 1
        if len(args) == 1:
            (input,) = args  # no dimension to max over in args
            dim = kwargs.pop("dim", None)  # remove dim from kwargs after obtaining it
        else:
            assert len(args) == 2
            assert "dim" not in kwargs
            input, dim = args  # dimension to max over in args
        keepdim = kwargs.get("keepdim", False)
        one_hot = kwargs.get("one_hot", True)
        # find maximum value (and corresponding argmax):
        if dim is None:
            shape = input.size()
            input_flat = input.flatten()
            max, argmax = input_flat.max(0, **kwargs)
            argmax = argmax.reshape(shape)
        else:
            max, argmax = input.max(dim, **kwargs)

        # save context and return:
        ctx.save_multiple_for_backward((dim, keepdim, argmax, one_hot))
        if dim is None:
            return max
        else:
            ctx.mark_non_differentiable(argmax)
            return max, argmax

    @staticmethod
    def backward(ctx, grad_output):
        dim, keepdim, argmax, one_hot = ctx.saved_tensors
        assert one_hot, (
            "cannot backpropagate through max layer that does not"
            "use one-hot representation because private indexing is unsupported"
        )
        # Handle special case where input is 0-dimensional
        if len(argmax.size()) == 0:
            return grad_output

        if not keepdim and dim is not None:
            grad_output = grad_output.unsqueeze(dim)
        return grad_output.mul(argmax)


@register_function("sigmoid")
class AutogradSigmoid(AutogradFunction):
    @staticmethod
    def forward(ctx, input):
        probs = input.sigmoid()
        ctx.save_for_backward(probs)
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        (probs,) = ctx.saved_tensors
        return grad_output.mul(probs).mul_(probs.neg().add_(1.0))


@register_function("softmax")
class AutogradSoftmax(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim):
        probs = input.softmax(dim)
        ctx.save_multiple_for_backward([probs, dim])
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        probs, dim = ctx.saved_tensors
        if grad_output.dim() == 0 or grad_output.size(dim) == 1:
            return grad_output.new(torch.zeros(grad_output.size()))
        return grad_output.add(-probs.mul(grad_output).sum(dim, keepdim=True)).mul_(probs)


@register_function("log_softmax")
class AutogradLogSoftmax(AutogradFunction):
    @staticmethod
    def forward(ctx, input, dim):
        probs = input.log_softmax(dim)
        ctx.save_multiple_for_backward([probs, dim])
        return probs

    @staticmethod
    def backward(ctx, grad_output):
        probs, dim = ctx.saved_tensors
        if grad_output.dim() == 0 or grad_output.size(dim) == 1:
            return grad_output.new(torch.zeros(grad_output.size()))
        z = probs.exp()
        result = grad_output - z * grad_output.sum(dim, keepdim=True)
        return result


@register_function("pad")
class AutogradPad(AutogradFunction):
    @staticmethod
    def forward(ctx, input, padding, value=0.0, mode="constant"):
        ctx.save_for_backward(padding)
        output = input.pad(padding, value=value, mode=mode)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (padding,) = ctx.saved_tensors
        for idx in range(0, len(padding), 2):
            dim = grad_output.dim() - (idx // 2) - 1
            start = padding[idx]
            end = grad_output.size(dim) - padding[idx + 1] - padding[idx]
            grad_output = grad_output.narrow(dim, start, end)
        return grad_output


@register_function("avg_pool2d")
class AutogradAvgPool2D(AutogradFunction):
    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0, ceil_mode=False):
        # preprocess inputs:
        if stride is None:
            stride = kernel_size
        if isinstance(stride, (int, float)):
            stride = (stride, stride)
        if isinstance(padding, (int, float)):
            padding = (padding, padding)
        if isinstance(kernel_size, (int, float)):
            kernel_size = (kernel_size, kernel_size)

        # perform average pooling:
        output = input.avg_pool2d(kernel_size, padding=padding, stride=stride)

        # store information for backward pass:
        ctx.save_multiple_for_backward((input.shape, output, kernel_size, padding, stride))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Computes the gradient with respect to the input"""
        input_shape, output, kernel_size, padding, stride = ctx.saved_tensors

        in_channels = input_shape[-3]
        # compute as d conv2d / d input with kernel as average filter
        kernel = torch.ones(in_channels, 1, kernel_size[0], kernel_size[1], device=grad_output.device) / (
            kernel_size[0] * kernel_size[1]
        )

        grad_input_padding = _grad_input_padding(
            grad_output,
            input_shape,
            stride,
            padding,
            kernel_size,
            dilation=([1] * len(stride)),
        )

        # set groups=in_channels so input gradient is computed per channel
        if isinstance(grad_output, crypten.CrypTensor):
            return grad_output.conv_transpose2d(
                kernel,
                bias=None,
                stride=stride,
                padding=padding,
                output_padding=grad_input_padding,
                groups=in_channels,
            )

        return torch.conv_transpose2d(
            grad_output,
            kernel,
            bias=None,
            stride=stride,
            padding=padding,
            output_padding=grad_input_padding,
            groups=in_channels,
        )


@register_function("max_pool2d")
class AutogradMaxPool2D(AutogradFunction):
    @staticmethod
    def forward(
        ctx,
        input,
        kernel_size,
        padding=0,
        stride=None,
        dilation=1,
        ceil_mode=False,
        return_indices=False,
    ):
        # preprocess inputs:
        if stride is None:
            stride = kernel_size
        if isinstance(stride, (int, float)):
            stride = (stride, stride)
        if isinstance(padding, (int, float)):
            padding = (padding, padding)
        if isinstance(dilation, (int, float)):
            dilation = (dilation, dilation)

        # perform max pooling:
        # Note return_indices is required to be True to computing backward.
        output, indices = input.max_pool2d(
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )

        # store information for backward pass and return:
        ctx.save_multiple_for_backward((input.size(), indices, kernel_size, padding, stride, dilation, ceil_mode))
        if return_indices:
            ctx.mark_non_differentiable(indices)
            return output, indices
        else:
            return output

    @staticmethod
    def backward(ctx, grad_output):
        (
            output_size,
            indices,
            kernel_size,
            padding,
            stride,
            dilation,
            ceil_mode,
        ) = ctx.saved_tensors
        assert padding[0] == padding[1], "padding must be same in all axes"
        return grad_output._max_pool2d_backward(
            indices,
            kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ceil_mode=ceil_mode,
            output_size=output_size,
        )


@register_function("conv1d")
class AutogradConv1D(AutogradFunction):
    @staticmethod
    def forward(ctx, input, kernel, padding=0, stride=1, dilation=1, groups=1):
        if isinstance(stride, (int, float)):
            stride = (stride,)
        if isinstance(padding, (int, float)):
            padding = (padding,)
        if isinstance(dilation, (int, float)):
            dilation = (dilation,)
        ctx.save_multiple_for_backward((input, kernel, padding, stride, dilation, groups))
        return input.conv1d(kernel, padding=padding, stride=stride, dilation=dilation, groups=groups)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient function adapts code from:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/grad.py

        # get input, kernel, and sizes:
        input, kernel, padding, stride, dilation, groups = ctx.saved_tensors
        batch_size = input.size(0)
        out_channels, in_channels, kernel_size = kernel.size()
        in_channels *= groups
        assert input.size(1) == in_channels, "wrong number of input channels"
        assert grad_output.size(1) == out_channels, "wrong number of output channels"
        assert grad_output.size(0) == batch_size, "wrong batch size"

        # TODO: Implement conv1d gradient under following condition:
        if groups > 1 and input.size(1) > groups:
            raise NotImplementedError("conv1d backward with groups > 1 and in_channels > groups not implemented")

        # compute gradient with respect to input:
        output_padding = _grad_input_padding(
            grad_output,
            input.size(),
            stride,
            padding,
            (kernel_size,),
            dilation=dilation,
        )
        grad_input = grad_output.conv_transpose1d(
            kernel,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )

        # compute gradient with respect to kernel:
        grad_output = grad_output.repeat(1, in_channels // groups, 1)
        grad_output = grad_output.view(grad_output.size(0) * grad_output.size(1), 1, grad_output.size(2))
        input = input.view(1, input.size(0) * input.size(1), input.size(2))
        grad_kernel = input.conv1d(
            grad_output,
            stride=dilation,
            padding=padding,
            dilation=stride,
            groups=in_channels * batch_size,
        )
        grad_kernel = grad_kernel.view(batch_size, grad_kernel.size(1) // batch_size, grad_kernel.size(2))
        grad_kernel = grad_kernel.sum(dim=0)
        grad_kernel = grad_kernel.view(in_channels // groups, out_channels, grad_kernel.size(1))
        grad_kernel = grad_kernel.transpose(0, 1).narrow(2, 0, kernel_size)
        return (grad_input, grad_kernel)


@register_function("conv2d")
class AutogradConv2D(AutogradFunction):
    @staticmethod
    def forward(ctx, input, kernel, stride=1, padding=0, dilation=1, groups=1):
        if isinstance(stride, (int, float)):
            stride = (stride, stride)
        if isinstance(padding, (int, float)):
            padding = (padding, padding)
        if isinstance(dilation, (int, float)):
            dilation = (dilation, dilation)
        ctx.save_multiple_for_backward((input, kernel, padding, stride, dilation, groups))
        return input.conv2d(kernel, stride=stride, padding=padding, dilation=dilation, groups=groups)

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient function adapts code from:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/grad.py

        # get input, kernel, and sizes:
        input, kernel, padding, stride, dilation, groups = ctx.saved_tensors
        batch_size = input.size(0)
        out_channels, in_channels, kernel_size_y, kernel_size_x = kernel.size()
        in_channels *= groups
        assert input.size(1) == in_channels, "wrong number of input channels"
        assert grad_output.size(1) == out_channels, "wrong number of output channels"
        assert grad_output.size(0) == batch_size, "wrong batch size"

        # TODO: Implement conv2d gradient under following condition:
        if groups > 1 and input.size(1) > groups:
            raise NotImplementedError("conv2d backward with groups > 1 and in_channels > groups not implemented")

        # compute gradient with respect to input:
        output_padding = _grad_input_padding(
            grad_output,
            input.size(),
            stride,
            padding,
            (kernel_size_y, kernel_size_x),
            dilation=dilation,
        )
        grad_input = grad_output.conv_transpose2d(
            kernel,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        )

        # compute gradient with respect to kernel:
        grad_output = grad_output.repeat(1, in_channels // groups, 1, 1)
        grad_output = grad_output.view(
            grad_output.size(0) * grad_output.size(1),
            1,
            grad_output.size(2),
            grad_output.size(3),
        )
        input = input.view(1, input.size(0) * input.size(1), input.size(2), input.size(3))
        # dilation and stride are swapped based on PyTorch's conv2d_weight implementation
        grad_kernel = input.conv2d(
            grad_output,
            stride=dilation,
            padding=padding,
            dilation=stride,
            groups=in_channels * batch_size,
        )
        grad_kernel = grad_kernel.view(
            batch_size,
            grad_kernel.size(1) // batch_size,
            grad_kernel.size(2),
            grad_kernel.size(3),
        )
        grad_kernel = (
            grad_kernel.sum(0)
            .view(
                in_channels // groups,
                out_channels,
                grad_kernel.size(2),
                grad_kernel.size(3),
            )
            .transpose(0, 1)
        )
        grad_kernel = grad_kernel.narrow(2, 0, kernel_size_y)
        grad_kernel = grad_kernel.narrow(3, 0, kernel_size_x)
        return (grad_input, grad_kernel)


@register_function("batchnorm")
class AutogradBatchNorm(AutogradFunction):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        running_mean=None,
        running_var=None,
        training=False,
        eps=1e-05,
        momentum=0.1,
        inv_var=None,
    ):
        """
        Computes forward step of batch norm by normalizing x
            and returning weight * x_norm + bias.

        Running mean and var are computed over the `C` dimension for an input
        of size `(N, C, +)`.

        Note: inv_var can introduce precision errors due to sqrt and division
            particularly when the number of samples in a batch is small.

        Args:
            ctx (autograd_cyptensor.AutogradContext): context which
                stores parameters such as weight and bias for backward step.
            input (tuple of torch.tensors or cryptensor):
                containing (x, weight, bias) with shapes `(N, C, +)`, `C`, and `C`
                in turn.
            training (bool): if training is True, running mean and var are
                updated with the momentum factor and stored in module. Forward
                is performed using batch statistics. If training is False,
                running statistics are used and therefore cannot be none.
            running_mean (torch.tensor or cryptensor): with shape `C`
            running_var (torch.tensor or cryptensor): with shape `C`
            eps (float): specifies epsilon used for numerical precision in inv_var
            momentum (float): moment factor used in updating running mean and var.

        Returns: (weight * normalized input + bias) of shape `(N, C, +)`.
        """

        # determine dimensions over which means and variances are computed:
        stats_dimensions = list(range(x.dim()))
        stats_dimensions.pop(1)

        # shape for broadcasting statistics with input:
        broadcast_shape = [1] * x.dim()
        broadcast_shape[1] = x.shape[1]

        # compute mean and variance, track batch statistics:
        if training:
            mean = x.mean(stats_dimensions)
            variance = x.var(stats_dimensions, unbiased=True)
            if running_mean is not None and running_var is not None:
                running_var.set(running_var * (1.0 - momentum) + variance * momentum)
                running_mean.set(running_mean * (1.0 - momentum) + mean * momentum)
        else:
            if running_mean is None or running_var is None:
                raise ValueError("Must provide running_mean and running_var when training is False")
            mean = running_mean
            variance = running_var

        if training or inv_var is None:
            # compute inverse variance:
            if torch.is_tensor(variance):
                inv_var = 1.0 / torch.sqrt(variance + eps)
            else:
                inv_var = (variance + eps).inv_sqrt()

        # reshape shape (C) to broadcastable (1, C, 1, +):
        mean = mean.reshape(broadcast_shape)
        inv_var = inv_var.reshape(broadcast_shape)
        weight = weight.reshape(broadcast_shape)
        bias = bias.reshape(broadcast_shape)

        # compute z-scores:
        x_norm = (x - mean) * inv_var

        # save context and return:
        ctx.save_multiple_for_backward((x_norm, weight, inv_var, training))
        return x_norm * weight + bias

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient with respect to x, weight, and bias.

        Statistics are assumed to be computed along dimension C
        for an input of shape (N, C, ...). Note, partials with respect to
        the input treat mean and variance as constants similar to torch.

        Args:
            ctx (autograd_cyptensor.AutogradContext): context containing
                x_norm, weight, and inv_var. Note weight
                and inv_var must be broadcastable with grad_output.
            grad_output (cryptensor): batchnorm output of shape (N, C, +).

        Returns:
            x_grad (cryptensor): gradient with respect to x with shape (N, C, +).
            weight_grad (cryptensor): gradient with respect to the weight of
                with shape (C).
            bias_grad (cryptensor): gradient with respect to bias of shape (C).
        """

        # retrieve context:
        x_norm, weight, inv_var, training = ctx.saved_tensors

        # determine dimensions over which means and variances are computed:
        stats_dimensions = list(range(len(grad_output.shape)))
        stats_dimensions.pop(1)

        # shape for broadcasting statistics with output gradient:
        broadcast_shape = [1] * grad_output.dim()
        broadcast_shape[1] = grad_output.shape[1]

        # compute gradient w.r.t. weight:
        grad_weight = grad_output.mul(x_norm)
        grad_weight = grad_weight.sum(stats_dimensions)

        # compute gradient w.r.t. bias:
        grad_bias = grad_output.sum(stats_dimensions)

        # compute gradient with respect to the input:
        grad_output = grad_output.mul(weight)
        grad_input = grad_output.mul(inv_var)
        if training:
            # compute gradient term that is due to the mean:
            num_element = reduce(lambda x, y: x * y, [grad_output.size(d) for d in stats_dimensions])
            grad_mean = grad_output.sum(stats_dimensions)
            grad_mean = grad_mean.reshape(broadcast_shape)
            grad_mean = grad_mean.mul(inv_var.div(-num_element))

            # compute gradient term that is due to the standard deviation:
            grad_std = x_norm.mul(grad_output).sum(stats_dimensions)
            grad_std = grad_std.reshape(broadcast_shape)
            grad_std = x_norm.mul(grad_std).mul(inv_var.div(-num_element))

            # put all the terms together:
            grad_input = grad_input.add(grad_mean).add(grad_std)

        # return gradients:
        return (grad_input, grad_weight, grad_bias)


@register_function("binary_cross_entropy")
class AutogradBinaryCrossEntropy(AutogradFunction):
    @staticmethod
    def forward(ctx, pred, target, skip_forward=False):
        ctx.mark_non_differentiable(target)
        ctx.save_multiple_for_backward([pred, target])
        if skip_forward:
            return pred.new(0)

        # Compute full forward pass
        log_pos, log_neg = crypten.stack([pred, 1.0 - pred]).log(input_in_01=True).unbind(dim=0)
        loss_values = target * log_pos + ((1.0 - target) * log_neg)
        return -(loss_values.mean())

    @staticmethod
    def backward(ctx, grad_output):
        pred, target = ctx.saved_tensors
        rec_pos, rec_neg = crypten.stack([pred, 1.0 - pred]).reciprocal(input_in_01=True).unbind(dim=0)
        grad = (rec_neg * (1.0 - target)) - rec_pos * target
        return grad.div_(target.nelement()).mul_(grad_output)


@register_function("binary_cross_entropy_with_logits")
class AutogradBinaryCrossEntropyWithLogits(AutogradFunction):
    @staticmethod
    def forward(ctx, logit, target, skip_forward=False):
        sigmoid_out = logit.sigmoid()
        assert sigmoid_out.size() == target.size(), "Incorrect input sizes for binary_cross_entropy_with_logits"
        ctx.mark_non_differentiable(target)
        ctx.save_multiple_for_backward([target, sigmoid_out])
        if skip_forward:
            return sigmoid_out.new(0)

        # Compute full forward pass
        log_pos, log_neg = crypten.stack([sigmoid_out, 1.0 - sigmoid_out]).log(input_in_01=True).unbind(dim=0)
        loss_values = target * log_pos + ((1.0 - target) * log_neg)
        return -(loss_values.mean())

    @staticmethod
    def backward(ctx, grad_output):
        target, sigmoid_out = ctx.saved_tensors
        return (sigmoid_out - target).div(target.nelement()).mul_(grad_output)


@register_function("rappor_loss")
class AutogradRAPPORLoss(AutogradFunction):
    @staticmethod
    def forward(ctx, logit, target, alpha, skip_forward=False):
        assert logit.size() == target.size(), "Logit and target sizes must match for rappor loss"
        pred = logit.sigmoid()
        ctx.mark_non_differentiable(target)
        if alpha == 0.0:
            ctx.save_multiple_for_backward([target, pred, None, alpha])
            pred_normalized = pred
        else:
            pred_normalized = alpha * pred + (1 - alpha) * (1 - pred)
            grad_correction = pred * (1 - pred)
            grad_correction *= (pred_normalized * (1 - pred_normalized)).reciprocal(input_in_01=True)
            ctx.save_multiple_for_backward([target, pred_normalized, grad_correction, alpha])

        if skip_forward:
            return pred.new(0)

        log_pos, log_neg = crypten.stack([pred_normalized, 1.0 - pred_normalized]).log(input_in_01=True).unbind(dim=0)

        loss_values = target * log_pos + (1.0 - target) * log_neg
        return -(loss_values.mean())

    @staticmethod
    def backward(ctx, grad_output):
        target, pred_normalized, grad_correction, alpha = ctx.saved_tensors

        if alpha == 0.0:
            return (pred_normalized - target).div(target.nelement()).mul_(grad_output)

        grad = (pred_normalized - target).div(target.nelement())
        grad *= 2 * alpha - 1
        grad *= grad_correction
        return grad.mul_(grad_output)


@register_function("cross_entropy")
class AutogradCrossEntropy(AutogradFunction):
    @staticmethod
    def forward(ctx, pred, target, skip_forward=False):
        # NOTE: target is assumed to be one-hot vector.
        assert pred.size() == target.size()

        # Ignore batch dimension
        dim = 1 if pred.dim() > 1 else 0
        softmax = pred.softmax(dim)

        ctx.save_multiple_for_backward([softmax, target])
        ctx.mark_non_differentiable(target)
        if skip_forward:
            return softmax.new(0)

        # Compute full forward pass
        loss_values = softmax.log(input_in_01=True).mul_(target).neg_()
        return loss_values.sum().div_(target.size(0))

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target = ctx.saved_tensors
        loss_grad = softmax.sub(target)
        return loss_grad.div_(target.size(0)).mul_(grad_output)
