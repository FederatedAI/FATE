#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
from collections import OrderedDict

import torch
import torch.onnx.symbolic_helper as sym_help

from ..cryptensor import CrypTensor
from ..functions.pooling import _adaptive_pool2d_helper
from ... import mpc


class Module:
    """
    Base Module class that mimics the torch.nn.Module class.
    """

    # allow for versioning of modules:
    _version = 1
    SUPPORTS_PLAINTEXT_INPUTS = False

    def __init__(self):
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._modules = OrderedDict()
        self.encrypted = False
        self.train()

    def __repr__(self):
        encrypted_str = "encrypted" if self.encrypted else "unencrypted"
        return f"{type(self).__name__} {encrypted_str} module"

    @staticmethod
    def from_onnx(attributes=None):
        """
        Constructs a CrypTen module from an ONNX Protobuf string or file.
        """
        raise NotImplementedError("Call this function on a Module type.")

    def forward(self, *args, **kwargs):
        """Perform forward pass on model."""
        raise NotImplementedError("forward not implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self, mode=True):
        """Sets the module in the specified training mode."""
        for param in self.parameters():
            param.requires_grad = mode
        self.training = mode

        # Recursively set train mode
        for module in self.children():
            module.train(mode=mode)
        return self

    def eval(self):
        """Sets the module in evaluation mode."""
        return self.train(False)

    def children(self):
        """Returns an iterator over immediate children modules.
        Yields:
            Module: a child module
        """
        for _, module in self.named_children():
            yield module

    def named_children(self):
        """Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.
        Yields:
            (string, Module): Tuple containing a name and child module
        Example::
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)
        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def register_module(self, name, module):
        """Registers child module in the module."""
        self._modules[name] = module

    def modules(self):
        """Returns an iterator over all modules in the network.
        Yields:
            Module: a module in the network
        Note:
            Duplicate modules are returned only once.
        """
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo=None, prefix=""):
        """Returns iterator over named modules (non-recursively)."""
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self.named_children():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def register_parameter(self, name, param, requires_grad=True):
        """
        Register parameter in the module. This function cannot register
        parameters in child modules.
        """
        if name in self._parameters or hasattr(self, name):
            raise ValueError("Parameter or field %s already exists." % name)
        param.requires_grad = requires_grad
        self._parameters[name] = param
        setattr(self, name, param)

    def set_parameter(self, name, param):
        """
        Sets value of parameter in the module. This function cannot set
        parameters in child modules.
        """
        if name not in self._parameters or not hasattr(self, name):
            raise ValueError("Parameter %s does not exist." % name)
        self._parameters[name] = param
        setattr(self, name, param)

    def set_parameter_from_shares(self, name, share, **kwargs):
        """
        Sets value of parameter in the module from shares. This functionality is
        only for MPC-encrypted models.

        Supported named arguments for `MPCTensor` parameters include the `precision`
        of the encoder (default = `None`), the rank of the `src` (default = 0),
        and the `ptype` of the shares (default = `crypten.mpc.arithmetic`).

        This function cannot set the parameters in child modules.
        """

        # functionality is only supported when parameters are MPCTensors:
        assert self.encrypted, "can only set parameters from shares in encrypted models"
        if name not in self._parameters or not hasattr(self, name):
            raise ValueError("Parameter %s does not exist." % name)
        cls = type(self._parameters[name])
        assert hasattr(
            self._parameters[name], "from_shares"
        ), "parameter type {} does not supporting setting from shares".format(cls)

        # load parameters from shares:
        self._parameters[name] = cls.from_shares(share, **kwargs)
        setattr(self, name, self._parameters[name])

    def parameters(self, recurse=True):
        """Returns an iterator over module parameters.
        This is typically passed to an optimizer.
        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
        Yields:
            CrypTensor or torch.Tensor: module parameter
        """
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, recurse=True, prefix=None):
        """
        Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.
        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
        Yields:
            (string, CrypTensor or torch.Tensor): Tuple containing the name and parameter
        """
        for name, param in self._parameters.items():
            param_name = name if prefix is None else prefix + "." + name
            yield param_name, param
        if recurse:
            for module_name, module in self.named_children():
                pre = module_name if prefix is None else prefix + "." + module_name
                yield from module.named_parameters(recurse=recurse, prefix=pre)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. The specified `prefix` will be
        used in names of parameters and buffers in this module. The `keep_vars`
        boolean determines if parameters and buffers are kept on the autograd tape.
        """
        for name, param in self.named_parameters(recurse=False):
            if param is not None:
                destination[prefix + name] = param if keep_vars else param.detach()
        for name, buffer in self.named_buffers(recurse=False):
            if buffer is not None:
                destination[prefix + name] = buffer if keep_vars else buffer.detach()

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """
        Returns a dictionary containing the state of the module. Both parameters
        and persistent buffers (e.g., running averages) are included. Keys are
        corresponding parameter and buffer names.
        """

        # save parameters and buffers of current module:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = {"version": Module._version}
        self._save_to_state_dict(destination, prefix, keep_vars)

        # recurse over modules:
        for name, module in self.named_children():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars=keep_vars)
        return destination

    def _load_from_state_dict_crypten(self, state_dict, prefix, strict):
        """
        Copies parameters and buffers from `state_dict` into only this module
        but not its children. This is called on every submodule in the
        `load_state_dict` function.
        """

        # get state dict for just the current module (without children)
        local_state = {key: val for key, val in self.named_parameters() if val is not None}

        # in strict mode, check for missing keys in the state_dict:
        if strict:
            for name in local_state.keys():
                key = prefix + name
                if key not in state_dict:
                    raise ValueError("Key {} not found in state dict.".format(key))

        # loop over parameters / buffers in module:
        for name, param in local_state.items():
            key = prefix + name
            input_param = state_dict[key]

            # size in state_dict should match size of parameters:
            if input_param.size() != param.size():
                raise ValueError(
                    "Size mismatch for {}: copying a param with"
                    "shape {} from checkpoint, the shape in"
                    "current model is {}.".format(key, input_param.size(), param.size())
                )
                continue

            # cannot copy encrypted tensors into unencrypted models and vice versa:
            param_encrypted = isinstance(input_param, crypten.CrypTensor)
            if param_encrypted:
                assert self.encrypted, "cannot copy encrypted parameters into unencrypted model"
            else:
                assert not self.encrypted, "cannot copy unencrypted parameters into encrypted model"

            # copy parameters from state_dict:
            with crypten.no_grad(), torch.no_grad():
                param.copy_(input_param)

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from `state_dict` into this module and its
        children. If `strict` is `True`, then the keys of `state_dict` must
        exactly match the keys returned by this module's `state_dict` function.
        """
        # check version of state_dict:
        if strict:
            metadata = getattr(state_dict, "_metadata", None)
            if metadata is None:
                raise ValueError("Specified state_dict does not have metadata.")
            version = metadata.get("version", -1)
            if version != Module._version:
                raise ValueError("Specified state_dict has incorrect version: {}".format(version))

        # make copy state_dict so that load() can modify it:
        state_dict = state_dict.copy()

        def load(module, prefix=""):
            """
            Closure that performs the loading of a module recursively.
            """
            module._load_from_state_dict_crypten(state_dict, prefix, strict)
            for name, child in module.named_children():
                if child is not None:
                    load(child, prefix + name + ".")

        # perform the actual loading:
        load(self)
        load = None  # break load->load reference cycle

    def zero_grad(self):
        """Sets gradients of all parameters to zero."""
        for param in self.parameters():
            param.grad = None

    def update_parameters(self, learning_rate, grad_threshold=100):
        """Performs gradient step on parameters.

        Parameters:
            grad_threshold - Because arithmetic operations can extremely rarely
                    return large incorrect results, we zero-out all elements
                    with magnitude larger than this given threshold. To turn
                    off thresholding, set to `None`.
        """
        assert self.training, "module not in training mode"
        with CrypTensor.no_grad(), torch.no_grad():
            for param in self.parameters():
                if param.grad is None:
                    continue
                if not self.encrypted and isinstance(param.grad, CrypTensor):
                    raise RuntimeError(
                        "Cannot update parameters of unencrypted "
                        "model using encrypted gradients. Encrypt "
                        "model before updating parameters."
                    )

                # Threshold gradients to prevent gradient explosion from wrap overflow
                if self.encrypted and grad_threshold is not None:
                    # Compute based on square value since abs is more expensive
                    square_threshold = grad_threshold * grad_threshold
                    grad = param.grad.mul(param.grad.square().lt(square_threshold))
                else:
                    grad = param.grad

                param.sub_(grad.mul_(learning_rate))

    def register_buffer(self, name, buffer):
        """
        Register buffer in the module. Buffers are encrypted like parameters but
        they are not updated by parameter updates.
        """
        if name in self._buffers or hasattr(self, name):
            raise ValueError("Buffer or field %s already exists." % name)
        buffer.requires_grad = False
        self._buffers[name] = buffer
        setattr(self, name, buffer)

    def set_buffer(self, name, buffer):
        """
        Sets value of buffer in the module. This function cannot set the
        parameters in child modules.
        """
        if name not in self._buffers or not hasattr(self, name):
            raise ValueError("Buffer %s does not exist." % name)
        self._buffers[name] = buffer
        setattr(self, name, buffer)

    def buffers(self, recurse=True):
        """Returns an iterator over module buffers.
        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.
        Yields:
            CrypTensor or torch.Tensor: module buffer
        """
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer

    def named_buffers(self, recurse=True, prefix=None):
        """Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.
        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.
        Yields:
            (string, CrypTensor or torch.Tensor): Tuple containing the name and buffer
        Example::
            >>> for name, buf in self.named_buffers():
            >>>    if name in ['running_var']:
            >>>        print(buf.size())
        """
        for name, buffer in self._buffers.items():
            buffer_name = name if prefix is None else prefix + "." + name
            yield buffer_name, buffer
        if recurse:
            for module_name, module in self.named_children():
                pre = module_name if prefix is None else prefix + "." + module_name
                yield from module.named_buffers(recurse=recurse, prefix=pre)

    def to(self, *args, **kwargs):
        """
        Moves and/or casts the parameters and buffers.

        This can be called as

        `to(device=None, dtype=None, non_blocking=False)`
        `to(dtype, non_blocking=False)`
        `to(tensor, non_blocking=False)`
        `to(memory_format=torch.channels_last)`

        Args:
            device (torch.device) – the desired device of the parameters
            and buffers in this module
            dtype (torch.dtype) – the desired floating point type of the
            floating point parameters and buffers in this module
            tensor (torch.Tensor) – Tensor whose dtype and device are the
            desired dtype and device for all parameters and buffers in this module
            memory_format (torch.memory_format) – the desired memory format
            for 4D parameters and buffers in this module (keyword only argument)

        """
        for name, param in self._parameters.items():
            self.set_parameter(name, param.to(*args, **kwargs))
        for name, buffer in self._buffers.items():
            self.set_buffer(name, buffer.to(*args, **kwargs))

        for module in self.children():
            module.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        """
        Moves all model parameters and buffers to the GPU.

        Args:
            device (int, optional) – if specified, all parameters will be copied
            to that device
        """
        for name, param in self._parameters.items():
            self.set_parameter(name, param.cuda(device=device))
        for name, buffer in self._buffers.items():
            self.set_buffer(name, buffer.cuda(device=device))

        for module in self.children():
            module.cuda(device=device)
        return self

    def cpu(self):
        """Moves all model parameters and buffers to the CPU."""
        for name, param in self._parameters.items():
            self.set_parameter(name, param.cpu())
        for name, buffer in self._buffers.items():
            self.set_buffer(name, buffer.cpu())

        for module in self.children():
            module.cpu()
        return self

    def _apply(self, fn):
        """Applies a function recursively on all modules."""
        for module in self.children():
            module._apply(fn)
        fn(self)
        return self

    def encrypt(self, mode=True, src=0):
        """Encrypts the model."""
        if mode != self.encrypted:
            # encrypt / decrypt parameters:
            self.encrypted = mode
            for name, param in self.named_parameters(recurse=False):
                requires_grad = param.requires_grad
                if mode:  # encrypt parameter
                    self.set_parameter(
                        name,
                        mpc.cryptensor(param, **{"src": src}, requires_grad=requires_grad),
                    )
                else:  # decrypt parameter
                    self.set_parameter(name, param.get_plain_text())
                    self._parameters[name].requires_grad = requires_grad

            # encrypt / decrypt buffers:
            for name, buffer in self.named_buffers(recurse=False):
                # encrypt buffer only if it's a torch tensor (not shapes)
                if mode and torch.is_tensor(buffer):
                    self.set_buffer(
                        name,
                        mpc.cryptensor(buffer, **{"src": src}, requires_grad=False),
                    )
                # decrypt buffer if it's a cryptensor
                elif isinstance(buffer, CrypTensor):
                    self.set_buffer(name, buffer.get_plain_text())
                    self._buffers[name].requires_grad = False

            # apply encryption recursively:
            return self._apply(lambda m: m.encrypt(mode=mode, src=src))
        return self

    def decrypt(self):
        """Decrypts model."""
        return self.encrypt(mode=False)

    def __getattribute__(self, name):
        if name != "forward":
            return object.__getattribute__(self, name)

        def forward_function(*args, **kwargs):
            """
            Silently encrypted Torch inputs tensors (deprecated).
            """
            if self.encrypted and not self.SUPPORTS_PLAINTEXT_INPUTS:
                if any(torch.is_tensor(arg) for arg in args):
                    warnings.warn(
                        "Earlier versions of CrypTen silently encrypted Torch tensors. "
                        "That behavior is now deprecated because it is dangerous. "
                        "Please make sure you feed your model CrypTensors when needed.",
                        DeprecationWarning,
                    )
            elif not self.encrypted:
                if any(isinstance(arg, CrypTensor) for arg in args):
                    raise RuntimeError(
                        "Cannot input CrypTensors into unencrypted model. "
                        "Encrypt the model before feeding it CrypTensors."
                    )
            return object.__getattribute__(self, name)(*tuple(args), **kwargs)

        return forward_function

    def __getattr__(self, name):
        """Redefine __getattr__ so that any parameters, modules or buffers
        inside the Module object can be accessed as attributes
        """
        if "_parameters" in self.__dict__:
            parameters = self.__dict__["_parameters"]
            if name in parameters:
                return parameters[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        if "_buffers" in self.__dict__:
            buffers = self.__dict__["_buffers"]
            if name in buffers:
                return buffers[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self, name, value):
        """Redefine __setattr__ so that any submodules created
        inside the Module object are registered with _modules
        OrderedDict.
        """

        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        modules = self.__dict__.get("_modules")
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError("cannot assign module before Module.__init__() call")
            remove_from(self.__dict__, self._parameters, self._buffers)
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError(
                    "cannot assign '{}' as child module '{}' "
                    "(torch.nn.Module or None expected)".format(torch.typename(value), name)
                )
            modules[name] = value
        else:
            for key in ["_parameters", "_modules", "_buffers"]:
                if key in self.__dict__ and name in self.__dict__[key]:
                    values = self.__dict__[key]
                    values[name] = value
                    return
            object.__setattr__(self, name, value)

    def add_module(self, name, module):
        """Adds and registers a submodule with a given name"""
        assert name not in self._modules.keys(), "Module %s already exists." % name
        self.register_module(name, module)


class Container(Module):
    """
    Container allows distinguishing between individual modules and containers.
    """

    pass


class Graph(Container):
    """
    Acyclic graph of modules.

    The module maintains a dict of named modules and a graph structure stored in
    a dict where each key is a module name, and the associated value is a list
    of module names that provide the input into the module. Each module may have
    an optional list of output names as well, that is stored as an attribute in
    the module by the `add_module` function.
    """

    def __init__(self, input_names, output_names, modules=None, graph=None):
        """
        Initializes a graph module with inputs named by `input_names`, that
        produces outputs named by `output_names`.

        Optionally, `modules` and the `graph` structure can be specified as well.
        Alternatively, the graph can be built up using the `add_module` function.
        """
        super().__init__()
        if not isinstance(input_names, (list, tuple)):
            input_names = [input_names]
        if not isinstance(output_names, (list, tuple)):
            output_names = [output_names]
        self.input_names = input_names
        self.output_names = output_names
        self._graph = {}
        if modules is not None:
            self._modules = modules
        if graph is not None:
            self._graph = graph

    def add_module(self, name, module, input_names=None, output_names=None):
        """
        Adds a `module` with the specified `name` to the graph. If the `module`
        expects inputs, their names should be specified via `input_names`.

        The module is expected to produce an output with `name`. However, if the
        module produces multiple outputs, these must be named in the
        `output_names` list.

        Both `input_names` and `output_names` are expected to be ordered.
        """
        assert name not in self._graph, "Module %s already exists." % name
        self.register_module(name, module)
        if input_names is not None:
            self._graph[name] = input_names
        if output_names is not None:
            module._output_names = output_names

    def forward(self, *args):
        assert len(args) == len(self.input_names), f"Expected {len(self.input_names)} inputs but received {len(args)}."

        # keep track of all values that have been computed:
        values = {self.input_names[idx]: args[idx] for idx in range(len(args))}
        computed = {key: False for key in self._graph.keys()}
        inputs_available = {key: [False for _ in range(len(value_list))] for key, value_list in self._graph.items()}

        def _mark_as_computed(name):
            """Marks a value as having been computed."""
            computed[name] = True
            for key, value_list in self._graph.items():
                if name in value_list:
                    inputs_available[key][value_list.index(name)] = True

        def _find_computable_node():
            """Find a node for which all inputs are available."""
            for key, inputs_available_list in inputs_available.items():
                if all(inputs_available_list) and not computed[key]:
                    return key
            return None

        def _clear_unused_values():
            """Clear values that are no longer needed (to save memory)."""
            remove_keys = []
            for remove_key in values.keys():
                can_be_removed = True

                # we cannot remove a value if it is still needed:
                for key, value_list in self._graph.items():
                    if not computed[key] and remove_key in value_list:
                        can_be_removed = False
                        break
                if can_be_removed:
                    remove_keys.append(remove_key)

            # remove all values we no longer need:
            for remove_key in remove_keys:
                del values[remove_key]
            # NOTE: We maintain inputs_available[remove_key] as True to
            # prevent re-computation of the node.

        # perform forward pass:
        for input_name in self.input_names:
            _mark_as_computed(input_name)
        node_to_compute = _find_computable_node()
        while node_to_compute is not None:
            # compute output of module:
            input = [values[name] for name in self._graph[node_to_compute]]
            if len(input) == 1:
                input = input[0]  # unpack iterable if possible
            module = self._modules[node_to_compute]
            output = module(input)

            # we may get one output:
            output_names = getattr(module, "_output_names", None)
            if output_names is None or len(output_names) == 1:
                if output_names is not None:
                    assert output_names[0] == node_to_compute, "invalid graph"
                values[node_to_compute] = output
                _mark_as_computed(node_to_compute)

            # or multiple outputs:
            else:
                assert isinstance(
                    output, tuple
                ), f"expected outputs {output_names} of {module} to be tuple, not {type(output)}"
                assert len(output_names) == len(
                    output
                ), f"expected {len(output_names)} outputs from {module}, received {len(output)}"
                for node, value in zip(output_names, output):
                    values[node] = value
                    _mark_as_computed(node)

            # return output if it is available:
            if all(computed[output_name] for output_name in self.output_names):
                result = [values[output_name] for output_name in self.output_names]
                return result[0] if len(result) == 1 else tuple(result)

            # find next node to compute:
            node_to_compute = _find_computable_node()

            # clean up values we no longer need:
            _clear_unused_values()

        # this should never happen:
        raise ValueError("nn.Graph.forward() failed. Is graph unconnected?")

    def to_pytorch(self):
        if not hasattr(self, "pytorch_model"):
            raise AttributeError("CrypTen Graph detached from PyTorch model.")
        if self.encrypted:
            raise ValueError("CrypTen model must be decrypted before calling to_pytorch()")
        with torch.no_grad():
            for name, param in self.pytorch_model.named_parameters():
                param.set_(self._modules[name].data)
        return self.pytorch_model


class Sequential(Graph):
    """
    Sequence of modules.
    """

    def __init__(self, *module_list, input_names=None):
        if input_names is None:
            input_names = ["input"]
        super().__init__(input_names, "output")
        if len(module_list) == 1 and isinstance(module_list[0], list):
            raise DeprecationWarning(
                "passing crypten.nn.Sequential a list is deprecated. Please "
                "pass unpacked arguments (e.g. Sequential(*my_modules))."
            )
            module_list = module_list[0]

        for idx, module in enumerate(module_list):
            if isinstance(module, OrderedDict):
                for key, val in module.items():
                    self.add_module(key, val, input_names)
                    input_names = key
                self.output_names = [key]
            else:
                module_name = str(idx)
                if idx > 0:
                    input_names = [str(idx - 1)]
                self.add_module(module_name, module, input_names)
                self.output_names = [module_name]


class ModuleList(Module):
    r"""Holds submodules in a list.

    :class:`~crypten.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~crypten.nn.Module` methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def __dir__(self):
        keys = super(ModuleList, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                del self._modules[str(k)]
        else:
            del self._modules[self._get_abs_string_index(idx)]
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))


__module_list_func_names = [
    "_get_abs_string_index",
    "__getitem__",
    "__setitem__",
    "__len__",
    "__iter__",
    "__iadd__",
    "insert",
    "append",
    "extend",
]


for func_name in __module_list_func_names:
    func = getattr(torch.nn.ModuleList, func_name)
    setattr(ModuleList, func_name, func)


class ModuleDict(Module):
    r"""Holds submodules in a dictionary.

    :class:`~crypten.nn.ModuleDict` can be indexed like a regular Python dictionary,
    but modules it contains are properly registered, and will be visible by all
    :class:`~crypten.nn.Module` methods.

    :class:`~crypten.nn.ModuleDict` is an **ordered** dictionary that respects

    * the order of insertion, and

    * in :meth:`~crypten.nn.ModuleDict.update`, the order of the merged ``OrderedDict``
      or another :class:`~crypten.nn.ModuleDict` (the argument to :meth:`~crypten.nn.ModuleDict.update`).

    Note that :meth:`~crypten.nn.ModuleDict.update` with other unordered mapping
    types (e.g., Python's plain ``dict``) does not preserve the order of the
    merged mapping.

    Arguments:
        modules (iterable, optional): a mapping (dictionary) of (string: module)
            or an iterable of key-value pairs of type (string, module)

    Example::

        class MyModule(nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.choices = nn.ModuleDict({
                        'conv': nn.Conv2d(10, 10, 3),
                        'pool': nn.MaxPool2d(3)
                })
                self.activations = nn.ModuleDict([
                        ['lrelu', nn.LeakyReLU()],
                        ['prelu', nn.PReLU()]
                ])

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x
    """

    def __init__(self, modules=None):
        super(ModuleDict, self).__init__()
        if modules is not None:
            self.update(modules)


__module_dict_func_names = [
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__len__",
    "__iter__",
    "__contains__",
    "clear",
    "pop",
    "keys",
    "items",
    "values",
    "update",
    "forward",
]


for func_name in __module_dict_func_names:
    func = getattr(torch.nn.ModuleDict, func_name)
    setattr(ModuleDict, func_name, func)


class Parameter(Module):
    """
    Module that holds a parameter tensor. If `trainable` is set to `False`, the
    parameter will be treated as a buffer: it will not be trained. Hence,
    parameters do not inherit `requires_grad` from the tensor they are initialized
    with.

    Parameters are encrypted when the `encrypt()` function is called on a module.
    """

    def __init__(self, param, trainable=True):
        super().__init__()

        # ensure that input is a PyTorch or CrypTen tensor:
        assert torch.is_tensor(param) or mpc.is_encrypted_tensor(
            param
        ), f"param must be PyTorch of CrypTen tensor, not {type(param)}"
        if isinstance(param, torch.nn.parameter.Parameter):
            param = param.data

        # register parameter or buffer:
        if trainable:
            self.register_parameter("data", param)
        else:
            self.register_buffer("data", param)

        # register whether or not module is encrypted:
        self.encrypted = mpc.is_encrypted_tensor(param)

    def forward(self, input):
        return self.data

    @property
    def requires_grad(self):
        return self.data.requires_grad


class Constant(Module):
    """
    Module that holds a constant tensor. If `trainable` is set to `False`, the
    parameter will be treated as a buffer: it will not be trained.

    Constants are not encrypted when the `encrypt()` function is called on a module.
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def __init__(self, value):
        super().__init__()
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        assert torch.is_tensor(value), f"value must be PyTorch tensor, not {type(value)}"
        self.value = value.to(dtype=torch.float)

    def forward(self, input):
        return self.value

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert "value" in attributes, "No value for Constant specified."
        return Constant(attributes["value"])

    def encrypt(self, mode=True, src=0):
        self.encrypted = mode
        return self


class ConstantOfShape(Module):
    """
    Modules that returns a matrix of specified size containing a constant.
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def __init__(self, value):
        super().__init__()
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        assert torch.is_tensor(value), f"value must be PyTorch tensor, not {type(value)}"
        self.value = value.to(dtype=torch.float)

    def forward(self, size):
        if torch.is_tensor(size):
            size = size.int().tolist()
        assert isinstance(size, (list, tuple)), f"size must be list or tuple, not {type(size)}"
        return self.value.expand(*size)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert "value" in attributes, "No value for ConstantOfShape specified."
        return ConstantOfShape(attributes["value"])

    def encrypt(self, mode=True, src=0):
        self.encrypted = mode
        return self


class Add(Module):
    """
    Module that sums two values.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        assert len(input) == 2, "input must contain two tensors"
        return input[0].add(input[1])

    @staticmethod
    def from_onnx(attributes=None):
        return Add()


class Sub(Module):
    """
    Module that subtracts two values.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        assert len(input) == 2, "input must contain two tensors"
        return input[0].sub(input[1])

    @staticmethod
    def from_onnx(attributes=None):
        return Sub()


class Mul(Module):
    """
    Module that multiplies two values.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        assert len(input) == 2, "input must contain two tensors"
        return input[0].mul(input[1])

    @staticmethod
    def from_onnx(attributes=None):
        return Mul()


class Div(Module):
    """
    Module that divides two values.
    """

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input must be list or tuple"
        assert len(input) == 2, "input must contain two tensors"
        return input[0].div(input[1])

    @staticmethod
    def from_onnx(attributes=None):
        return Div()


class Pow(Module):
    """
    Module that takes input to some power, where the power is an integer.
    """

    def forward(self, input):
        base, power = input
        if torch.is_tensor(power) and power.nelement() == 1:
            power = power.item()
            if int(power) == power:  # try to convert power to integer if possible
                power = int(power)
        return base.pow(power)

    @staticmethod
    def from_onnx(attributes=None):
        return Pow()


class Sqrt(Module):
    """
    Module that takes square-root of the input.
    """

    def forward(self, input):
        return input.sqrt()

    @staticmethod
    def from_onnx(attributes=None):
        return Sqrt()


class Exp(Module):
    """
    Module that calculates the exponential of the given input tensor, element-wise.
    """

    def forward(self, input):
        return input.exp()

    @staticmethod
    def from_onnx(attributes=None):
        return Exp()


class Erf(Module):
    """
    Module that calculates the error function of the given input tensor, element-wise.
    """

    def forward(self, input):
        return input.erf()

    @staticmethod
    def from_onnx(attributes=None):
        return Erf()


class _Reduce(Module):
    """
    Base class for the functionality of ONNX ReduceMean (defined here as Mean),
    and ONNX ReduceSum (defined here as Sum).
    """

    def __init__(self, dim, keepdim=False, reduction_fn="mean"):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.reduction_fn = reduction_fn

    def forward(self, input):
        return getattr(input, self.reduction_fn)(self.dim, keepdim=self.keepdim)


class Mean(_Reduce):
    """
    Module that computes the mean of the input tensor's element along the provided axes.
    If `keepdim` is True, the output tensor is of the same size as input
    except in the dimension(s) `dim` where it is of size 1.
    Otherwise, `dim` is squeezed, resulting in the output tensor having 1
    (or `len(dim)`) fewer dimension(s).
    """

    def __init__(self, dim, keepdim=False):
        super().__init__(dim, keepdim, "mean")

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        keepdim = _identify_bool_attributes_with_defaults(attributes, "keepdims", 1)
        return Mean(attributes["axes"], keepdim)


class Sum(_Reduce):
    """
    Module that computes the sum of the input tensor's element along the provided axes.
    If `keepdim` is True, the output tensor is of the same size as input
    except in the dimension(s) `dim` where it is of size 1.
    Otherwise, `dim` is squeezed, resulting in the output tensor having 1
    (or `len(dim)`) fewer dimension(s).
    """

    def __init__(self, dim, keepdim=False):
        super().__init__(dim, keepdim, "sum")

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        keepdim = _identify_bool_attributes_with_defaults(attributes, "keepdims", 1)
        return Sum(attributes["axes"], keepdim)


class Transpose(Module):
    """
    Module that transposes the input tensor similar to
    `numpy.transpose`. For example, when perm=(1, 0, 2), given an input
    tensor of shape (1, 2, 3), the output shape will be (2, 1, 3). Note
    that the signature of this module matches the ONNX specification
    and differs from `torch.transpose`

    Args:
        `perm`: list of ints
    """

    def __init__(self, perm):
        super().__init__()
        self.perm = perm

    def forward(self, input):
        # New Linear jit tracer causes Transpose module to have a weight
        if hasattr(self, "weight"):
            input = self.weight

        assert input.dim() == len(self.perm)
        return input.permute(self.perm)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        # TODO: ONNX specification says the permutation should be
        # reversed if not provided in the attributes.  Because we
        # don't have the input size here, we need figure out a
        # different way of supporting this, if we want to do that.
        return Transpose(attributes["perm"])


class Squeeze(Module):
    r"""
    Returns a tensor with all the dimensions of :attr:`input` of size `1` removed.

    For example, if `input` is of shape:
    :math:`(A \times 1 \times B \times C \times 1 \times D)` then the `out` tensor
    will be of shape: :math:`(A \times B \times C \times D)`.

    When :attr:`dimension` is given, a squeeze operation is done only in the given
    dimension. If `input` is of shape: :math:`(A \times 1 \times B)`,
    ``squeeze(input, 0)`` leaves the tensor unchanged, but ``squeeze(input, 1)``
    will squeeze the tensor to the shape :math:`(A \times B)`.

    .. note:: The returned tensor shares the storage with the input tensor,
            so changing the contents of one will change the contents of the other.

    Args:
        dimension (int, optional): if given, the input will be squeezed only in
            this dimension
    """

    def __init__(self, dimension):
        super().__init__()
        if isinstance(dimension, (list, tuple)):
            assert len(dimension) == 1, "can only squeeze one dimension at a time"
            dimension = dimension[0]
        self.dimension = dimension

    def forward(self, input):
        return input.squeeze(self.dimension)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        dimension = attributes["axes"]
        assert len(dimension) == 1, "can only squeeze one dimension at a time"
        return Squeeze(dimension[0])


class Unsqueeze(Module):
    """
    Module that unsqueezes a tensor.
    Returns a new tensor with a dimension of size one inserted at the
    specified position.

    The returned tensor shares the same underlying data with this tensor.
    A :attr:`dimension` value within the range ``[-input.dim() - 1, input.dim() + 1)``
    can be used. Negative :attr:`dimension` will correspond to :meth:`unsqueeze`
    applied at :attr:`dimension` = ``dim + input.dim() + 1``.

    Args:
        dimension (int): the index at which to insert the singleton dimension
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def __init__(self, dimension):
        super().__init__()
        if isinstance(dimension, (list, tuple)):
            assert len(dimension) == 1, "can only squeeze one dimension at a time"
            dimension = dimension[0]
        self.dimension = dimension

    def forward(self, input):
        if isinstance(input, list):
            assert len(input) == 2, "list input must be [x, dimension]"
            input, dimension = input
            assert len(dimension) == 1, "can only unsqueeze one dimension at a time"
            dimension = int(dimension.item())
        else:
            dimension = self.dimension
        return input.unsqueeze(dimension)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        dimension = attributes.get("axes", [None])
        assert len(dimension) == 1, "can only unsqueeze one dimension at a time"
        return Unsqueeze(dimension[0])


class Slice(Module):
    """
    Module that slices the input along the specified `axes` (list of `int`s) from
    the indices in `start`s to the indices in `end`s.

    This module definition matches ONNX opset version 11.
    """

    def __init__(self, starts, ends, axes=None):
        super().__init__()
        self.starts = starts
        self.ends = ends
        self.axes = axes

    def forward(self, x):
        # Process inputs:
        axes = None
        if isinstance(x, list):
            if len(x) == 3:
                x, starts, ends = x
                axes, steps = self.axes, 1
            elif len(x) == 4:
                x, starts, ends, axes = x
                steps = 1
            elif len(x) == 5:
                x, starts, ends, axes, steps = x
                if not torch.eq(steps.int(), 1).all():
                    raise ValueError("Only steps value of 1 currently supported.")
            else:
                raise ValueError("list input x must have 3, 4, or 5, values")
            starts, ends = starts.int().tolist(), ends.int().tolist()
        else:
            starts, ends, axes = self.starts, self.ends, self.axes
            steps = 1
        if axes is None:
            axes = list(range(len(starts)))

        # Perform slicing:
        output = x
        for idx, axis in enumerate(axes):
            start, end = int(starts[idx]), int(ends[idx])
            length = min(end, output.size(int(axis))) - start
            output = output.narrow(int(axis), start, length)
        return output

    @staticmethod
    def from_onnx(attributes=None):
        return Slice(
            attributes.get("starts", None),
            attributes.get("ends", None),
            axes=attributes.get("axes", None),
        )


class Expand(Module):
    """
    Module that expands a tensor to the specified size.
    """

    def forward(self, x):
        # unpack inputs:
        input, shape = tuple(x)
        if torch.is_tensor(shape):
            shape = shape.long().tolist()

        # broadcasting in ONNX is different from PyTorch when shape is 1:
        for idx in range(len(shape)):
            if shape[idx] == 1 and input.size(idx) > 1:
                shape[idx] = input.size(idx)

        # perform the expansion:
        return input.expand(shape)

    @staticmethod
    def from_onnx(attributes=None):
        return Expand()


class Cast(Module):
    """
    Module that casts the input tensor to the specified type.
    """

    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        if torch.is_tensor(x):
            return x.to(dtype=self.dtype)
        return x  # this is a no-op as MPCTensors do not know their dtype

    @staticmethod
    def from_onnx(attributes=None):
        dtype = sym_help._get_const(attributes["to"], "i", "dtype")
        return Cast(dtype=sym_help.scalar_type_to_pytorch_type[dtype])


class Range(Module):
    """
    Module that returns a tensor with the specified range.
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def forward(self, x):
        if len(x) == 2:
            start, end = tuple(x)
            step = 1
        elif len(x) == 3:
            start, end, step = tuple(x)
        else:
            raise ValueError(f"Expected 2 or 3 inputs, but received {len(x)}.")
        return torch.arange(start, end, step)

    @staticmethod
    def from_onnx(attributes=None):
        return Range()


class Equal(Module):
    """
    Module that compares two tensors to determine which elements are equal.
    """

    def forward(self, x):
        x1, x2 = tuple(x)
        if x1.size() != x2.size():
            return False
        return x1.eq(x2)

    @staticmethod
    def from_onnx(attributes=None):
        return Equal()


class Where(Module):
    """
    Module that returns elements from one tensor or the other depending on the
    value of the specified condition.
    """

    def forward(self, x):
        condition, x1, x2 = tuple(x)
        return mpc.where(condition, x1, x2)

    @staticmethod
    def from_onnx(attributes=None):
        return Where()


class Flatten(Module):
    """
    Module that flattens the input tensor into a 2D matrix.

    Args:
        axis (int, optional): must not be larger than dimension
    """

    def __init__(self, axis=1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        if self.axis == 0:
            return x.view(1, -1)
        else:
            assert self.axis <= x.dim(), "axis must not be larger than dimension"
            prod = 1
            for i in range(self.axis):
                prod *= x.size(i)
            return x.view(prod, -1)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        # axis : int (default is 1)
        axis = 1
        if "axis" in attributes:
            axis = int(attributes["axis"])
            assert axis >= 0, "axis must not be negative"
        return Flatten(axis)


class Shape(Module):
    """
    Module that returns the shape of a tensor. If the input tensor is encrypted,
    the output size vector will be encrypted, too.
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x, dim=None):
        dim = dim if dim is not None else self.dim
        if dim is None:
            size = torch.tensor(x.size())
        else:
            size = torch.tensor(x.size(dim))
        return size

    @staticmethod
    def from_onnx(attributes=None):
        return Shape()


class Concat(Module):
    """
    Module that concatenates tensors along a dimension.

    Args:
        dim (int, optional): the dimension over which to concatenate
    """

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, input):
        assert isinstance(input, (list, tuple)), "input needs to be a list or tuple"
        assert len(input) >= 1, "need at least one tensor to concatenate"
        return mpc.cat(input, self.dimension)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        dimension = attributes["axis"]
        return Concat(dimension)


class Reshape(Module):
    """
    Module that reshapes tensors to new dimensions.
    Returns a tensor with same data and number of elements as :attr:`self`,
    but with the specified shape.

    When possible, the returned tensor will be a view
    of :attr:`self`. Otherwise, it will be a copy. Contiguous inputs and inputs
    with compatible strides can be reshaped without copying, but you should not
    depend on the copying vs. viewing behavior.

    See :meth:`torch.Tensor.view` on when it is possible to return a view.
    A single dimension may be -1, in which case it's inferred from the remaining
    dimensions and the number of elements in :attr:`self`.

    Args:
        input (tuple): contains input tensor and shape (torch.Size)
    """

    def __init__(self, shape=None):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, tensor, shape=None):
        if isinstance(tensor, list) and len(tensor) == 2:
            tensor, shape = tensor
        shape = shape if shape is not None else self.shape
        assert shape is not None, "Reshape requires a shape in forward if not supplied in initialization"
        if torch.is_tensor(shape):
            shape = torch.Size(shape.long())
        return tensor.reshape(shape)

    @staticmethod
    def from_onnx(attributes=None):
        if "shape" in attributes:
            return Reshape(shape=attributes["shape"])
        return Reshape()


class Dropout(Module):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Furthermore, the outputs are scaled by a factor of
    :math:`\frac{1}{1-p}` during training. This means that during evaluation
    the module simply computes an identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if inplace:
            logging.warning("CrypTen Dropout module does not support inplace computation.")
        self.p = p

    def forward(self, input):
        return input.dropout(p=self.p, training=self.training)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        return Dropout(attributes["ratio"])


class DropoutNd(Module):
    """Randomly zero out entire channels (a channel is a nD feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a nD tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Args:
        p (float, optional): probability of an element to be zero-ed.
    """

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        if inplace:
            logging.warning("CrypTen DropoutNd module does not support inplace computation.")
        self.p = p

    def forward(self, input):
        return input._feature_dropout(p=self.p, training=self.training)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        return DropoutNd(attributes["ratio"])


class Dropout2d(DropoutNd):
    r"""
    Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv2d` modules.

    Args:
        p (float, optional): probability of an element to be zero-ed.
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)
    """

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        return Dropout2d(attributes["ratio"])


class Dropout3d(DropoutNd):
    r"""Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`\text{input}[i, j]`).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    Usually the input comes from :class:`nn.Conv3d` modules.

    Args:
        p (float, optional): probability of an element to be zeroed.

    Shape:
        - Input: :math:`(N, C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` (same shape as input)
    """

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        return Dropout3d(attributes["ratio"])


class Gather(Module):
    r"""
    Module that gathers elements from tensor according to indices. Given data tensor
    of rank :math:`r >= 1`, and indices tensor of rank :math:`q`, gather entries of
    the axis dimension of data (by default outer-most one as `axis = 0`)
    indexed by indices, and concatenates them in an output tensor of rank
    :math:`q + (r - 1)`. For example, for `axis = 0`: Let :math:`k =
    indices[i_{0}, ..., i_{q-1}]`. Then :math:`output[i_{0}, ..., i_{q-1}, j_{0},
    ..., j_{r-2}] = input[k, j_{0}, ..., j_{r-2}]`. This is an operation from the
    ONNX specification.

    Args:
        dimension (int): the axis along which to index
        index(tensor): the indices to select along the `dimension`
    """

    SUPPORTS_PLAINTEXT_INPUTS = True

    def __init__(self, dimension, indices=None):
        super().__init__()
        self.dimension = dimension
        self.indices = indices

    def forward(self, input):
        if not isinstance(input, (list, tuple)):
            tensor = input
            indices = self.indices
        elif len(input) == 1:
            tensor = input[0]
            indices = self.indices
        else:
            tensor, indices = input

        # indices need to be a PyTorch tensor:
        if mpc.is_encrypted_tensor(indices):
            raise ValueError("Cannot perform Gather operation using encrypted indices.")
        elif isinstance(indices, (int, list, tuple)):
            indices = torch.tensor(indices)
        indices = indices.long()

        # CrypTensor input
        if mpc.is_encrypted_tensor(tensor):
            result = tensor.take(indices, self.dimension)

        # Torch tensor input
        elif self.dimension is None or tensor.dim() == 0:
            result = torch.take(tensor, indices)
        else:
            all_indices = [slice(0, x) for x in tensor.size()]
            all_indices[self.dimension] = indices
            result = tensor[all_indices]
        return result

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        if "axis" not in attributes:
            attributes["axis"] = None
        if "shape" not in attributes:
            attributes["shape"] = None
        return Gather(attributes["axis"], indices=attributes["shape"])


class _ConstantPad(Module):
    """
    Module that pads a tensor.
    """

    def __init__(self, padding, value, ndims, mode="constant"):
        super().__init__()
        if isinstance(padding, (int)):
            padding = [padding, padding] * ndims
        self.padding = padding
        self.value = value
        self.mode = mode

    def forward(self, input):
        if isinstance(input, list):
            assert len(input) == 2, "input should be [tensor, pads] list"
            padding = tuple(input[1].int().tolist())
            input = input[0]
        else:
            padding = self.padding
        return input.pad(padding, value=self.value, mode=self.mode)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert attributes["mode"] == b"constant", "only constant padding supported"
        return _ConstantPad(None, 0, 0, mode="constant")


class ConstantPad1d(_ConstantPad):
    """
    Module that pads a 1D tensor.
    """

    def __init__(self, padding, value, mode="constant"):
        super(ConstantPad1d, self).__init__(padding, value, 1, mode=mode)


class ConstantPad2d(_ConstantPad):
    """
    Module that pads a 2D tensor.
    """

    def __init__(self, padding, value, mode="constant"):
        super(ConstantPad2d, self).__init__(padding, value, 2, mode=mode)


class ConstantPad3d(_ConstantPad):
    """
    Module that pads a 3D tensor.
    """

    def __init__(self, padding, value, mode="constant"):
        super(ConstantPad3d, self).__init__(padding, value, 3, mode=mode)


class Gemm(Module):
    """
    Module that performs a general matrix multiplication.

    Unlike the `Linear` module, this module is stateless.
    """

    def __init__(self, alpha=1.0, beta=1.0, trans_a=False, trans_b=False):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.trans_a = trans_a
        self.trans_b = trans_b

    def forward(self, x):
        a, b, c = tuple(x)
        if self.trans_a:
            a = a.t()
        if self.trans_b:
            b = b.t()
        output = a.matmul(b).mul(self.alpha)
        output = output.add(c.mul(self.beta))
        return output

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert "alpha" in attributes, "attribute alpha missing"
        assert "beta" in attributes, "attribute beta missing"
        return Gemm(
            alpha=attributes["alpha"],
            beta=attributes["beta"],
            trans_a=attributes.get("transA", False),
            trans_b=attributes.get("transB", False),
        )


class Linear(Module):
    """
    Module that performs linear transformation.
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    """  # noqa: W605

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        # initialize model parameters:
        pytorch_module = torch.nn.Linear(in_features, out_features, bias=bias)
        self.register_parameter("weight", pytorch_module.weight)
        if bias:
            self.register_parameter("bias", pytorch_module.bias)

    def forward(self, x):
        output = x.matmul(self.weight.t())
        if hasattr(self, "bias"):
            output = output.add(self.bias)
        return output


class MatMul(Module):
    """
    Matrix product of two tensors.
    The behavior depends on the dimensionality of the tensors as followsf
        - If both tensors are 1-dimensional, the dot product (scalar) is returned.
        - If both arguments are 2-dimensional, the matrix-matrix product is returned.
        - If the first argument is 1-dimensional and the second argument is
        2-dimensional, a 1 is prepended to its dimension for the purpose of the
        matrix multiply. After the matrix multiply, the prepended dimension is removed.
        - If the first argument is 2-dimensional and the second argument is
        1-dimensional, the matrix-vector product is returned.
        - If both arguments are at least 1-dimensional and at least one argument is
        N-dimensional (where N > 2), then a batched matrix multiply is returned.
        If the first argument is 1-dimensional, a 1 is prepended to its dimension
        for the purpose of the batched matrix multiply and removed after. If the
        second argument is 1-dimensional, a 1 is appended to its dimension for the
        purpose of the batched matrix multiple and removed after.
        The non-matrix (i.e. batch) dimensions are broadcasted (and thus
        must be broadcastable).  For example, if :attr:`input` is a
        :math:`(j \times 1 \times n \times m)` tensor and :attr:`other` is
        a :math:`(k \times m \times p)` tensor, :attr:`out` will be an
        :math:`(j \times k \times n \times p)` tensor.

    Arguments:
        Option 1: [input1, input2]
            input1: first input matrix to be multiplied
            input2: second input matrix to be multiplied.
        Option 2: input1
            input1: first input matrix to be multiplied, if module
            is already initialized with the second (i.e. multiplier) matrix.
    """

    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            self.register_parameter("weight", weight)

    def forward(self, x):
        if hasattr(self, "weight"):
            output = x.matmul(self.weight)
        else:
            assert isinstance(x, (list, tuple)), "input must be list or tuple"
            assert len(x) == 2, "input must contain two tensors"
            output = x[0].matmul(x[1])
        return output

    @staticmethod
    def from_onnx(attributes=None):
        return MatMul()


class Conv(Module):
    """
    Module that performs convolution, following the ONNX specification of that
    operation. Unlike other Conv modules, this module is stateless.
    """

    def __init__(self, stride, padding, dilation, groups=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        # unpack inputs:
        if len(x) == 2:
            x, weight = x
            bias = None
        elif len(x) == 3:
            x, weight, bias = x
        else:
            raise ValueError(f"Conv module must have 2 or 3 inputs, not {len(x)}")

        # prepare inputs into convolution function:
        dim = weight.dim() - 2
        if dim < 1 or dim > 2:
            raise ValueError(f"Convolution on {dim}-dimensional input is not supported.")
        args = [weight]
        kwargs = {
            "stride": self.stride,
            "padding": self.padding,
            "dilation": self.dilation,
            "groups": self.groups,
        }

        # identify correct convolution function to use:
        if torch.is_tensor(x):
            func = getattr(torch.nn.functional, f"conv{dim}d", None)
            args = [x] + args + bias  # torch function takes different inputs
        else:
            func = getattr(x, f"conv{dim}d", None)

        # perform the convolution:
        x = func(*args, **kwargs)

        # add the bias term if it is specified, and wasn;t already added:
        if not torch.is_tensor(x) and bias is not None:
            bias = bias.unsqueeze(0)
            while bias.dim() < x.dim():
                bias = bias.unsqueeze(-1)
            x = x.add(bias)
        return x

    @staticmethod
    def from_onnx(attributes=None):
        # check attribute inputs:
        if attributes is None:
            attributes = {}
        for attr in ["strides", "pads", "dilations"]:
            assert attr in attributes, f"missing attribute {attr}"

        # CrypTen and Torch use a single padding number per dimension:
        padding = attributes["pads"]
        padding = [padding[idx] for idx in range(0, len(padding), 2)]

        # return module:
        return Conv(
            attributes["strides"],
            padding,
            attributes["dilations"],
            groups=attributes.get("group", 1),
        )


# TODO: Eliminate copy-pasta by implementing _Conv parent class
class Conv1d(Module):
    r"""
    Module that performs 1D convolution.

    Applies a 1D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)


    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    and :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`,
    :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the
        height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
        the height dimension, and the second `int` for the width dimension

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
        the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular`
        Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements.
        Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
        Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where

          .. math::
              L_{out} = \left\lfloor\frac{L_{in}  +
              2 \times \text{padding} - \text{dilation} \times
              (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """  # noqa: W605

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        # check inputs:
        super().__init__()
        assert isinstance(stride, int), "stride must be an integer"
        assert isinstance(padding, int), "padding must be an integer"

        # initialize model parameters:
        pytorch_module = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.register_parameter("weight", pytorch_module.weight)
        if bias:
            self.register_parameter("bias", pytorch_module.bias)

        # set other instance fields:
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        x = x.conv1d(
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if hasattr(self, "bias"):
            x = x.add(self.bias.unsqueeze(-1))
        return x


class Conv2d(Module):
    r"""
    Module that performs 2D convolution.

    Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}},
    H_{\text{out}}, W_{\text{out}})` can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of implicit zero-paddings on both
      sides for :attr:`padding` number of points for each dimension.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels,
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters, of size:
          :math:`\left\lfloor\frac{out\_channels}{in\_channels}\right\rfloor`.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`,
    :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the
        height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
        the height dimension, and the second `int` for the width dimension

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
        the input. Default: 0
        padding_mode (string, optional). Accepted values `zeros` and `circular`
        Default: `zeros`
        dilation (int or tuple, optional): Spacing between kernel elements.
        Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
        Default: ``True``

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  +
              2 \times \text{padding}[0] - \text{dilation}[0] \times
              (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  +
              2 \times \text{padding}[1] - \text{dilation}[1] \times
              (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        # check inputs:
        super().__init__()

        # initialize model parameters:
        pytorch_module = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.register_parameter("weight", pytorch_module.weight)
        if bias:
            self.register_parameter("bias", pytorch_module.bias)
        else:
            self.bias = pytorch_module.bias

        # set other instance fields:
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x):
        x = x.conv2d(
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if hasattr(self, "bias") and self.bias is not None:
            x = x.add(self.bias.unsqueeze(-1).unsqueeze(-1))
        return x


class ReLU(Module):
    r"""
    Module that computes rectified linear unit (ReLU) activations element-wise.

    :math:`\text{ReLU}(x)= \max(0, x)`
    """

    def __init__(self, inplace=False):
        super().__init__()
        if inplace:
            logging.warning("CrypTen ReLU module does not support inplace computation.")

    def forward(self, x):
        return x.relu()

    @staticmethod
    def from_onnx(attributes=None):
        return ReLU()


class Hardtanh(Module):
    r"""Applies the Hardtanh function element-wise

    HardTtnh is defined as:

    .. math::
        \text{Hardtanh}(x) = \begin{cases}
            1 & \text{ if } x > 1 \\
            -1 & \text{ if } x < -1 \\
            x & \text{ otherwise } \\
        \end{cases}

    The range of the linear region :math:`[-1, 1]` can be adjusted using
    :attr:`min_val` and :attr:`max_val`.

    Args:
        min_val: minimum value of the linear region range. Default: -1
        max_val: maximum value of the linear region range. Default: 1
        inplace: can optionally do the operation in-place. Default: ``False``

    Keyword arguments :attr:`min_value` and :attr:`max_value`
    have been deprecated in favor of :attr:`min_val` and :attr:`max_val`.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/Hardtanh.png

    Examples::

        >>> m = nn.Hardtanh(-2, 2)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, min_val=-1.0, max_val=1.0, inplace=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        if inplace:
            logging.warning("CrypTen Hardtanh module does not support inplace computation.")

    def forward(self, input):
        if isinstance(input, list):
            input, min_val, max_val = input
            min_val, max_val = min_val.item(), max_val.item()
        else:
            min_val, max_val = self.min_val, self.max_val
        return input.hardtanh(min_val, max_val)

    @staticmethod
    def from_onnx(attributes=None):
        return Hardtanh(
            min_val=attributes.get("min", -1.0),
            max_val=attributes.get("max", 1.0),
        )


class ReLU6(Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLU6()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace=False):
        if inplace:
            logging.warning("CrypTen ReLU6 module does not support inplace computation.")
        super(ReLU6, self).__init__(min_val=0, max_val=6, inplace=False)


class Sigmoid(Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    """

    def forward(self, x):
        return x.sigmoid()

    @staticmethod
    def from_onnx(attributes=None):
        return Sigmoid()


class Softmax(Module):
    r"""Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:

    .. math::
        \text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Arguments:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.softmax(self.dim)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert "axis" in attributes, "axis attribute missing"
        return Softmax(attributes["axis"])


class LogSoftmax(Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
    input Tensor. The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) =
        \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Arguments:
        dim (int): A dimension along which LogSoftmax will be computed.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range \[-inf, 0\)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input.log_softmax(self.dim)

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        assert "axis" in attributes, "axis attribute missing"
        return LogSoftmax(attributes["axis"])


class _Pool2d(Module):
    """
    Module that performs 2D pooling.
    Applies a 2D max or average pooling over an input signal composed of several input
    planes.

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on
    both sides for :attr:`padding` number of points. :attr:`dilation` controls
    the spacing between the kernel points. It is harder to describe, but this
    `link`_ has a nice visualization of what :attr:`dilation` does.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`,
    :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the
        height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
        the height dimension, and the second `int` for the width dimension

    Args:
        pool_type (str): specifies "average" or "max" pooling
        kernel_size: the size of the window to take a max over
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(self, pool_type, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__()
        self.pool_type = pool_type
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, x):
        args = [self.kernel_size]
        kwargs = {
            "stride": self.stride,
            "padding": self.padding,
            "ceil_mode": self.ceil_mode,
        }
        if self.pool_type == "average":
            return x.avg_pool2d(*args, **kwargs)
        elif self.pool_type == "max":
            return x.max_pool2d(*args, **kwargs)
        else:
            raise ValueError("Unknown pooling type: %s" % self.pool_type)

    @staticmethod
    def from_onnx(pool_type, attributes=None):
        # check attributes:
        if attributes is None:
            attributes = {}
        if "pads" not in attributes:
            attributes["pads"] = [0]
        assert _all_the_same(["kernel_shape"]), "only square kernels are supported"
        assert _all_the_same(attributes["strides"]), "stride must be the same in each dimension"
        attributes["ceil_mode"] = attributes.get("ceil_mode", 0)
        attributes["ceil_mode"] = attributes["ceil_mode"] > 0

        # initialize module
        args = [attributes["kernel_shape"][0]]
        kwargs = {
            "stride": attributes["strides"][0],
            "padding": attributes["pads"][0],
            "ceil_mode": attributes["ceil_mode"],
        }
        if pool_type == "average":
            return AvgPool2d(*args, **kwargs)
        elif pool_type == "max":
            return MaxPool2d(*args, **kwargs)
        else:
            raise ValueError("Unknown pooling type: %s" % pool_type)


class AvgPool2d(_Pool2d):
    r"""
    Module that Applies a 2D average pooling
    over an input signal composed of several input planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C, H, W)`, output :math:`(N, C, H_{out}, W_{out})` and
    :attr:`kernel_size` :math:`(kH, kW)` can be precisely described as:

    .. math::

        out(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1}
        \sum_{n=0}^{kW-1} input(N_i, C_j, stride[0] \times h + m, stride[1]
        \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on
    both sides for :attr:`padding` number of points.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the
        height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for
        the height dimension, and the second `int` for the width dimension

    Args:
        kernel_size: the size of the window
        stride: the stride of the window. Default value is :attr:`kernel_size`
        padding: implicit zero padding to be added on both sides

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})`
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] -
                \text{kernel\_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
                \text{kernel\_size}[1]}{\text{stride}[1]} + 1\right\rfloor
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__("average", kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    @staticmethod
    def from_onnx(attributes=None):
        return super(AvgPool2d, AvgPool2d).from_onnx("average", attributes=attributes)


class MaxPool2d(_Pool2d):
    """
    Module that performs 2D max pooling (see :meth:`AvgPool2d`)
    """

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False):
        super().__init__("max", kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode)

    @staticmethod
    def from_onnx(attributes=None):
        return super(MaxPool2d, MaxPool2d).from_onnx("max", attributes=attributes)


class AdaptiveAvgPool2d(Module):
    r"""Applies a 2D adaptive average pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveAvgPool2d((5,7))
        >>> input = crypten.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveAvgPool2d(7)
        >>> input = crypten.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveAvgPool2d((None, 7))
        >>> input = crypten.randn(1, 64, 10, 9)
        >>> output = m(input)
    """

    def __init__(self, output_size=None):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def extra_repr(self) -> str:
        return "output_size={}".format(self.output_size)

    def forward(self, input_tensor, output_size=None):
        if output_size is None:
            output_size = self.output_size
        assert (
            output_size is not None
        ), "AdaptiveAvgPool2d requires an output_size in forward if not supplied in initialization"
        resized_input, args, kwargs = _adaptive_pool2d_helper(input_tensor, output_size, reduction="mean")
        return resized_input.avg_pool2d(*args, **kwargs)

    @staticmethod
    def from_onnx(attributes=None):
        if "shape" in attributes:
            return AdaptiveAvgPool2d(output_size=attributes["shape"])
        return AdaptiveAvgPool2d()


class AdaptiveMaxPool2d(Module):
    r"""Applies a 2D adaptive max pooling over an input signal composed of several input planes.

    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.

    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H.
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.

    Examples:
        >>> # target output size of 5x7
        >>> m = nn.AdaptiveMaxPool2d((5,7))
        >>> input = crypten.randn(1, 64, 8, 9)
        >>> output = m(input)
        >>> # target output size of 7x7 (square)
        >>> m = nn.AdaptiveMaxPool2d(7)
        >>> input = crypten.randn(1, 64, 10, 9)
        >>> output = m(input)
        >>> # target output size of 10x7
        >>> m = nn.AdaptiveMaxPool2d((None, 7))
        >>> input = crypten.randn(1, 64, 10, 9)
        >>> output = m(input)

    """

    def __init__(self, output_size=None):
        super(AdaptiveMaxPool2d, self).__init__()
        self.output_size = output_size

    def extra_repr(self) -> str:
        return "output_size={}".format(self.output_size)

    def forward(self, input_tensor, output_size=None):
        if output_size is None:
            output_size = self.output_size
        assert (
            output_size is not None
        ), "AdaptiveMaxPool2d requires an output_size in forward if not supplied in initialization"
        resized_input, args, kwargs = _adaptive_pool2d_helper(input_tensor, output_size, reduction="max")
        return resized_input.max_pool2d(*args, **kwargs)

    @staticmethod
    def from_onnx(attributes=None):
        if "shape" in attributes:
            return AdaptiveMaxPool2d(output_size=attributes["shape"])
        return AdaptiveMaxPool2d()


class GlobalAveragePool(Module):
    """
    GlobalAveragePool consumes an input tensor and applies average pooling
    across the values in the same channel. This is equivalent to AveragePool
    with kernel size equal to the spatial dimension of input tensor. This is an
    operation from the ONNX specification.
    """

    def forward(self, input):
        assert input.dim() > 2, "input needs to have more than two dimensions"

        # sum over all but batch dimension:
        result = input.shallow_copy()
        for dim in range(2, input.dim()):
            result = result.sum(dim, keepdim=True)

        # return average value:
        first_two_dims = input.size(0) * input.size(1)
        return result.div(input.nelement() / float(first_two_dims))

    @staticmethod
    def from_onnx(attributes=None):
        return GlobalAveragePool()


class BatchNormalization(Module):
    """
    Module that performs batch normalization following the ONNX specification.

    Unlike the `BatchNorm1d`, `BatchNorm2d`, and `BatchNorm3d` classes, this
    module is stateless. It takes `input`, `weight`, `bias`, `running_mean`, and
    `running_var` tensors as input into `forward()`.
    """

    def __init__(self, eps=1e-05, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.inv_var = None
        self._running_var_id = None

    def forward(self, x):
        assert len(x), f"BatchNormalization expects 5 inputs, not {len(x)}"
        input, weight, bias, running_mean, running_var = x

        # in inference mode, we may be able to re-use inverse variance:
        if not self.train:
            if id(running_var) != self._running_var_id:
                self.inv_var = self._compute_inv_var(running_var)
                self._running_var_id = id(running_var)
        else:
            self.inv_var = None
            self._running_var_id = None

        # perform batch normalization:
        output = input.batchnorm(
            weight,
            bias,
            running_mean=running_mean,
            running_var=running_var,
            training=self.training,
            eps=self.eps,
            momentum=self.momentum,
            inv_var=self.inv_var,
        )
        if self.training:  # NOTE: Training graph is different from evaluation graph.
            return output, running_mean, running_var, None, None
        else:
            return output

    def _compute_inv_var(self, running_var):
        """Computes inverse variance."""
        if isinstance(running_var, CrypTensor):
            inv_var = running_var.add(self.eps).inv_sqrt()
        else:
            inv_var = running_var.add(self.eps).sqrt().reciprocal()
        return inv_var

    @staticmethod
    def from_onnx(attributes=None):
        if attributes is None:
            attributes = {}
        return BatchNormalization(
            eps=attributes.get("epsilon", 1e-05),
            momentum=1.0 - attributes.get("momentum", 0.9),
        )  # NOTE: Role of momentum is reversed in ONNX specification.


class _BatchNorm(Module):
    """
    Module that performs batch normalization on 1D tensors. It is used as the
    base implementation for the `BatchNorm1d`, `BatchNorm2d`, and `BatchNorm3d`
    classes.

    Unlike the `BatchNormalization` class, this module is stateful.
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()

        # initialize model parameters and buffers:
        pytorch_module = torch.nn.BatchNorm1d(num_features)
        for param in ["weight", "bias"]:
            self.register_parameter(param, getattr(pytorch_module, param))
        for buffer in ["running_mean", "running_var"]:
            self.register_buffer(buffer, getattr(pytorch_module, buffer))

        # set model attributes:
        self.eps = eps
        self.momentum = momentum

        # do not precompute inverse variance during training
        self.inv_var = None

    def forward(self, input):
        return input.batchnorm(
            self.weight,
            self.bias,
            running_mean=self.running_mean,
            running_var=self.running_var,
            training=self.training,
            eps=self.eps,
            momentum=self.momentum,
            inv_var=self.inv_var,
        )

    @staticmethod
    def from_onnx(parameters=None, attributes=None):
        # preprocess all attributes:
        if parameters is None:
            parameters = {}
        if attributes is None:
            attributes = {}

        num_features = len(parameters["running_mean"])

        # create module:
        if "momentum" not in attributes:
            attributes["momentum"] = 0.1
        kwargs = {"eps": attributes["epsilon"], "momentum": attributes["momentum"]}
        module = _BatchNorm(num_features, **kwargs)

        # set parameters:
        for key, value in parameters.items():
            if key in ["running_mean", "running_var"]:
                module.set_buffer(key, value)
            else:
                module.set_parameter(key, value)
        return module

    def train(self, mode=True):
        """Freezes the inverse variance during inference to save computation"""
        super().train(mode=mode)
        if self.training:
            self.inv_var = None
        elif isinstance(self.running_var, CrypTensor):
            self.inv_var = self.running_var.add(self.eps).inv_sqrt()
        else:
            self.inv_var = self.running_var.add(self.eps).sqrt().reciprocal()
        return self


class BatchNorm1d(_BatchNorm):
    """
    Module that performs batch normalization on 1D tensors.
    """

    pass


class BatchNorm2d(_BatchNorm):
    """
    Module that performs batch normalization on 2D tensors.
    """

    pass


class BatchNorm3d(_BatchNorm):
    """
    Module that performs batch normalization on 3D tensors.
    """

    pass


class GroupNorm(Module):
    """
    Module that performs group normalization on tensors.
    """

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        raise NotImplementedError("GroupNorm is not implemented.")


def _all_the_same(items):
    """
    Checks whether all values in a list are the same.
    """
    return all(items[0] == item for item in items)


def _identify_bool_attributes_with_defaults(attributes, attr_name, attr_value, default=True):
    """For boolean attributes that have default values in the ONNX specification
    checks to see if they are present in `attributes`, and assigns the
    default if not present and appropriate value if present. Note `attr_value`
    must be the value `attributes[attr_name]` if the default is to be kept.
    """
    output = default
    if attr_name in attributes and attributes[attr_name] != attr_value:
        output = not default
    return output
