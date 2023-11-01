#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import logging
import sys

import fate.arch.tensor.mpc.nn as cnn
import torch

# List of modules to import and additional classes to update from them
__import_list = {
    "alexnet": [],
    "densenet": ["_DenseLayer", "_DenseBlock", "_Transition"],
    "googlenet": ["Inception", "InceptionAux", "BasicConv2d"],
    "inception": [
        "BasicConv2d",
        "InceptionA",
        "InceptionB",
        "InceptionC",
        "InceptionD",
        "InceptionE",
        "InceptionAux",
    ],
    "mnasnet": ["_InvertedResidual"],
    "mobilenet": [],
    "resnet": ["BasicBlock", "Bottleneck"],
    "shufflenetv2": ["InvertedResidual"],
    "squeezenet": ["Fire"],
    "vgg": [],
}


__all__ = []


def __import_module_copy(module_name):
    """
    Returns a copy of an imported module so it can be modified
    without modifying future imports of the given module
    """
    starting_modules = sys.modules.copy()

    module_spec = importlib.util.find_spec(module_name)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    new_modules = set(sys.modules) - set(starting_modules)

    del module_spec
    for m in new_modules:
        del sys.modules[m]

    return module


def __import_model_package_copy(import_name):
    """
    Returns a copy of an imported model whose package contains
    a function of the same name.
    """
    starting_modules = sys.modules.copy()

    model_type = importlib.import_module(f"torchvision.models.{import_name}")
    new_modules = set(sys.modules) - set(starting_modules)
    for m in new_modules:
        del sys.modules[m]

    return model_type


def __update_model_class_inheritance(cls):
    """
    Updates the class inheritance of a torch.nn.Module to instead use
    crypten.nn.Module
    """
    bases = []
    for m in cls.__bases__:
        if m == torch.nn.Module:
            bases.append(cnn.Module)
        elif m == torch.nn.Sequential:
            bases.append(cnn.Sequential)
        elif m == torch.nn.ModuleDict:
            bases.append(cnn.ModuleDict)
        else:
            bases.append(m)

    cls.__bases__ = tuple(bases)


class FunctionalReplacement:
    """Replacement for `torch.nn.functional` that overwrites torch functionals to be crypten compatible"""

    @staticmethod
    def dropout(x, **kwargs):
        return x.dropout(**kwargs)

    @staticmethod
    def relu(x, **kwargs):
        return x.relu()

    @staticmethod
    def adaptive_avg_pool2d(x, *args):
        return cnn.AdaptiveAvgPool2d(*args)(x)

    @staticmethod
    def avg_pool2d(x, *args, **kwargs):
        return x.avg_pool2d(*args, **kwargs)

    @staticmethod
    def max_pool2d(x, *args, **kwargs):
        return x.max_pool2d(*args, **kwargs)


def __update_torch_functions(module):
    if hasattr(module, "nn"):
        module.nn = cnn

    # TODO: fix replacement in global `torch` module - perhaps use __torch_function__
    if hasattr(module, "torch"):
        module.torch.flatten = lambda x, *args: x.flatten(*args)
        module.torch.transpose = lambda x, *args: x.transpose(*args)
        # module.torch.cat = lambda *args, **kwargs: args[0].cat(*args, **kwargs)

    if hasattr(module, "F"):
        module.F = FunctionalReplacement()


def __get_module_list(model_name, model_type):
    return __import_list[model_name] + model_type.__all__


try:
    models = __import_module_copy("torchvision").models

except ModuleNotFoundError:
    models = None
    logging.warning("Unable to load torchvision models.")


if models is not None:
    for import_name in __import_list.keys():
        try:
            model_type = getattr(models, import_name)
        except AttributeError:
            logging.warning(f"Could not load {import_name} from torchvision.modules")
            continue

        try:
            # If function imported rather than package, replace with package
            if not hasattr(model_type, "__all__"):
                model_type = __import_model_package_copy(import_name)

            __update_torch_functions(model_type)
            module_list = __get_module_list(import_name, model_type)
            for module_name in module_list:
                module = getattr(model_type, module_name)

                # Replace class inheritance from torch.nn.Module to crypten.nn.Module
                if isinstance(module, type):
                    __update_model_class_inheritance(module)

                module.load_state_dict = (
                    lambda *args, **kwargs: cnn.Module.load_state_dict(
                        *args, strict=False, **kwargs
                    )
                )

                if module_name in model_type.__all__:
                    globals()[module_name] = module
                    __all__.append(module_name)
        except (RuntimeError, AttributeError) as e:
            # Log that module produced an error
            logging.warning(e)


raise DeprecationWarning(
    "crypten.models is being deprecated. To import models from torchvision, ",
    "please import them directly and use crypten.nn.from_pytorch() to convert",
    " to CrypTen models.",
)
