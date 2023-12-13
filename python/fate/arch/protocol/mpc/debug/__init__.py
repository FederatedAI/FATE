#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from functools import wraps

from fate.arch.protocol.mpc.config import cfg
from .debug import configure_logging, MultiprocessingPdb, validate_correctness

pdb = MultiprocessingPdb()

__all__ = ["pdb", "configure_logging", "validate_correctness"]


def register_validation(getattr_function):
    @wraps(getattr_function)
    def validate_attribute(self, name):
        # Get dispatched function call
        function = getattr_function(self, name)

        if not cfg.safety.mpc.debug.validation_mode:
            return function

        # Run validation
        return validate_correctness(self, function, name)

    return validate_attribute
