#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pdb as pythondebugger
import sys

from fate.arch.protocol.mpc.config import cfg


class MultiprocessingPdb(pythondebugger.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            with open("/dev/stdin") as file:
                sys.stdin = file
                pythondebugger.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def configure_logging():
    """Configures a logging template useful for debugging multiple processes."""

    level = logging.INFO
    logging.getLogger().setLevel(level)
    logging.basicConfig(
        level=level,
        format=("[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d]" + "[%(processName)s] %(message)s"),
    )


def crypten_print(*args, dst=0, **kwargs):
    """
    Prints a message to only parties whose rank is contained by `dst` kwarg (default: 0).
    """
    if isinstance(dst, int):
        dst = [dst]
    assert isinstance(dst, (list, tuple)), "print destination must be a list or tuple of party ranks"
    import crypten.communicator as comm

    if comm.get().get_rank() in dst:
        print(*args, **kwargs)


def crypten_log(*args, level=logging.INFO, dst=0, **kwargs):
    """
    Logs a message to logger of parties whose rank is contained by `dst` kwarg (default: 0).

    Uses logging.INFO as default level.
    """
    if isinstance(dst, int):
        dst = [dst]
    assert isinstance(dst, (list, tuple)), "log destination must be a list or tuple of party ranks"
    import crypten.communicator as comm

    if comm.get().get_rank() in dst:
        logging.log(level, *args, **kwargs)


def crypten_print_in_order(*args, **kwargs):
    """
    Calls print(*args, **kwargs) on each party in rank order to ensure each party
    can print its full message uninterrupted and the full output is deterministic
    """
    import crypten.communicator as comm

    for i in range(comm.get().get_world_size()):
        if comm.get().get_rank() == i:
            print(*args, **kwargs)
        comm.get().barrier()


def validate_correctness(self, func, func_name, tolerance=0.5):
    import crypten
    import torch

    if not hasattr(torch.tensor([]), func_name):
        return func

    def validation_function(*args, **kwargs):
        with cfg.temp_override({"debug.validation_mode": False}):
            # Compute crypten result
            result_enc = func(*args, **kwargs)
            result = result_enc.get_plain_text() if crypten.is_encrypted_tensor(result_enc) else result_enc

            args = list(args)

            # Compute torch result for corresponding function
            for i, arg in enumerate(args):
                if crypten.is_encrypted_tensor(arg):
                    args[i] = args[i].get_plain_text()

            kwargs.pop("input_in_01", None)
            for key, value in kwargs.items():
                if crypten.is_encrypted_tensor(value):
                    kwargs[key] = value.get_plain_text()
            reference = getattr(self.get_plain_text(), func_name)(*args, **kwargs)

            # TODO: Validate properties - Issue is tuples can contain encrypted tensors
            if not torch.is_tensor(reference):
                return result_enc

            # Check sizes match
            if result.size() != reference.size():
                crypten_log(f"Size mismatch: Expected {reference.size()} but got {result.size()}")
                raise ValueError(f"Function {func_name} returned incorrect size")

            # Check that results match
            diff = (result - reference).abs_()
            norm_diff = diff.div(result.abs() + reference.abs()).abs_()
            test_passed = norm_diff.le(tolerance) + diff.le(tolerance * 0.1)
            test_passed = test_passed.gt(0).all().item() == 1
            if not test_passed:
                crypten_log(f"Function {func_name} returned incorrect values")
                crypten_log("Result %s" % result)
                crypten_log("Result - Reference = %s" % (result - reference))
                raise ValueError(f"Function {func_name} returned incorrect values")

        return result_enc

    return validation_function
