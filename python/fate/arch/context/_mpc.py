#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import typing
import warnings
from typing import overload

import torch

from fate.arch.protocol import mpc
import logging

if typing.TYPE_CHECKING:
    from fate.arch.context import Context
    from fate.arch.protocol.mpc.primitives import ArithmeticSharedTensor
logger = logging.getLogger(__name__)

TO1 = typing.TypeVar("TO1")
TO2 = typing.TypeVar("TO2")


class MPC:
    def __init__(self, ctx: "Context"):
        self._ctx = ctx

    @property
    def rank(self):
        from fate.arch.protocol.mpc.communicator import Communicator

        return Communicator.get().get_rank()

    @property
    def world_size(self):
        from fate.arch.protocol.mpc.communicator import Communicator

        return Communicator.get().get_world_size()

    def init(self):
        """
        Initializes the MPCTensor module by initializing the default provider
        and setting up the RNG generators.
        """
        from fate.arch.protocol.mpc import ttp_required, _setup_prng
        from fate.arch.protocol.mpc.communicator import Communicator

        if Communicator.is_initialized():
            warnings.warn("CrypTen is already initialized.", RuntimeWarning)
            return

        # Initialize communicator
        Communicator.initialize(self._ctx, init_ttp=ttp_required())

        # # Setup party name for file save / load
        # if party_name is not None:
        #     comm.get().set_name(party_name)
        #
        # Setup seeds for Random Number Generation
        if Communicator.get().get_rank() < Communicator.get().get_world_size():
            from fate.arch.protocol.mpc import generators

            _setup_prng(self._ctx, generators)
            if ttp_required():
                from fate.arch.protocol.mpc.provider.ttp_provider import TTPClient

                TTPClient._init()

    @property
    def communicator(self):
        from fate.arch.protocol.mpc.communicator import Communicator

        return Communicator.get()

    def encrypt(self, *args, cryptensor_type=None, **kwargs):
        return mpc.cryptensor(self._ctx, *args, cryptensor_type=cryptensor_type, **kwargs)

    def lazy_encrypt(self, f: typing.Callable[[], torch.Tensor], *args, cryptensor_type=None, **kwargs):
        src = kwargs.get("src")
        assert src is not None, "src should not be None"
        tensor = f() if self.rank == src else None
        x = self.encrypt(tensor, *args, cryptensor_type=cryptensor_type, broadcast_size=True, **kwargs)
        return x

    def encode(self, x, precision_bits=None):
        from fate.arch.protocol.mpc.mpc import FixedPointEncoder

        return FixedPointEncoder(precision_bits=precision_bits).encode(x)

    @classmethod
    def is_encrypted_tensor(cls, obj):
        """
        Returns True if obj is an encrypted tensor.
        """
        from fate.arch.protocol.mpc import CrypTensor

        return isinstance(obj, CrypTensor)

    def print(self, message, dst=None, print_func=None):
        if dst is None:
            dst = [0]
        if print_func is None:
            print_func = print
        if self.rank in dst:
            print_func(message)

    @overload
    def info(self, message, dst: int = 0):
        pass

    @overload
    def info(self, message, dst: typing.List[int]):
        pass

    def info(self, message, **kwargs):
        dst = kwargs.get("dst", None)
        return self._log(message, logging.INFO, dst)

    @overload
    def debug(self, message, dst: int = 0):
        pass

    @overload
    def debug(self, message, dst: typing.List[int]):
        pass

    def debug(self, message, **kwargs):
        dst = kwargs.get("dst", None)
        return self._log(message, logging.DEBUG, dst)

    @overload
    def warning(self, message, dst: int = 0):
        pass

    @overload
    def warning(self, message, dst: typing.List[int]):
        pass

    def warning(self, message, **kwargs):
        dst = kwargs.get("dst", None)
        return self._log(message, logging.WARNING, dst)

    @overload
    def error(self, message, dst: int = 0):
        pass

    @overload
    def error(self, message, dst: typing.List[int]):
        pass

    def error(self, message, **kwargs):
        dst = kwargs.get("dst", None)
        return self._log(message, logging.ERROR, dst)

    def _log(self, message, level, dst):
        if dst is None:
            dst = [0]
        if isinstance(dst, int):
            dst = [dst]
        if self.rank in dst:
            logger.log(msg=message, stacklevel=3, level=level)

    def cond(self, value1, value2, dst):
        """
        Returns value1 if rank == dst, otherwise returns value2.
        """
        return value1 if self.rank == dst else value2

    def cond_call(
        self, func1: typing.Callable[[], TO1], func2: typing.Callable[[], TO2], dst
    ) -> typing.Union[TO1, TO2]:
        """
        Calls func1 if rank == dst, otherwise calls func2.
        """
        if self.rank == dst:
            return func1()
        else:
            return func2() if func2 is not None else None

    def option(self, value, dst):
        """
        Returns value if rank == dst, otherwise returns None.
        """
        return value if self.rank == dst else None

    def option_call(self, func: typing.Callable[[], TO1], dst) -> typing.Union[TO1, None]:
        """
        Calls func if rank == dst, otherwise returns None.
        """
        return func() if self.rank == dst else None

    def option_assert(self, cond, message, dst):
        """
        Asserts cond if rank == dst, otherwise returns None.
        """
        if self.rank == dst:
            assert cond, f"rank:{dst} {message}"

    def split_variable(self, x, *ranks: typing.List[int]):
        """
        Split x into len(ranks) parts, each part is None except the part corresponding to the rank in ranks.
        """
        return (x if self.rank == rank else None for rank in ranks)

    @property
    def sshe(self):
        from fate.arch.protocol.mpc.primitives.sshe import SSHE

        return SSHE

    def init_tensor(self, shape, init_func, src):
        from fate.arch.protocol.mpc.primitives import ArithmeticSharedTensor

        if self.rank == src:
            return ArithmeticSharedTensor(self._ctx, init_func(shape), src=src)

        else:
            return ArithmeticSharedTensor(self._ctx, None, size=shape, src=src)
