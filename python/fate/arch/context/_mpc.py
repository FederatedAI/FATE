import typing
import warnings
from typing import overload


from fate.arch.protocol import mpc
import logging

if typing.TYPE_CHECKING:
    from fate.arch.context import Context
    from fate.arch.tensor.phe import PHETensor, PHETensorDecryptor
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

    def cond_call(
        self, func1: typing.Callable[[], TO1], func2: typing.Callable[[], TO2] = None, dst=0
    ) -> typing.Union[TO1, TO2]:
        """
        Calls func1 if rank == dst, otherwise calls func2.
        """
        if self.rank == dst:
            return func1()
        else:
            return func2() if func2 is not None else None

    @property
    def sshe(self):
        from fate.arch.protocol.mpc.sshe import SSHE

        return SSHE

    def random_tensor(self, shape, src=0, generator=None):
        import torch
        from fate.arch.protocol.mpc.primitives import ArithmeticSharedTensor

        if self.rank == src:
            return ArithmeticSharedTensor(self._ctx, torch.rand(shape, generator=generator), src=src)

        else:
            return ArithmeticSharedTensor(self._ctx, None, size=shape, src=src)
