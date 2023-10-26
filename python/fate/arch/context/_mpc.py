from fate.arch.protocol.mpc.communicator.communicator import Communicator
from fate.arch.tensor.mpc.cryptensor import CrypTensor
from fate.arch.protocol.mpc import init
from fate.arch.tensor import mpc


class MPC:
    def __init__(self, ctx):
        self._ctx = ctx

    def init(self):
        init(self._ctx)

    @property
    def communicator(self):
        return Communicator.get()

    @classmethod
    def cryptensor(cls, *args, cryptensor_type=None, **kwargs):
        return mpc.cryptensor(*args, cryptensor_type=cryptensor_type, **kwargs)

    @classmethod
    def is_encrypted_tensor(cls, obj):
        """
        Returns True if obj is an encrypted tensor.
        """
        return isinstance(obj, CrypTensor)
