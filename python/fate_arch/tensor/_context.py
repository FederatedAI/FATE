from typing import TYPE_CHECKING, Any, List, Tuple
from typing_extensions import Literal
from contextlib import contextmanager

from ._federation import _Parties
import typing
from enum import Enum

from fate_arch.common import Party
from fate_arch.federation.transfer_variable import IterationGC
from fate_arch.session import get_session
from .abc.tensor import PHEEncryptorABC, PHEDecryptorABC

# for better type checking
if TYPE_CHECKING:
    from ._tensor import FPTensor, PHETensor


class ExcutionState:
    def __init__(self, tag) -> None:
        self._tag = tag

    def generate_tag(self) -> str:
        return self._tag


class DefaultState(ExcutionState):
    def __init__(self) -> None:
        super().__init__("default")


class FitState(ExcutionState):
    ...


class PredictState(ExcutionState):
    ...


class IterationState(ExcutionState):
    def __init__(self, tag: str, index: int = -1) -> None:
        self._tag = tag
        self._index = index

    def generate_tag(self) -> str:
        return self._tag


class CipherKind(Enum):
    PHE = 1
    PHE_PAILLIER = 2


class Device(Enum):
    CPU = 1
    GPU = 2
    FPGA = 3


class Context:
    def __init__(self, start_iter_num=-1) -> None:
        self._device = None
        self._iter_num = start_iter_num
        self._push_gc_dict = {}
        self._pull_gc_dict = {}

        self._binded_variables = {}
        self._current_iter = None
        self._flowid = None

        self._execution_state = DefaultState()

        self._cypher_utils = CypherUtils(self)
        self._tensor_utils = TensorUtils(self)

    def device_init(self, **kwargs):
        self._device = Device.CPU

    def device(self) -> Device:
        if self._device is None:
            raise RuntimeError(f"init device first")
        return self._device

    @property
    def cypher_utils(self):
        return self._cypher_utils

    @property
    def tensor_utils(self):
        return self._tensor_utils

    @contextmanager
    def create_iter(self, max_iter, template="{i}"):
        # cache previous state
        previous_state = self._execution_state
        current_tag = self.generate_federation_tag()

        def _state_iterator():
            for i in range(max_iter):
                # the tags in the iteration need to be distinguishable
                template_formated = template.format(i=i)
                self._execution_state = IterationState(f"{current_tag}.{template_formated}", i)
                yield i

        yield _state_iterator()
        # recover state
        self._execution_state = previous_state

    def generate_federation_tag(self):
        return self._execution_state.generate_tag()

    def remote(self, target: _Parties, key: str, value):
        self._push(target.parties, key, value)
        return self

    def get(self, source: _Parties, key: str):
        return self._pull(source.parties, key)[0]

    def get_multi(self, source: _Parties, key: str) -> List:
        return self._pull(source.parties, key)

    def _push(self, parties: List[Party], key, value):
        if key not in self._push_gc_dict:
            self._push_gc_dict[key] = IterationGC()
        get_session().federation.remote(
            v=value,
            name=key,
            tag=self.generate_federation_tag(),
            parties=parties,
            gc=self._push_gc_dict[key],
        )

    def _pull(self, parties: List[Party], key):
        if key not in self._pull_gc_dict:
            self._pull_gc_dict[key] = IterationGC()
        return get_session().federation.get(
            name=key,
            tag=self.generate_federation_tag(),
            parties=parties,
            gc=self._pull_gc_dict[key],
        )


class TensorUtils:
    """utils for tensor operation such as:
    1. creation: zeros, ones, random
    2. recv from remote: get, get_multi

    Notes:
        1. methods perfix with `phe_` is bound to `PHETensor`
        2. others is bound to `FPTensor`
    """

    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    def zeros(self, shape) -> "FPTensor":
        ...

    def get(self, source: _Parties, key: str) -> "FPTensor":
        from ._tensor import FPTensor

        tensor = self._ctx.get(source, key)
        if not isinstance(tensor, FPTensor):
            raise ValueError(
                f"{PHETensor.__name__} expected while {type(tensor).__name__} got"
            )
        return tensor

    def get_multi(self, source: _Parties, key: str) -> typing.List["FPTensor"]:
        from ._tensor import FPTensor

        tensors = self._ctx.get_multi(source, key)
        for tensor in tensors:
            if not isinstance(tensor, PHETensor):
                raise ValueError(
                    f"{FPTensor.__name__} expected while {type(tensor).__name__} got"
                )
        return tensors

    def phe_get(self, source: _Parties, key: str) -> "PHETensor":
        from ._tensor import PHETensor

        tensor = self._ctx.get(source, key)
        if not isinstance(tensor, PHETensor):
            raise ValueError(
                f"{PHETensor.__name__} expected while {type(tensor).__name__} got"
            )
        return tensor

    def phe_get_multi(self, source: _Parties, key: str) -> typing.List["PHETensor"]:
        from ._tensor import PHETensor

        tensors = self._ctx.get_multi(source, key)
        for tensor in tensors:
            if not isinstance(tensor, PHETensor):
                raise ValueError(
                    f"{PHETensor.__name__} expected while {type(tensor).__name__} got"
                )
        return tensors


class CypherUtils:
    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx

    @typing.overload
    def keygen(
        self, kind: Literal[CipherKind.PHE], key_length: int
    ) -> Tuple["PHEEncryptor", "PHEDecryptor"]:
        ...

    @typing.overload
    def keygen(self, kind: CipherKind, **kwargs) -> Any:
        ...

    def keygen(self, kind, key_length: int, **kwargs):
        if kind == CipherKind.PHE or kind == CipherKind.PHE_PAILLIER:
            if self._ctx._device == Device.CPU:
                from .impl.tensor.multithread import PaillierPHECipherLocal

                encryptor, decryptor = PaillierPHECipherLocal().keygen(
                    key_length=key_length
                )
                return PHEEncryptor(encryptor), PHEDecryptor(decryptor)
        else:
            raise NotImplementedError(f"keygen for kind `{kind}` is not implemented")

    def phe_get_encryptor(self, source: _Parties, key: str) -> "PHEEncryptor":
        encryptor = self._ctx.get(source, key)
        ...


class PHEEncryptor:
    def __init__(self, encryptor: PHEEncryptorABC) -> None:
        self._encryptor = encryptor

    def encrypt(self, tensor: "FPTensor"):
        from ._tensor import PHETensor

        return PHETensor(tensor._ctx, self._encryptor.encrypt(tensor._tensor))

    @classmethod
    def get(cls, ctx: Context, source: _Parties, key: str) -> "PHEEncryptor":
        return PHEEncryptor(ctx.get(source, key))

    @classmethod
    def get_multi(
        cls, ctx: Context, source: _Parties, key: str
    ) -> List["PHEEncryptor"]:
        return [PHEEncryptor(encryptor) for encryptor in ctx.get_multi(source, key)]

    def remote(self, ctx: Context, target: _Parties, key: str):
        return ctx.remote(target, key, self._encryptor)


class PHEDecryptor:
    def __init__(self, decryptor: PHEDecryptorABC) -> None:
        self._decryptor = decryptor

    def decrypt(self, tensor: "PHETensor") -> "FPTensor":
        from ._tensor import FPTensor

        return FPTensor(tensor._ctx, self._decryptor.decrypt(tensor._tensor))
