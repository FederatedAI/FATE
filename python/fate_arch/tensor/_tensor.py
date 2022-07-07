import typing
from typing import Any, Union
from ._federation import _Parties
from ._context import Context, PHEEncryptor, PHEDecryptor
from .abc.tensor import FPTensorABC, PHETensorABC


class FPTensor:
    def __init__(self, ctx: Context, tensor: FPTensorABC) -> None:
        self._ctx = ctx
        self._tensor = tensor

    def __add__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__add__)

    def __radd__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__add__)

    def __sub__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__sub__)

    def __rsub__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__rsub__)

    def __mul__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__mul__)

    def __rmul__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        return self._binary_op(other, self._tensor.__rmul__)

    def __matmul__(self, other: "FPTensor") -> "FPTensor":
        if isinstance(other, FPTensor):
            return FPTensor(self._ctx, self._tensor.__matmul__(other._tensor))
        else:
            return NotImplemented

    def __rmatmul__(self, other: "FPTensor") -> "FPTensor":
        if isinstance(other, FPTensor):
            return FPTensor(self._ctx, self._tensor.__rmatmul__(other._tensor))
        else:
            return NotImplemented

    @typing.overload
    def encrypted(self, encryptor: "PHEEncryptor") -> "PHETensor":
        ...

    @typing.overload
    def encrypted(self, encryptor):
        ...

    def encrypted(self, encryptor):
        return encryptor.encrypt(self)

    def remote(self, target: _Parties, name: str):
        return self._ctx.remote(target, name, self)

    @classmethod
    def get(cls, ctx: Context, source: _Parties, name: str) -> "FPTensor":
        return ctx.get(source, name)

    def _binary_op(self, other, func):
        if isinstance(other, FPTensor):
            return FPTensor(self._ctx, func(other._tensor))
        elif isinstance(other, (int, float)):
            return FPTensor(self._ctx, func(other))
        else:
            return NotImplemented

    @property
    def T(self):
        return FPTensor(self._ctx, self._tensor.T)


class PHETensor:
    def __init__(self, ctx: Context, tensor: PHETensorABC) -> None:
        self._tensor = tensor
        self._ctx = ctx

    def __add__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__add__)

    def __radd__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__radd__)

    def __sub__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__sub__)

    def __rsub__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__rsub__)

    def __mul__(self, other: Union[FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__mul__)

    def __rmul__(self, other: Union[FPTensor, int, float]) -> "PHETensor":
        return self._binary_op(other, self._tensor.__rmul__)

    def __matmul__(self, other: FPTensor) -> "PHETensor":
        if isinstance(other, FPTensor):
            return PHETensor(self._ctx, self._tensor.__matmul__(other._tensor))
        else:
            return NotImplemented

    def __rmatmul__(self, other: FPTensor) -> "PHETensor":
        if isinstance(other, FPTensor):
            return PHETensor(self._ctx, self._tensor.__rmatmul__(other._tensor))
        else:
            return NotImplemented

    def T(self) -> "PHETensor":
        return PHETensor(self._ctx, self._tensor.T())

    @typing.overload
    def decrypt(self, decryptor: "PHEDecryptor") -> FPTensor:
        ...

    @typing.overload
    def decrypt(self, decryptor) -> Any:
        ...

    def decrypt(self, decryptor):
        return decryptor.decrypt(self)

    def remote(self, target: _Parties, name: str):
        return self._ctx.remote(target, name, self)

    @classmethod
    def get(cls, ctx: Context, source: _Parties, name: str) -> "PHETensor":
        return ctx.get(source, name)

    def _binary_op(self, other, func):
        if isinstance(other, (PHETensor, FPTensor)):
            return PHETensor(self._ctx, func(other._tensor))
        elif isinstance(other, (int, float)):
            return PHETensor(self._ctx, func(other))
        return NotImplemented
