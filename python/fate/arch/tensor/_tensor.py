from typing import Any, Union, overload

from fate.arch.common import Party

from .abc.tensor import PHEDecryptorABC, PHEEncryptorABC, PHETensorABC


class PHEEncryptor:
    def __init__(self, encryptor: PHEEncryptorABC) -> None:
        self._encryptor = encryptor

    def encrypt(self, tensor: "FPTensor"):

        return PHETensor(self._encryptor.encrypt(tensor._tensor))


class PHEDecryptor:
    def __init__(self, decryptor: PHEDecryptorABC) -> None:
        self._decryptor = decryptor

    def decrypt(self, tensor: "PHETensor") -> "FPTensor":

        return FPTensor(self._decryptor.decrypt(tensor._tensor))


class FPTensor:
    def __init__(self, tensor) -> None:
        self._tensor = tensor

    @property
    def shape(self):
        return self._tensor.shape

    def __add__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__add__"):
            return NotImplemented
        return self._binary_op(other, self._tensor.__add__)

    def __radd__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__radd__"):
            return self.__add__(other)
        return self._binary_op(other, self._tensor.__add__)

    def __sub__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__sub__"):
            return NotImplemented
        return self._binary_op(other, self._tensor.__sub__)

    def __rsub__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__rsub__"):
            return self.__mul__(-1).__add__(other)
        return self._binary_op(other, self._tensor.__rsub__)

    def __mul__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__mul__"):
            return NotImplemented
        return self._binary_op(other, self._tensor.__mul__)

    def __rmul__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        if not hasattr(self._tensor, "__rmul__"):
            return self.__mul__(other)
        return self._binary_op(other, self._tensor.__rmul__)

    def __matmul__(self, other: "FPTensor") -> "FPTensor":
        if not hasattr(self._tensor, "__matmul__"):
            return NotImplemented
        if isinstance(other, FPTensor):
            return FPTensor(self._tensor.__matmul__(other._tensor))
        else:
            return NotImplemented

    def __rmatmul__(self, other: "FPTensor") -> "FPTensor":
        if not hasattr(self._tensor, "__rmatmul__"):
            return NotImplemented
        if isinstance(other, FPTensor):
            return FPTensor(self._tensor.__rmatmul__(other._tensor))
        else:
            return NotImplemented

    def _binary_op(self, other, func):
        if isinstance(other, FPTensor):
            return FPTensor(func(other._tensor))
        elif isinstance(other, (int, float)):
            return FPTensor(func(other))
        else:
            return NotImplemented

    @property
    def T(self):
        return FPTensor(self._tensor.T)

    def __federation_hook__(self, ctx, key, parties):
        deserializer = FPTensorFederationDeserializer(key)
        # 1. remote deserializer with objs
        ctx._push(parties, key, deserializer)
        # 2. remote table
        ctx._push(parties, deserializer.table_key, self._tensor)


class PHETensor:
    def __init__(self, tensor: PHETensorABC) -> None:
        self._tensor = tensor

    @property
    def shape(self):
        return self._tensor.shape

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
            return PHETensor(self._tensor.__matmul__(other._tensor))
        else:
            return NotImplemented

    def __rmatmul__(self, other: FPTensor) -> "PHETensor":
        if isinstance(other, FPTensor):
            return PHETensor(self._tensor.__rmatmul__(other._tensor))
        else:
            return NotImplemented

    def T(self) -> "PHETensor":
        return PHETensor(self._tensor.T())

    @overload
    def decrypt(self, decryptor: "PHEDecryptor") -> FPTensor:
        ...

    @overload
    def decrypt(self, decryptor) -> Any:
        ...

    def decrypt(self, decryptor):
        return decryptor.decrypt(self)

    def _binary_op(self, other, func):
        if isinstance(other, (PHETensor, FPTensor)):
            return PHETensor(func(other._tensor))
        elif isinstance(other, (int, float)):
            return PHETensor(func(other))
        return NotImplemented

    def __federation_hook__(self, ctx, key, parties):
        deserializer = PHETensorFederationDeserializer(key)
        # 1. remote deserializer with objs
        ctx._push(parties, key, deserializer)
        # 2. remote table
        ctx._push(parties, deserializer.table_key, self._tensor)


class PHETensorFederationDeserializer(FederationDeserializer):
    def __init__(self, key) -> None:
        self.table_key = self.make_frac_key(key, "tensor")

    def do_deserialize(self, ctx: Context, party: Party) -> PHETensor:
        tensor = ctx._pull([party], self.table_key)[0]
        return PHETensor(ctx, tensor)


class FPTensorFederationDeserializer(FederationDeserializer):
    def __init__(self, key) -> None:
        self.table_key = self.make_frac_key(key, "tensor")

    def do_deserialize(self, ctx: Context, party: Party) -> FPTensor:
        tensor = ctx._pull([party], self.table_key)[0]
        return FPTensor(ctx, tensor)
