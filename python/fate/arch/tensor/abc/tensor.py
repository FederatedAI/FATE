import abc
import typing
from typing import Protocol


class FPTensorProtocol(Protocol):
    def __add__(
        self, other: typing.Union["FPTensorProtocol", float, int]
    ) -> "FPTensorProtocol":
        ...

    def __radd__(
        self, other: typing.Union["FPTensorProtocol", float, int]
    ) -> "FPTensorProtocol":
        ...

    def __sub__(
        self, other: typing.Union["FPTensorProtocol", float, int]
    ) -> "FPTensorProtocol":
        ...

    def __rsub__(
        self, other: typing.Union["FPTensorProtocol", float, int]
    ) -> "FPTensorProtocol":
        ...

    def __mul__(
        self, other: typing.Union["FPTensorProtocol", float, int]
    ) -> "FPTensorProtocol":
        ...

    def __rmul__(
        self, other: typing.Union["FPTensorProtocol", float, int]
    ) -> "FPTensorProtocol":
        ...

    def __matmul__(self, other: "FPTensorProtocol") -> "FPTensorProtocol":
        ...

    def __rmatmul__(self, other: "FPTensorProtocol") -> "FPTensorProtocol":
        ...


class PHETensorABC(abc.ABC):
    """Tensor implements Partial Homomorphic Encryption schema:
    1. decrypt(encrypt(a) + encrypt(b)) = a + b
    2. decrypt(encrypt(a) * b) = a * b
    """

    @abc.abstractmethod
    def __add__(
        self, other: typing.Union["PHETensorABC", "FPTensorProtocol", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __radd__(
        self, other: typing.Union["PHETensorABC", "FPTensorProtocol", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __sub__(
        self, other: typing.Union["PHETensorABC", "FPTensorProtocol", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __rsub__(
        self, other: typing.Union["PHETensorABC", "FPTensorProtocol", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __mul__(
        self, other: typing.Union["PHETensorABC", "FPTensorProtocol", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __rmul__(
        self, other: typing.Union["PHETensorABC", "FPTensorProtocol", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __matmul__(self, other: FPTensorProtocol) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __rmatmul__(self, other: FPTensorProtocol) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def serialize(self):
        ...

    @abc.abstractmethod
    def T(self) -> "PHETensorABC":
        ...


class PHEEncryptorABC(abc.ABC):
    @abc.abstractmethod
    def encrypt(self, tensor: FPTensorProtocol) -> PHETensorABC:
        ...


class PHEDecryptorABC(abc.ABC):
    @abc.abstractmethod
    def decrypt(self, tensor: PHETensorABC) -> FPTensorProtocol:
        ...


class PHECipherABC(abc.ABC):
    @abc.abstractclassmethod
    def keygen(cls, **kwargs) -> typing.Tuple[PHEEncryptorABC, PHEDecryptorABC]:
        ...
