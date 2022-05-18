import abc
import typing


class FPTensorABC(abc.ABC):
    @classmethod
    def zeors(cls, shape) -> "FPTensorABC":
        ...

    @abc.abstractmethod
    def __add__(self, other: typing.Union["FPTensorABC", float, int]) -> "FPTensorABC":
        ...

    @abc.abstractmethod
    def __radd__(self, other: typing.Union["FPTensorABC", float, int]) -> "FPTensorABC":
        ...

    @abc.abstractmethod
    def __sub__(self, other: typing.Union["FPTensorABC", float, int]) -> "FPTensorABC":
        ...

    @abc.abstractmethod
    def __rsub__(self, other: typing.Union["FPTensorABC", float, int]) -> "FPTensorABC":
        ...

    @abc.abstractmethod
    def __mul__(self, other: typing.Union["FPTensorABC", float, int]) -> "FPTensorABC":
        ...

    @abc.abstractmethod
    def __rmul__(self, other: typing.Union["FPTensorABC", float, int]) -> "FPTensorABC":
        ...

    @abc.abstractmethod
    def __matmul__(self, other: "FPTensorABC") -> "FPTensorABC":
        ...

    @abc.abstractmethod
    def __rmatmul__(self, other: "FPTensorABC") -> "FPTensorABC":
        ...


class PHETensorABC(abc.ABC):
    """Tensor implements Partial Homomorphic Encryption schema:
    1. decrypt(encrypt(a) + encrypt(b)) = a + b
    2. decrypt(encrypt(a) * b) = a * b
    """

    @abc.abstractmethod
    def __add__(
        self, other: typing.Union["PHETensorABC", "FPTensorABC", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __radd__(
        self, other: typing.Union["PHETensorABC", "FPTensorABC", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __sub__(
        self, other: typing.Union["PHETensorABC", "FPTensorABC", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __rsub__(
        self, other: typing.Union["PHETensorABC", "FPTensorABC", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __mul__(
        self, other: typing.Union["PHETensorABC", "FPTensorABC", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __rmul__(
        self, other: typing.Union["PHETensorABC", "FPTensorABC", float, int]
    ) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __matmul__(self, other: FPTensorABC) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def __rmatmul__(self, other: FPTensorABC) -> "PHETensorABC":
        ...

    @abc.abstractmethod
    def serialize(self):
        ...

    @abc.abstractmethod
    def T(self) -> "PHETensorABC":
        ...


class PHEEncryptorABC(abc.ABC):
    @abc.abstractmethod
    def encrypt(self, tensor: FPTensorABC) -> PHETensorABC:
        ...


class PHEDecryptorABC(abc.ABC):
    @abc.abstractmethod
    def decrypt(self, tensor: PHETensorABC) -> FPTensorABC:
        ...


class PHECipherABC(abc.ABC):
    @abc.abstractclassmethod
    def keygen(cls, **kwargs) -> typing.Tuple[PHEEncryptorABC, PHEDecryptorABC]:
        ...
