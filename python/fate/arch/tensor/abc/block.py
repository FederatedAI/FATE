import abc
import typing


class FPBlockABC:
    @classmethod
    def zeors(cls, shape) -> "FPBlockABC":
        ...

    @abc.abstractmethod
    def __add__(self, other: typing.Union["FPBlockABC", float, int]) -> "FPBlockABC":
        ...

    @abc.abstractmethod
    def __radd__(self, other: typing.Union["FPBlockABC", float, int]) -> "FPBlockABC":
        ...

    @abc.abstractmethod
    def __sub__(self, other: typing.Union["FPBlockABC", float, int]) -> "FPBlockABC":
        ...

    @abc.abstractmethod
    def __rsub__(self, other: typing.Union["FPBlockABC", float, int]) -> "FPBlockABC":
        ...

    @abc.abstractmethod
    def __mul__(self, other: typing.Union["FPBlockABC", float, int]) -> "FPBlockABC":
        ...

    @abc.abstractmethod
    def __rmul__(self, other: typing.Union["FPBlockABC", float, int]) -> "FPBlockABC":
        ...

    @abc.abstractmethod
    def __matmul__(self, other: "FPBlockABC") -> "FPBlockABC":
        ...

    @abc.abstractmethod
    def __rmatmul__(self, other: "FPBlockABC") -> "FPBlockABC":
        ...


class PHEBlockABC:
    """Tensor implements Partial Homomorphic Encryption schema:
    1. decrypt(encrypt(a) + encrypt(b)) = a + b
    2. decrypt(encrypt(a) * b) = a * b
    """

    @abc.abstractmethod
    def __add__(
        self, other: typing.Union["PHEBlockABC", "FPBlockABC", float, int]
    ) -> "PHEBlockABC":
        ...

    @abc.abstractmethod
    def __radd__(
        self, other: typing.Union["PHEBlockABC", "FPBlockABC", float, int]
    ) -> "PHEBlockABC":
        ...

    @abc.abstractmethod
    def __sub__(
        self, other: typing.Union["PHEBlockABC", "FPBlockABC", float, int]
    ) -> "PHEBlockABC":
        ...

    @abc.abstractmethod
    def __rsub__(
        self, other: typing.Union["PHEBlockABC", "FPBlockABC", float, int]
    ) -> "PHEBlockABC":
        ...

    @abc.abstractmethod
    def __mul__(
        self, other: typing.Union["PHEBlockABC", "FPBlockABC", float, int]
    ) -> "PHEBlockABC":
        ...

    @abc.abstractmethod
    def __rmul__(
        self, other: typing.Union["PHEBlockABC", "FPBlockABC", float, int]
    ) -> "PHEBlockABC":
        ...

    @abc.abstractmethod
    def __matmul__(self, other: FPBlockABC) -> "PHEBlockABC":
        ...

    @abc.abstractmethod
    def __rmatmul__(self, other: FPBlockABC) -> "PHEBlockABC":
        ...

    @abc.abstractmethod
    def serialize(self):
        ...

    # @abc.abstractmethod
    def T(self) -> "PHEBlockABC":
        ...


class PHEBlockEncryptorABC:
    @abc.abstractmethod
    def encrypt(self, tensor: FPBlockABC) -> PHEBlockABC:
        ...


class PHEBlockDecryptorABC:
    @abc.abstractmethod
    def decrypt(self, tensor: PHEBlockABC) -> FPBlockABC:
        ...


class PHEBlockCipherABC:
    @abc.abstractclassmethod
    def keygen(
        cls, **kwargs
    ) -> typing.Tuple[PHEBlockEncryptorABC, PHEBlockDecryptorABC]:
        ...
