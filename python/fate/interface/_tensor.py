from typing import Any, List, Protocol, Union, overload


class FPTensor:
    shape: List[int]
    T: "FPTensor"

    def __add__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        ...

    def __radd__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        ...

    def __sub__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        ...

    def __rsub__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        ...

    def __mul__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        ...

    def __rmul__(self, other: Union["FPTensor", float, int]) -> "FPTensor":
        ...

    def __matmul__(self, other: "FPTensor") -> "FPTensor":
        ...

    def __rmatmul__(self, other: "FPTensor") -> "FPTensor":
        ...


class PHETensor(Protocol):
    shape: List[int]
    T: "FPTensor"

    def __add__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        ...

    def __radd__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        ...

    def __sub__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        ...

    def __rsub__(self, other: Union["PHETensor", FPTensor, int, float]) -> "PHETensor":
        ...

    def __mul__(self, other: Union[FPTensor, int, float]) -> "PHETensor":
        ...

    def __rmul__(self, other: Union[FPTensor, int, float]) -> "PHETensor":
        ...

    def __matmul__(self, other: FPTensor) -> "PHETensor":
        ...

    def __rmatmul__(self, other: FPTensor) -> "PHETensor":
        ...

    @overload
    def decrypt(self, decryptor: "PHEDecryptor") -> FPTensor:
        ...

    @overload
    def decrypt(self, decryptor) -> Any:
        ...

    def decrypt(self, decryptor):
        return decryptor.decrypt(self)


class PHEEncryptor:
    def encrypt(self, tensor: "FPTensor") -> "PHETensor":
        ...


class PHEDecryptor:
    def decrypt(self, tensor: "PHETensor") -> "FPTensor":
        ...
