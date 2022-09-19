import typing

import numpy as np
import numpy.typing as npt

class Cipherblock:
    @property
    def shape(self) -> typing.List[int]: ...
    def add_cipherblock(self, other: Cipherblock) -> Cipherblock: ...
    def add_plaintext_f64(self, other: npt.NDArray[np.float64]) -> Cipherblock: ...
    def add_plaintext_f32(self, other: npt.NDArray[np.float32]) -> Cipherblock: ...
    def add_plaintext_i64(self, other: npt.NDArray[np.int64]) -> Cipherblock: ...
    def add_plaintext_i32(self, other: npt.NDArray[np.int32]) -> Cipherblock: ...
    def add_plaintext_scalar_f64(self, other: typing.Union[float, np.float64]) -> Cipherblock: ...
    def add_plaintext_scalar_f32(self, other: typing.Union[float, np.float32]) -> Cipherblock: ...
    def add_plaintext_scalar_i64(self, other: typing.Union[int, np.int64]) -> Cipherblock: ...
    def add_plaintext_scalar_i32(self, other: typing.Union[int, np.int32]) -> Cipherblock: ...

    def sub_cipherblock(self, other: Cipherblock) -> Cipherblock: ...
    def sub_plaintext_f64(self, other: npt.NDArray[np.float64]) -> Cipherblock: ...
    def sub_plaintext_f32(self, other: npt.NDArray[np.float32]) -> Cipherblock: ...
    def sub_plaintext_i64(self, other: npt.NDArray[np.int64]) -> Cipherblock: ...
    def sub_plaintext_i32(self, other: npt.NDArray[np.int32]) -> Cipherblock: ...
    def sub_plaintext_scalar_f64(self, other: typing.Union[float, np.float64]) -> Cipherblock: ...
    def sub_plaintext_scalar_f32(self, other: typing.Union[float, np.float32]) -> Cipherblock: ...
    def sub_plaintext_scalar_i64(self, other: typing.Union[int, np.int64]) -> Cipherblock: ...
    def sub_plaintext_scalar_i32(self, other: typing.Union[int, np.int32]) -> Cipherblock: ...

    def mul_plaintext_f64(self, other: npt.NDArray[np.float64]) -> Cipherblock: ...
    def mul_plaintext_f32(self, other: npt.NDArray[np.float32]) -> Cipherblock: ...
    def mul_plaintext_i64(self, other: npt.NDArray[np.int64]) -> Cipherblock: ...
    def mul_plaintext_i32(self, other: npt.NDArray[np.int32]) -> Cipherblock: ...
    def mul_plaintext_scalar_f64(self, other: typing.Union[float, np.float64]) -> Cipherblock: ...
    def mul_plaintext_scalar_f32(self, other: typing.Union[float, np.float32]) -> Cipherblock: ...
    def mul_plaintext_scalar_i64(self, other: typing.Union[int, np.int64]) -> Cipherblock: ...
    def mul_plaintext_scalar_i32(self, other: typing.Union[int, np.int32]) -> Cipherblock: ...

    def matmul_plaintext_ix2_f64(self, other: npt.NDArray[np.float64]) -> Cipherblock: ...
    def matmul_plaintext_ix2_f32(self, other: npt.NDArray[np.float32]) -> Cipherblock: ...
    def matmul_plaintext_ix2_i64(self, other: npt.NDArray[np.int64]) -> Cipherblock: ...
    def matmul_plaintext_ix2_i32(self, other: npt.NDArray[np.int32]) -> Cipherblock: ...
    def matmul_plaintext_ix1_f64(self, other: npt.NDArray[np.float64]) -> Cipherblock: ...
    def matmul_plaintext_ix1_f32(self, other: npt.NDArray[np.float32]) -> Cipherblock: ...
    def matmul_plaintext_ix1_i64(self, other: npt.NDArray[np.int64]) -> Cipherblock: ...
    def matmul_plaintext_ix1_i32(self, other: npt.NDArray[np.int32]) -> Cipherblock: ...
    def rmatmul_plaintext_ix2_f64(self, other: npt.NDArray[np.float64]) -> Cipherblock: ...
    def rmatmul_plaintext_ix2_f32(self, other: npt.NDArray[np.float32]) -> Cipherblock: ...
    def rmatmul_plaintext_ix2_i64(self, other: npt.NDArray[np.int64]) -> Cipherblock: ...
    def rmatmul_plaintext_ix2_i32(self, other: npt.NDArray[np.int32]) -> Cipherblock: ...
    def rmatmul_plaintext_ix1_f64(self, other: npt.NDArray[np.float64]) -> Cipherblock: ...
    def rmatmul_plaintext_ix1_f32(self, other: npt.NDArray[np.float32]) -> Cipherblock: ...
    def rmatmul_plaintext_ix1_i64(self, other: npt.NDArray[np.int64]) -> Cipherblock: ...
    def rmatmul_plaintext_ix1_i32(self, other: npt.NDArray[np.int32]) -> Cipherblock: ...

    def sum(self) -> Cipherblock: ...
    def mean(self) -> Cipherblock: ...


class PK:
    def encrypt_f64(self, a: npt.NDArray[np.float64]) -> Cipherblock: ...
    def encrypt_f32(self, a: npt.NDArray[np.float32]) -> Cipherblock: ...
    def encrypt_i64(self, a: npt.NDArray[np.int64]) -> Cipherblock: ...
    def encrypt_i32(self, a: npt.NDArray[np.int32]) -> Cipherblock: ...

class SK:
    def decrypt_f64(self, a: Cipherblock) -> npt.NDArray[np.float64]: ...
    def decrypt_f32(self, a: Cipherblock) -> npt.NDArray[np.float32]: ...
    def decrypt_i64(self, a: Cipherblock) -> npt.NDArray[np.int64]: ...
    def decrypt_i32(self, a: Cipherblock) -> npt.NDArray[np.int32]: ...

def keygen(bit_size: int) -> typing.Tuple[PK, SK]:...
