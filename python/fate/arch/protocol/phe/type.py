from typing import Generic, TypeVar

EV = TypeVar("EV")
V = TypeVar("V")
PK = TypeVar("PK")
Coder = TypeVar("Coder")


class TensorEvaluator(Generic[EV, V, PK, Coder]):
    @staticmethod
    def add(a: EV, b: EV, pk: PK) -> EV:
        ...

    @staticmethod
    def add_plain(a: EV, b: V, pk: PK, coder: Coder, output_dtype=None) -> EV:
        ...

    @staticmethod
    def add_plain_scalar(a: EV, b, pk: PK, coder: Coder, output_dtype) -> EV:
        encoded = coder.encode(b, dtype=output_dtype)
        encrypted = pk.encrypt_encoded_scalar(encoded, obfuscate=False)
        return a.add_scalar(pk.pk, encrypted)
