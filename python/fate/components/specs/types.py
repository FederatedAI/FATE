from typing import TypeVar

from typing_extensions import Annotated


class OutputAnnotated:
    ...


class InputAnnotated:
    ...


T = TypeVar("T")
Output = Annotated[T, OutputAnnotated]
Input = Annotated[T, InputAnnotated]
