import enum
from typing import List, TypeVar

from typing_extensions import Annotated


class OutputAnnotated:
    ...


class InputAnnotated:
    ...


T = TypeVar("T")
Output = Annotated[T, OutputAnnotated]
Input = Annotated[T, InputAnnotated]


class roles(str, enum.Enum):
    GUEST = "guest"
    HOST = "host"
    ARBITER = "arbiter"

    @classmethod
    def get_all(cls) -> List["roles"]:
        return [roles.GUEST, roles.HOST, roles.ARBITER]


class stages(str, enum.Enum):
    TRAIN = "train"
    PREDICT = "predict"


class labels(str, enum.Enum):
    TRAINABLE = "trainable"
