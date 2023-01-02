from typing import Union

from ._dstorage import DStorage
from ._dtype import dtype
from ._lstorage import LStorage
from ._shape import DAxis, Shape

Storage = Union[LStorage, DStorage]
__all__ = ["dtype", "Shape", "DAxis", "LStorage", "DStorage", "Storage"]
