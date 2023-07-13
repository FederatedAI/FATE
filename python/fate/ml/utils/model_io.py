from typing import Optional


class ModelIO:
    _META = "meta"
    _DATA = "data"

    def __init__(self, data: dict, meta: Optional[dict] = None):
        self.data = data
        self.meta = meta

    def dict(self):
        return {
            self._DATA: self.data,
            self._META: self.meta if self.meta is not None else {},
        }

    @classmethod
    def from_dict(cls, d: dict):
        data = d[cls._DATA]
        if cls._META in d:
            meta = d[cls._META]
        else:
            meta = None
        return cls(data, meta)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.data}, meta={self.meta})"
