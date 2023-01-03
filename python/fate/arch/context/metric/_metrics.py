from typing import Dict, Optional

from ._type import Metrics


class ROCMetrics(Metrics):
    type = "roc"

    def __init__(self, name, data) -> None:
        self.name = name
        self.data = data
        self.nemaspace: Optional[str] = None
        self.groups: Dict[str, str] = {}

    def dict(self) -> dict:
        return dict(
            name=self.name,
            namespace=self.nemaspace,
            groups=self.groups,
            type=self.type,
            metadata={},
            data=self.data,
        )
