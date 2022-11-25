import abc
from typing import Dict, List, Optional, Union


class ArtifactChannel(abc.ABC):
    def __init__(
            self,
            name: str,
            channel_type: Union[str, Dict],
            task_name: Optional[str] = None,
    ):
        self.name = name
        self.channel_type = channel_type
        self.task_name = task_name or None

    def __str__(self):
        return "{" + f"channel:task={self.task_name};name={self.name};type={self.channel_type};" + "}"

    def __repr__(self):
        return str(self)
