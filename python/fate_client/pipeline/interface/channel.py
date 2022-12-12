import abc
from typing import Dict, List, Optional, Union


class ArtifactChannel(abc.ABC):
    def __init__(
            self,
            name: str,
            channel_type: Union[str, Dict],
            task_name: Optional[str] = None,
            source: str = "task_output_artifact"
    ):
        self.name = name
        self.channel_type = channel_type
        self.task_name = task_name or None
        self.source = source

    def __str__(self):
        return "{" + f"channel:task={self.task_name};" \
                     f"name={self.name};" \
                     f"type={self.channel_type};" \
                     f"source={self.source};" + "}"

    def __repr__(self):
        return str(self)
