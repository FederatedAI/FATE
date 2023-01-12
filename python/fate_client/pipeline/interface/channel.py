#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
