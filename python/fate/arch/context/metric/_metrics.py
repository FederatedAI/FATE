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
