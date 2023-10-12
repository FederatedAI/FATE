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
import logging
from typing import List, Tuple


LOGGER = logging.getLogger(__name__)


class Federation:
    def _push(
        self,
        v,
        name: str,
        tag: str,
        parties: List[Tuple[str, str]],
    ):
        ...

    def push(
        self,
        v,
        name: str,
        tag: str,
        parties: List[Tuple[str, str]],
    ):
        self._push(
            v=v,
            name=name,
            tag=tag,
            parties=parties,
        )

    def _pull(
        self,
        name: str,
        tag: str,
        parties: List[Tuple[str, str]],
    ) -> List:
        raise NotImplementedError("pull is not supported in standalone federation")

    def pull(
        self,
        name: str,
        tag: str,
        parties: List[Tuple[str, str]],
    ) -> List:
        return self._pull(
            name=name,
            tag=tag,
            parties=parties,
        )
