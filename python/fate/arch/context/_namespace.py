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
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_NS_FEDERATION_SPLIT = "."


class NS:
    def __init__(self, name, deep, parent: Optional["NS"] = None) -> None:
        self.name = name
        self.deep = deep
        self.parent = parent

        if self.parent is None:
            self._federation_tag = self.get_name()
            self._metric_groups = []
        else:
            self._federation_tag = f"{self.parent._federation_tag}{_NS_FEDERATION_SPLIT}{self.get_name()}"
            self._metric_groups = [*self.parent._metric_groups, self.parent.get_group()]

    @property
    def federation_tag(self):
        return self._federation_tag

    @property
    def metric_groups(self) -> List[Tuple[str, Optional[int]]]:
        return self._metric_groups

    def get_name(self):
        return self.name

    def get_group(self):
        return self.name, None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, deep={self.deep}"

    def indexed_ns(self, index: int):
        return IndexedNS(index=index, name=self.name, deep=self.deep, parent=self.parent)

    def sub_ns(self, name: str):
        return NS(name=name, deep=self.deep + 1, parent=self)


class IndexedNS(NS):
    def __init__(self, index, name: str, deep: int, parent: Optional["NS"] = None) -> None:
        self.index = index
        super().__init__(name=name, deep=deep, parent=parent)

    def get_name(self):
        return f"{self.name}-{self.index}"

    def get_group(self):
        return self.name, self.index

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(index={self.index}, name={self.name}, deep={self.deep})"


default_ns = NS(name="default", deep=0)
