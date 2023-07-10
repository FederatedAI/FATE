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
from typing import Optional

import pydantic

logger = logging.getLogger(__name__)

_NS_FEDERATION_SPLIT = "."


class MetricsKey(pydantic.BaseModel):
    groups: tuple
    namespaces: tuple

    def __hash__(self):
        # the hash is computed as a combination of all the attributes
        return hash((self.groups, self.namespaces))

    def __eq__(self, other):
        if isinstance(other, MetricsKey):
            return (self.groups, self.namespaces) == (other.groups, other.namespaces)
        return False


class NS:
    def __init__(self, name, deep, is_special=False, parent: Optional["NS"] = None) -> None:
        self.name = name
        self.deep = deep
        self.is_special = is_special

        self.parent = parent
        self._federation_tag_cache = None
        self._metrics_keys_cache = None

    def get_federation_tag(self):
        if self._federation_tag_cache is None:
            if self.parent is None:
                self._federation_tag_cache = self.get_name()
            else:
                self._federation_tag_cache = (
                    f"{self.parent.get_federation_tag()}{_NS_FEDERATION_SPLIT}{self.get_name()}"
                )
        return self._federation_tag_cache

    def get_metrics_keys(self):
        if self._metrics_keys_cache is None:
            pre = self.parent.get_metrics_keys() if self.parent is not None else MetricsKey(groups=(), namespaces=())
            if self.is_special:
                self._metrics_keys_cache = MetricsKey(groups=(*pre.groups, self.name), namespaces=pre.namespaces)
            else:
                self._metrics_keys_cache = MetricsKey(groups=pre.groups, namespaces=(*pre.namespaces, self.name))
        return self._metrics_keys_cache

    def get_name(self):
        return self.name

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, deep={self.deep}, special_tags={self.is_special})"

    def indexed_ns(self, index: int):
        return IndexedNS(index=index, name=self.name, deep=self.deep, is_special=self.is_special, parent=self.parent)

    def sub_ns(self, name: str, is_special=False):
        return NS(name=name, deep=self.deep + 1, parent=self, is_special=is_special)


class IndexedNS(NS):
    def __init__(self, index, name: str, deep: int, is_special: bool = False, parent: Optional["NS"] = None) -> None:
        self.index = index
        super().__init__(name=name, deep=deep, is_special=is_special, parent=parent)

    def get_name(self):
        return f"{self.name}-{self.index}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(index={self.index}, name={self.name}, deep={self.deep}, is_special={self.is_special})"


default_ns = NS(name="default", deep=0)

# class Namespace:
#     """
#     Summary, Metrics may be namespace awared:
#     ```
#     namespace = Namespace()
#     ctx = Context(...summary=XXXSummary(namespace))
#     ```
#     """

#     def __init__(self, namespaces=None) -> None:
#         if namespaces is None:
#             namespaces = []
#         self.namespaces = namespaces

#     @contextmanager
#     def into_subnamespace(self, subnamespace: str):
#         self.namespaces.append(subnamespace)
#         try:
#             yield self
#         finally:
#             self.namespaces.pop()

#     @property
#     def namespace(self):
#         return ".".join(self.namespaces)

#     def fedeation_tag(self) -> str:
#         return ".".join(self.namespaces)

#     def sub_namespace(self, namespace):
#         return Namespace([*self.namespaces, namespace])

# @overload
# @contextmanager
# def iter_namespaces(
#     self, start: int, stop: int, *, prefix_name=""
# ) -> Generator[Generator["Namespace", None, None], None, None]:
#     ...

# @overload
# @contextmanager
# def iter_namespaces(
#     self, stop: int, *, prefix_name=""
# ) -> Generator[Generator["Namespace", None, None], None, None]:
#     ...

# @contextmanager
# def iter_namespaces(self, *args, prefix_name=""):
#     assert 0 < len(args) <= 2, "position argument should be 1 or 2"
#     if len(args) == 1:
#         start, stop = 0, args[0]
#     if len(args) == 2:
#         start, stop = args[0], args[1]

#     prev_namespace_state = self._namespace_state

#     def _state_iterator() -> Generator["Namespace", None, None]:
#         for i in range(start, stop):
#             # the tags in the iteration need to be distinguishable
#             template_formated = f"{prefix_name}iter_{i}"
#             self._namespace_state = IterationState(prev_namespace_state.sub_namespace(template_formated))
#             yield self

#     # with context returns iterator of Contexts
#     # namespaec state inside context is changed alone with iterator comsued
#     yield _state_iterator()

#     # restore namespace state when leaving with context
#     self._namespace_state = prev_namespace_state
