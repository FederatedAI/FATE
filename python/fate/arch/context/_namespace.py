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
from contextlib import contextmanager
from typing import Generator, List, Literal, Optional, Set, overload

logger = logging.getLogger(__name__)

_NS_FEDERATION_SPLIT = "."


class NS:
    def __init__(self, name, deep, special_tags: Optional[Set] = None, parent: Optional["NS"] = None) -> None:
        self.name = name
        self.deep = deep
        self.special_tags = special_tags

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
            current_keys = dict(name=self.name, special_tags=self.special_tags)
            if self.parent is None:
                self._metrics_keys_cache = [current_keys]
            else:
                self._metrics_keys_cache = [*self.parent.get_metrics_keys(), current_keys]
        return self._metrics_keys_cache

    def get_name(self):
        return self.name

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, deep={self.deep}, special_tags={self.special_tags})"

    def indexed_ns(self, index: int):
        return IndexedNS(
            index=index, name=self.name, deep=self.deep, special_tags=self.special_tags, parent=self.parent
        )

    def sub_ns(self, name: str):
        return NS(name=name, deep=self.deep + 1, parent=self)


class IndexedNS(NS):
    def __init__(
        self, index, name: str, deep: int, special_tags: Optional[Set] = None, parent: Optional["NS"] = None
    ) -> None:
        self.index = index
        super().__init__(name=name, deep=deep, special_tags=special_tags, parent=parent)

    def get_name(self):
        return f"{self.name}-{self.index}"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(index={self.index}, name={self.name}, deep={self.deep}, special_tags={self.special_tags})"


default_ns = NS(name="default", deep=0, special_tags={"Default"})


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
