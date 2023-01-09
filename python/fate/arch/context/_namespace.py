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
from typing import Generator, overload

logger = logging.getLogger(__name__)


class Namespace:
    """
    Summary, Metrics may be namespace awared:
    ```
    namespace = Namespace()
    ctx = Context(...summary=XXXSummary(namespace))
    ```
    """

    def __init__(self, namespaces=None) -> None:
        if namespaces is None:
            namespaces = []
        self.namespaces = namespaces

    @contextmanager
    def into_subnamespace(self, subnamespace: str):
        self.namespaces.append(subnamespace)
        try:
            yield self
        finally:
            self.namespaces.pop()

    @property
    def namespace(self):
        return ".".join(self.namespaces)

    def fedeation_tag(self) -> str:
        return ".".join(self.namespaces)

    def sub_namespace(self, namespace):
        return Namespace([*self.namespaces, namespace])

    @overload
    @contextmanager
    def iter_namespaces(
        self, start: int, stop: int, *, prefix_name=""
    ) -> Generator[Generator["Namespace", None, None], None, None]:
        ...

    @overload
    @contextmanager
    def iter_namespaces(
        self, stop: int, *, prefix_name=""
    ) -> Generator[Generator["Namespace", None, None], None, None]:
        ...

    @contextmanager
    def iter_namespaces(self, *args, prefix_name=""):
        assert 0 < len(args) <= 2, "position argument should be 1 or 2"
        if len(args) == 1:
            start, stop = 0, args[0]
        if len(args) == 2:
            start, stop = args[0], args[1]

        prev_namespace_state = self._namespace_state

        def _state_iterator() -> Generator["Namespace", None, None]:
            for i in range(start, stop):
                # the tags in the iteration need to be distinguishable
                template_formated = f"{prefix_name}iter_{i}"
                self._namespace_state = IterationState(prev_namespace_state.sub_namespace(template_formated))
                yield self

        # with context returns iterator of Contexts
        # namespaec state inside context is changed alone with iterator comsued
        yield _state_iterator()

        # restore namespace state when leaving with context
        self._namespace_state = prev_namespace_state
