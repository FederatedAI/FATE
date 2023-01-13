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
from copy import copy
from typing import Iterator, List, Optional

from fate.interface import T_ROLE, ComputingEngine
from fate.interface import Context as ContextInterface
from fate.interface import FederationEngine, MetricsHandler, PartyMeta

from ..unify import device
from ._cipher import CipherKit
from ._federation import GC, Parties, Party
from ._namespace import Namespace
from .io.kit import IOKit
from .metric import MetricsWrap

logger = logging.getLogger(__name__)


class Context(ContextInterface):
    """
    implement fate.interface.ContextInterface

    Note: most parameters has default dummy value,
          which is convenient when used in script.
          please pass in custom implements as you wish
    """

    def __init__(
        self,
        context_name: Optional[str] = None,
        device: device = device.CPU,
        computing: Optional[ComputingEngine] = None,
        federation: Optional[FederationEngine] = None,
        metrics_handler: Optional[MetricsHandler] = None,
        namespace: Optional[Namespace] = None,
    ) -> None:
        self.context_name = context_name
        self.metrics = MetricsWrap(metrics_handler)

        if namespace is None:
            namespace = Namespace()
        self.namespace = namespace
        self.super_namespace = Namespace()

        self.cipher: CipherKit = CipherKit(device)
        self._io_kit: IOKit = IOKit()

        self._computing = computing
        self._federation = federation
        self._role_to_parties = None

        self._gc = GC()

    def with_namespace(self, namespace: Namespace):
        context = copy(self)
        context.namespace = namespace
        return context

    def into_group_namespace(self, group_name: str, group_id: str):
        context = copy(self)
        context.metrics = context.metrics.into_group(group_name, group_id)
        context.namespace = self.namespace.sub_namespace(f"{group_name}_{group_id}")
        return context

    def range(self, end):
        for i in range(end):
            yield i, self.with_namespace(self.namespace.sub_namespace(f"{i}"))

    def iter(self, iterable):
        for i, it in enumerate(iterable):
            yield self.with_namespace(self.namespace.sub_namespace(f"{i}")), it

    @property
    def computing(self):
        return self._get_computing()

    @property
    def federation(self) -> FederationEngine:
        return self._get_federation()

    @contextmanager
    def sub_ctx(self, namespace: str) -> Iterator["Context"]:
        try:
            yield self.with_namespace(self.namespace.sub_namespace(namespace))
        finally:
            ...

    def set_federation(self, federation: FederationEngine):
        self._federation = federation

    @property
    def guest(self) -> Party:
        return Party(
            self._get_federation(),
            self._get_parties("guest")[0],
            self.namespace,
        )

    @property
    def hosts(self) -> Parties:
        return Parties(
            self._get_federation(),
            self._get_federation().local_party,
            self._get_parties("host"),
            self.namespace,
        )

    @property
    def arbiter(self) -> Party:
        return Party(
            self._get_federation(),
            self._get_parties("arbiter")[0],
            self.namespace,
        )

    @property
    def local(self):
        return self._get_federation().local_party

    @property
    def parties(self) -> Parties:
        return Parties(
            self._get_federation(),
            self._get_federation().local_party,
            self._get_parties(),
            self.namespace,
        )

    def _get_parties(self, role: Optional[T_ROLE] = None) -> List[PartyMeta]:
        # update role to parties mapping
        if self._role_to_parties is None:
            self._role_to_parties = {}
            for party in self._get_federation().parties:
                self._role_to_parties.setdefault(party[0], []).append(party)

        parties = []
        if role is None:
            for role_parties in self._role_to_parties.values():
                parties.extend(role_parties)
        else:
            if role not in self._role_to_parties:
                raise RuntimeError(f"no {role} party has configurated")
            else:
                parties.extend(self._role_to_parties[role])
        return parties

    def _get_federation(self):
        if self._federation is None:
            raise RuntimeError(f"federation not set")
        return self._federation

    def _get_computing(self):
        if self._computing is None:
            raise RuntimeError(f"computing not set")
        return self._computing

    def reader(self, uri, **kwargs):
        return self._io_kit.reader(self, uri, **kwargs)

    def writer(self, uri, **kwargs):
        return self._io_kit.writer(self, uri, **kwargs)

    def destroy(self):
        try:
            self.computing.destroy()
        except:
            logger.exception("computing engine close failed", stack_info=True)

        try:
            self.federation.destroy()
        except:
            logger.exception("computing engine close failed", stack_info=True)
