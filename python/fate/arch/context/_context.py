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
import typing
from typing import Iterable, List, Literal, Optional, Tuple, TypeVar

from fate.interface import PartyMeta

from ..unify import device
from ._cipher import CipherKit
from ._federation import Parties, Party
from ._metrics import MetricsWrap, NoopMetricsHandler
from ._namespace import NS, default_ns

if typing.TYPE_CHECKING:
    from fate.interface import CSessionABC, FederationEngine

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Context:
    """
    implement fate.interface.ContextInterface

    Note: most parameters has default dummy value,
          which is convenient when used in script.
          please pass in custom implements as you wish
    """

    def __init__(
        self,
        device: device = device.CPU,
        computing: Optional["CSessionABC"] = None,
        federation: Optional["FederationEngine"] = None,
        metrics_handler: Optional = None,
        namespace: Optional[NS] = None,
        cipher: Optional[CipherKit] = None,
    ) -> None:
        self._device = device
        self._computing = computing
        self._federation = federation
        self._metrics_handler = metrics_handler
        self.namespace = namespace
        self.cipher = cipher

        if self._metrics_handler is None:
            self._metrics_handler = NoopMetricsHandler()
        if self.namespace is None:
            self.namespace = default_ns
        if self.cipher is None:
            self.cipher: CipherKit = CipherKit(device)

        self._role_to_parties = None
        self._is_destroyed = False

    @property
    def metrics(self):
        return MetricsWrap(self._metrics_handler, self.namespace)

    def with_namespace(self, namespace: NS):
        return Context(
            device=self._device,
            computing=self._computing,
            federation=self._federation,
            metrics_handler=self._metrics_handler,
            namespace=namespace,
            cipher=self.cipher,
        )

    @property
    def computing(self):
        return self._get_computing()

    @property
    def federation(self) -> "FederationEngine":
        return self._get_federation()

    def sub_ctx(self, name: str) -> "Context":
        return self.with_namespace(self.namespace.sub_ns(name=name))

    @property
    def on_iterations(self) -> "Context":
        return self.sub_ctx("iterations")

    @property
    def on_batches(self) -> "Context":
        return self.sub_ctx("iterations")

    @property
    def on_cross_validations(self) -> "Context":
        return self.sub_ctx("cross_validations")

    def ctxs_range(self, end: int) -> Iterable[Tuple[int, "Context"]]:
        """
        create contexes with namespaces indexed from 0 to end(excluded)
        """
        for i in range(end):
            yield i, self.with_namespace(self.namespace.indexed_ns(index=i))

    def ctxs_zip(self, iterable: Iterable[T]) -> Iterable[Tuple["Context", T]]:
        """
        zip contexts with iterable with namespaces indexed from 0
        """
        for i, it in enumerate(iterable):
            yield self.with_namespace(self.namespace.indexed_ns(index=i)), it

    def set_federation(self, federation: "FederationEngine"):
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
    def is_on_guest(self):
        return self.local[0] == "guest"

    @property
    def is_on_host(self):
        return self.local[0] == "host"

    @property
    def is_on_arbiter(self):
        return self.local[0] == "arbiter"

    @property
    def parties(self) -> Parties:
        return Parties(
            self._get_federation(),
            self._get_federation().local_party,
            self._get_parties(),
            self.namespace,
        )

    def _get_parties(self, role: Optional[Literal["guest", "host", "arbiter"]] = None) -> List[PartyMeta]:
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

    def destroy(self):
        if not self._is_destroyed:
            try:
                self.federation.destroy()
                logger.debug("federation engine destroy done")
            except:
                logger.exception("federation engine destroy failed", stack_info=True)

            try:
                self.computing.destroy()
                logger.debug("computing engine destroy done")
            except:
                logger.exception("computing engine destroy failed", stack_info=True)
            self._is_destroyed = True
