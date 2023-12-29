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
from typing import Iterable, Literal, Optional, Tuple, TypeVar, overload

from fate.arch.trace import auto_trace
from ._cipher import CipherKit
from ._metrics import InMemoryMetricsHandler, MetricsWrap
from ._namespace import NS, default_ns
from ._parties import Parties, Party
from ..unify import device as device_type

logger = logging.getLogger(__name__)

T = TypeVar("T")

if typing.TYPE_CHECKING:
    from fate.arch.federation.api import Federation
    from fate.arch.computing.api import KVTableContext


class Context:
    """
    Note: most parameters has default dummy value,
          which is convenient when used in script.
          please pass in custom implements as you wish
    """

    def __init__(
        self,
        device: device_type = device_type.CPU,
        computing: Optional["KVTableContext"] = None,
        federation: Optional["Federation"] = None,
        metrics_handler: Optional = None,
        namespace: Optional[NS] = None,
        cipher: Optional[CipherKit] = None,
    ) -> None:
        self._device = device
        self._computing = computing
        self._federation = federation
        self._metrics_handler = metrics_handler
        self._namespace = namespace
        self._cipher = cipher

        if self._namespace is None:
            self._namespace = default_ns
        if self._cipher is None:
            self._cipher: CipherKit = CipherKit(device)
        self._cipher.set_ctx(self)

        self._role_to_parties = None
        self._is_destroyed = False

        self._mpc = None

    @property
    def mpc(self):
        from ._mpc import MPC

        if self._mpc is None:
            self._mpc = MPC(self)

        return self._mpc

    @property
    def device(self):
        return self._device

    @property
    def namespace(self):
        return self._namespace

    @property
    def cipher(self):
        return self._cipher

    def set_cipher(self, cipher_mapping):
        self._cipher = CipherKit(self._device, {"phe": {self._device: cipher_mapping["phe"]}})

    def set_metric_handler(self, metrics_handler):
        self._metrics_handler = metrics_handler

    @property
    def metrics(self):
        if self._metrics_handler is None:
            self._metrics_handler = InMemoryMetricsHandler()
        return MetricsWrap(self._metrics_handler, self._namespace)

    def with_namespace(self, namespace: NS):
        return Context(
            device=self._device,
            computing=self._computing,
            federation=self._federation,
            metrics_handler=self._metrics_handler,
            namespace=namespace,
            cipher=self._cipher,
        )

    @property
    def computing(self) -> "KVTableContext":
        return self._get_computing()

    @property
    def federation(self) -> "Federation":
        return self._get_federation()

    def sub_ctx(self, name: str) -> "Context":
        return self.with_namespace(self._namespace.sub_ns(name=name))

    def indexed_ctx(self, index: int) -> "Context":
        return self.with_namespace(self._namespace.indexed_ns(index))

    @property
    def on_iterations(self) -> "Context":
        return self.sub_ctx("iterations")

    @property
    def on_batches(self) -> "Context":
        return self.sub_ctx("batches")

    @property
    def on_cross_validations(self) -> "Context":
        return self.sub_ctx("cross_validations")

    @overload
    def ctxs_range(self, end: int) -> Iterable[Tuple[int, "Context"]]:
        ...

    @overload
    def ctxs_range(self, start: int, end: int) -> Iterable[Tuple[int, "Context"]]:
        ...

    def ctxs_range(self, *args, **kwargs) -> Iterable[Tuple[int, "Context"]]:
        """
        create contexes with namespaces indexed from 0 to end(excluded)
        """

        if "start" in kwargs:
            start = kwargs["start"]
            if "end" not in kwargs:
                raise ValueError("End value must be provided")
            end = kwargs["end"]
            if len(args) > 0:
                raise ValueError("Too many arguments")
        else:
            if "end" in kwargs:
                end = kwargs["end"]
                if len(args) > 1:
                    raise ValueError("Too many arguments")
                elif len(args) == 0:
                    raise ValueError("Start value must be provided")
                else:
                    start = args[0]
            else:
                if len(args) == 1:
                    start, end = 0, args[0]
                elif len(args) == 2:
                    start, end = args
                else:
                    raise ValueError("Too few arguments")

        for i in range(start, end):
            yield i, self.with_namespace(self._namespace.indexed_ns(index=i))

    def ctxs_zip(self, iterable: Iterable[T]) -> Iterable[Tuple["Context", T]]:
        """
        zip contexts with iterable with namespaces indexed from 0
        """
        for i, it in enumerate(iterable):
            yield self.with_namespace(self._namespace.indexed_ns(index=i)), it

    def set_federation(self, federation: "Federation"):
        self._federation = federation

    @property
    def guest(self) -> Party:
        return self._get_parties("guest")[0]

    @property
    def hosts(self) -> Parties:
        return self._get_parties("host")

    @property
    def arbiter(self) -> Party:
        return self._get_parties("arbiter")[0]

    @property
    def rank(self):
        return self.local.rank

    @property
    def local(self):
        role, party_id = self._get_federation().local_party
        for party in self._get_parties(role):
            if party.party[1] == party_id:
                return party
        raise RuntimeError("local party not found")

    @property
    def is_on_guest(self):
        return self._federation.local_party[0] == "guest"

    @property
    def is_on_host(self):
        return self._federation.local_party[0] == "host"

    @property
    def is_on_arbiter(self):
        return self._federation.local_party[0] == "arbiter"

    @property
    def parties(self) -> Parties:
        return self._get_parties()

    @property
    def world_size(self):
        return self._get_federation().world_size

    def _get_parties(self, role: Optional[Literal["guest", "host", "arbiter", "local"]] = None) -> Parties:
        # update role to parties mapping
        if self._role_to_parties is None:
            self._role_to_parties = {}
            for i, party in enumerate(self._get_federation().parties):
                self._role_to_parties.setdefault(party[0], []).append((i, party))

        parties = []
        if role is None:
            for role_parties in self._role_to_parties.values():
                parties.extend(role_parties)
        else:
            if role not in self._role_to_parties:
                raise RuntimeError(f"no {role} party has configured")
            else:
                parties.extend(self._role_to_parties[role])
        parties.sort(key=lambda x: x[0])
        return Parties(
            self,
            self._get_federation(),
            self._get_computing(),
            parties,
            self._namespace,
        )

    def _get_federation(self):
        if self._federation is None:
            raise RuntimeError(f"federation not set")
        return self._federation

    def _get_computing(self) -> "KVTableContext":
        if self._computing is None:
            raise RuntimeError(f"computing not set")
        return self._computing

    @auto_trace
    def destroy(self):
        if not self._is_destroyed:
            try:
                self.federation.destroy()
                logger.debug("federation engine destroy done")
            except Exception as e:
                logger.exception(f"federation engine destroy failed: {e}")

            try:
                self.computing.destroy()
                logger.debug("computing engine destroy done")
            except Exception as e:
                logger.exception(f"computing engine destroy failed: {e}")
            self._is_destroyed = True
