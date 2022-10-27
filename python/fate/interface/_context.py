from contextlib import contextmanager
from typing import Iterable, Iterator, Protocol, Tuple, TypeVar

from ._anonymous import Anonymous
from ._cache import Cache
from ._checkpoint import CheckpointManager
from ._cipher import CipherKit
from ._computing import ComputingEngine
from ._federation import FederationEngine
from ._log import Logger
from ._metric import Metrics
from ._party import Parties, Party
from ._summary import Summary

T = TypeVar("T")


class Context(Protocol):
    summary: Summary
    metrics: Metrics
    cache: Cache
    anonymous_generator: Anonymous
    checkpoint_manager: CheckpointManager
    log: Logger
    guest: Party
    hosts: Parties
    arbiter: Party
    parties: Parties
    cipher: CipherKit
    computing: ComputingEngine
    federation: FederationEngine

    @contextmanager
    def sub_ctx(self, namespace) -> Iterator["Context"]:
        ...

    def range(self, end) -> Iterator[Tuple[int, "Context"]]:
        ...

    def iter(self, iterable: Iterable[T]) -> Iterator[Tuple["Context", T]]:
        ...
