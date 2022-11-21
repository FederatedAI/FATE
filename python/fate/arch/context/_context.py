import logging
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import Iterator, List, Optional

from fate.interface import (
    LOGMSG,
    T_ROLE,
    Anonymous,
    Cache,
    CheckpointManager,
    ComputingEngine,
)
from fate.interface import Context as ContextInterface
from fate.interface import FederationEngine
from fate.interface import Logger as LoggerInterface
from fate.interface import Metric as MetricInterface
from fate.interface import MetricMeta as MetricMetaInterface
from fate.interface import Metrics, PartyMeta, Summary

from ..unify import Backend, device
from ._cipher import CipherKit
from ._federation import GC, Parties, Party
from ._io import IOKit
from ._namespace import Namespace
from ._tensor import TensorKit


@dataclass
class Metric(MetricInterface):
    key: str
    value: float
    timestamp: Optional[float] = None


class MetricMeta(MetricMetaInterface):
    def __init__(self, name: str, metric_type: str, extra_metas: Optional[dict] = None):
        self.name = name
        self.metric_type = metric_type
        self.metas = {}
        self.extra_metas = extra_metas

    def update_metas(self, metas: dict):
        self.metas.update(metas)


class DummySummary(Summary):
    """
    dummy summary save nowhre
    """

    def __init__(self) -> None:
        self._summary = {}

    @property
    def summary(self):
        return self._summary

    def save(self):
        pass

    def reset(self, summary: dict):
        self._summary = summary

    def add(self, key: str, value):
        self._summary[key] = value


class DummyMetrics(Metrics):
    def __init__(self) -> None:
        self._data = []
        self._meta = []

    def log(self, name: str, namespace: str, data: List[Metric]):
        self._data.append((name, namespace, data))

    def log_meta(self, name: str, namespace: str, meta: MetricMeta):
        self._meta.append((name, namespace, meta))

    def log_warmstart_init_iter(self, iter_num):  # FIXME: strange here
        ...


class DummyCache(Cache):
    def __init__(self) -> None:
        self.cache = []

    def add_cache(self, key, value):
        self.cache.append((key, value))


class DummyCheckpointManager(CheckpointManager):
    ...


class ContextLogger(LoggerInterface):
    def __init__(
        self,
        context_name: Optional[str] = None,
        namespace: Optional[Namespace] = None,
        level=logging.DEBUG,
    ) -> None:
        self.logger = getLogger("fate.arch.context")
        self.namespace = namespace
        self.context_name = context_name

        self.logger.setLevel(level)

        formats = []
        if self.context_name is not None:
            formats.append("%(context_name)s")
        if self.namespace is not None:
            formats.append("%(namespace)s")
        formats.append("%(pathname)s:%(lineno)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter(" - ".join(formats))

        # console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log(self, level: int, msg: LOGMSG):
        if Logger.isEnabledFor(self.logger, level):
            if callable(msg):
                msg = msg()
            extra = {}
            if self.namespace is not None:
                extra["namespace"] = self.namespace.namespace
            if self.context_name is not None:
                extra["context_name"] = self.context_name
            self.logger.log(level, msg, stacklevel=3, extra=extra)

    def info(self, msg: LOGMSG):
        return self.log(logging.INFO, msg)

    def debug(self, msg: LOGMSG):
        return self.log(logging.DEBUG, msg)

    def error(self, msg: LOGMSG):
        return self.log(logging.ERROR, msg)

    def warning(self, msg: LOGMSG):
        return self.log(logging.WARNING, msg)


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
        summary: Summary = DummySummary(),
        metrics: Metrics = DummyMetrics(),
        cache: Cache = DummyCache(),
        checkpoint_manager: CheckpointManager = DummyCheckpointManager(),
        log: Optional[LoggerInterface] = None,
        namespace: Optional[Namespace] = None,
    ) -> None:
        self.context_name = context_name
        self.summary = summary
        self.metrics = metrics
        self.cache = cache
        self.checkpoint_manager = checkpoint_manager

        if namespace is None:
            namespace = Namespace()
        self.namespace = namespace

        if log is None:
            log = ContextLogger(context_name, self.namespace)
        self.log = log
        self.cipher: CipherKit = CipherKit(device)
        self.tensor: TensorKit = TensorKit(computing, device)
        self._io_kit: IOKit = IOKit()

        self._computing = computing
        self._federation = federation
        self._role_to_parties = None

        self._gc = GC()

    def with_namespace(self, namespace: Namespace):
        context = copy(self)
        context.namespace = namespace
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
    def federation(self):
        return self._get_federation()

    @contextmanager
    def sub_ctx(self, namespace: str) -> Iterator["Context"]:
        with self.namespace.into_subnamespace(namespace):
            try:
                yield self
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

    def reader(self, uri, **kwargs) -> "Reader":
        return self._io_kit.reader(self, uri, **kwargs)

    def writer(self, uri, **kwargs) -> "Writer":
        return self._io_kit.writer(self, uri, **kwargs)
