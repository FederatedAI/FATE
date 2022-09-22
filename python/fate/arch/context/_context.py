import logging
from contextlib import contextmanager
from dataclasses import dataclass
from logging import Logger, disable, getLogger
from typing import List, Literal, Optional, Tuple, Iterator

from fate.interface import LOGMSG, Anonymous, Cache, CheckpointManager
from fate.interface import Context as ContextInterface
from fate.interface import Logger as LoggerInterface
from fate.interface import Metric as MetricInterface
from fate.interface import MetricMeta as MetricMetaInterface
from fate.interface import Metrics, Summary
from fate.interface import ComputingEngine
from ..session import Session

from ._federation import GC, FederationEngine
from ._namespace import Namespace
from ..common._parties import PartiesInfo, Party


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


# FIXME: vary complex to use, may take times to fix
class DummyAnonymous(Anonymous):
    ...


class DummyCheckpointManager(CheckpointManager):
    ...


class DummyLogger(LoggerInterface):
    def __init__(
        self,
        context_name: Optional[str] = None,
        namespace: Optional[Namespace] = None,
        level=logging.DEBUG,
        disable_buildin=True,
    ) -> None:
        if disable_buildin:
            self._disable_buildin()

        self.logger = getLogger("fate.dummy")
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

    @classmethod
    def _disable_buildin(cls):
        from ..common.log import getLogger

        logger = getLogger()
        logger.disabled = True

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
        computing: Optional[ComputingEngine] = None,
        federation: Optional[FederationEngine] = None,
        summary: Summary = DummySummary(),
        metrics: Metrics = DummyMetrics(),
        cache: Cache = DummyCache(),
        anonymous_generator: Anonymous = DummyAnonymous(),
        checkpoint_manager: CheckpointManager = DummyCheckpointManager(),
        log: Optional[LoggerInterface] = None,
        disable_buildin_logger=True,  # FIXME: just clear old loggers, remove in future
        namespace: Optional[Namespace] = None,
    ) -> None:
        self.context_name = context_name
        self.summary = summary
        self.metrics = metrics
        self.cache = cache
        self.anonymous_generator = anonymous_generator
        self.checkpoint_manager = checkpoint_manager

        if namespace is None:
            namespace = Namespace()
        self.namespace = namespace

        if log is None:
            log = DummyLogger(
                context_name, self.namespace, disable_buildin=disable_buildin_logger
            )
        self.log = log

        self._computing = computing
        self._federation = federation
        self._session = Session()
        self._gc = GC()

    def init_computing(self, computing_session_id=None):
        self._session.init_computing(computing_session_id=computing_session_id)

    def init_federation(
        self,
        federation_id,
        local_party: Tuple[Literal["guest", "host", "arbiter"], str],
        parties: List[Tuple[Literal["guest", "host", "arbiter"], str]],
    ):
        if self._federation is None:
            self._federation = FederationEngine(
                federation_id, local_party, parties, self, self._session, self.namespace
            )

    @contextmanager
    def sub_ctx(self, namespace) -> Iterator["Context"]:
        with self.namespace.into_subnamespace(namespace):
            try:
                yield self
            finally:
                ...

    @property
    def guest(self):
        return self._get_party_util().guest

    @property
    def hosts(self):
        return self._get_party_util().hosts

    @property
    def arbiter(self):
        return self._get_party_util().arbiter

    @property
    def parties(self):
        return self._get_party_util().parties

    def _get_party_util(self) -> FederationEngine:
        if self._federation is None:
            raise RuntimeError("federation session not init")
        return self._federation
