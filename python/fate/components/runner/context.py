from typing import List, Optional

from fate.arch.context import Backend, Context, Metric, MetricMeta, Namespace
from fate.arch.unify import device
from fate.interface import Cache, ComputingEngine, FederationEngine
from fate.interface import Logger as LoggerInterface
from fate.interface import Metric as MetricInterface
from fate.interface import MetricMeta as MetricMetaInterface
from fate.interface import Metrics, PartyMeta, Summary

from .parser.anonymous import copy
from .parser.checkpoint import CheckpointManager


class ComponentContext(Context):
    """
    implement fate.interface.Context for flow runner

    this implemention is specificated for fate.components, ie:
      - `summary` and `metrics` are traceback using flow's track client
      - `metrics` has additional propety `surfix` for CV/Stepwise to log metric separated in fold
      - ...
    """

    def __init__(
        self,
        context_name: Optional[str] = None,
        device: device = device.CPU,
        computing: Optional[ComputingEngine] = None,
        federation: Optional[FederationEngine] = None,
        log: Optional[LoggerInterface] = None,
        namespace: Optional[Namespace] = None,
    ) -> None:
        self.namespace = namespace

        # self.checkpoint_manager: CheckpointManager = checkpoint_manager
        # self.summary: ComponentSummary = ComponentSummary(tracker)
        # self.metrics: ComponentsMetrics = ComponentsMetrics(tracker)
        self.cache: ComponentsCache = ComponentsCache()

        super().__init__(
            context_name=context_name,
            device=device,
            computing=computing,
            federation=federation,
        )


class ComponentSummary(Summary):
    def __init__(self, tracker) -> None:
        self.tracker = tracker
        self._summary = {}

    def save(self):
        self.tracker.log_component_summary(summary_data=self._summary)

    @property
    def summary(self):
        return copy.deepcopy(self._summary)

    def reset(self, summary: dict):
        if not isinstance(summary, dict):
            raise ValueError(
                f"summary should be of dict type, received {type(summary)} instead."
            )
        self._summary = copy.deepcopy(summary)

    def add(self, key: str, value):
        if (exists_value := self._summary.get(key)) is not None:
            raise ValueError(f"key `{key}` already exists with value `{exists_value}`")
        self._summary[key] = value


class ComponentsMetrics(Metrics):
    def __init__(self, tracker) -> None:
        self.tracker = tracker
        self.name_surfix = None

    def get_name(self, name):
        if self.name_surfix is not None:
            name = f"{name}{self.name_surfix}"
        return name

    def log(self, name: str, namespace: str, data: List[MetricInterface]):
        name = self.get_name(name)
        self.tracker.log_metric_data(
            metric_name=name,
            metric_namespace=namespace,
            metrics=data,
        )

    def log_meta(self, name: str, namespace: str, meta: MetricMetaInterface):
        name = self.get_name(name)
        meta.update_metas({"curve_name": name})
        self.tracker.set_metric_meta(
            metric_name=name,
            metric_namespace=namespace,
            metric_meta=meta,
        )

    def log_warmstart_init_iter(self, iter_num):
        metric_meta = MetricMeta(
            name="train",
            metric_type="init_iter",
            extra_metas={
                "unit_name": "iters",
            },
        )

        self.log_meta(name="init_iter", namespace="train", meta=metric_meta)
        self.log(
            name="init_iter",
            namespace="train",
            data=[Metric("init_iter", iter_num)],
        )


class ComponentsCache(Cache):
    def __init__(self) -> None:
        self.cache: List = []

    def add_cache(self, key, value):
        self.cache.append((key, value))
