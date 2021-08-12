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
#

import importlib
import inspect
import typing
from pathlib import Path

from fate_arch.common import log
from federatedml.model_base import ModelBase
from federatedml.param.base_param import BaseParam

_ml_base = Path(__file__).resolve().parent.parent.parent

LOGGER = log.getLogger()


class ComponentMeta:
    __name_to_obj: typing.Dict[str, "ComponentMeta"] = {}

    def __init__(self, name) -> None:
        self.name = name
        self._role_to_runner_cls = {}
        self._role_to_runner_cls_getter = {}  # lazy
        self._param_cls = None
        self._param_cls_getter = None  # lazy

        self.__name_to_obj[name] = self

    def impl_runner(self, *args: str):
        def _wrap(cls):
            if isinstance(cls, ModelBase):
                for role in args:
                    self._role_to_runner_cls[role] = cls
            elif inspect.isfunction(cls):
                for role in args:
                    self._role_to_runner_cls_getter[role] = cls
            else:
                raise NotImplementedError(f"type of {cls} not supported")

            return cls

        return _wrap

    @property
    def impl_param(self):
        def _wrap(cls):
            if isinstance(cls, BaseParam):
                self._param_cls = cls
            elif inspect.isfunction(cls):
                self._param_cls_getter = cls
            else:
                raise NotImplementedError(f"type of {cls} not supported")
            return cls

        return _wrap

    def register_info(self):
        return {
            self.name: dict(
                module=self.__module__,
            )
        }

    @classmethod
    def get_meta(cls, name):
        return cls.__name_to_obj[name]

    def _get_runner(self, role: str):
        if role in self._role_to_runner_cls:
            return self._role_to_runner_cls[role]

        elif role in self._role_to_runner_cls_getter:
            return self._role_to_runner_cls_getter[role]()

        else:
            raise ModuleNotFoundError(
                f"Runner for component `{self.name}` at role `{role}` not found"
            )

    def get_run_obj(self, role: str):
        return self._get_runner(role)()

    def get_run_obj_name(self, role: str) -> str:
        return self._get_runner(role).__name__

    def get_param_obj(self, cpn_name: str):
        if self._param_cls is not None:
            param_obj = self._param_cls()
        elif self._param_cls_getter is not None:
            param_obj = self._param_cls_getter()()
        else:
            raise ModuleNotFoundError(f"Param for component `{self.name}` not found")
        return param_obj.set_name(f"{self.name}#{cpn_name}")

    def get_supported_roles(self):
        return set(self._role_to_runner_cls) | set(self._role_to_runner_cls_getter)


def _search_components(path):
    try:
        module_name = (
            path.absolute()
            .relative_to(_ml_base)
            .with_suffix("")
            .__str__()
            .replace("/", ".")
        )
        module = importlib.import_module(module_name)
    except ImportError as e:
        # or skip ?
        raise e
    _obj_pairs = inspect.getmembers(module, lambda obj: isinstance(obj, ComponentMeta))
    return _obj_pairs


class Components:
    @classmethod
    def get_names(cls) -> typing.Dict[str, dict]:
        names = {}
        _components_base = Path(__file__).resolve().parent
        for p in _components_base.glob("**/*.py"):
            for name, obj in _search_components(p):
                info = obj.register_info()
                LOGGER.info(f"component register {name} with cache info {info}")
                names.update(info)
        return names

    @classmethod
    def get(cls, name: str, cache) -> ComponentMeta:
        if cache:
            importlib.import_module(cache["module"])

        return ComponentMeta.get_meta(name)


"""
define components
"""
dataio_cpn_meta = ComponentMeta("DataIO")


@dataio_cpn_meta.impl_param
def dataio_param():
    from federatedml.param.dataio_param import DataIOParam

    return DataIOParam


@dataio_cpn_meta.impl_runner("guest", "host")
def dataio_runner():
    from federatedml.util.data_io import DataIO

    return DataIO


hetero_binning_cpn_meta = ComponentMeta("HeteroFeatureBinning")


@hetero_binning_cpn_meta.impl_param
def hetero_feature_binning_param():
    from federatedml.param.feature_binning_param import HeteroFeatureBinningParam

    return HeteroFeatureBinningParam


@hetero_binning_cpn_meta.impl_runner("guest")
def hetero_feature_binning_guest_runner():
    from federatedml.feature.hetero_feature_binning.hetero_binning_guest import (
        HeteroFeatureBinningGuest,
    )

    return HeteroFeatureBinningGuest


@hetero_binning_cpn_meta.impl_runner("host")
def hetero_feature_binning_host_runner():
    from federatedml.feature.hetero_feature_binning.hetero_binning_host import (
        HeteroFeatureBinningHost,
    )

    return HeteroFeatureBinningHost


hetero_feature_selection_cpn_meta = ComponentMeta("HeteroFeatureSelection")


@hetero_feature_selection_cpn_meta.impl_param
def hetero_feature_selection_param():
    from federatedml.param.feature_selection_param import FeatureSelectionParam

    return FeatureSelectionParam


@hetero_feature_selection_cpn_meta.impl_runner("guest")
def hetero_feature_selection_guest_runner():
    from federatedml.feature.hetero_feature_selection.feature_selection_guest import (
        HeteroFeatureSelectionGuest,
    )

    return HeteroFeatureSelectionGuest


@hetero_feature_selection_cpn_meta.impl_runner("host")
def hetero_feature_selection_host_runner():
    from federatedml.feature.hetero_feature_selection.feature_selection_host import (
        HeteroFeatureSelectionHost,
    )

    return HeteroFeatureSelectionHost


intersection_cpn_meta = ComponentMeta("Intersection")


@intersection_cpn_meta.impl_param
def intersection_param():
    from federatedml.param.intersect_param import IntersectParam

    return IntersectParam


@intersection_cpn_meta.impl_runner("guest")
def intersection_guest_runner():
    from federatedml.statistic.intersect.intersect_model import IntersectGuest

    return IntersectGuest


@intersection_cpn_meta.impl_runner("host")
def intersection_host_runner():
    from federatedml.statistic.intersect.intersect_model import IntersectHost

    return IntersectHost


hetero_lr_cpn_meta = ComponentMeta("HeteroLR")


@hetero_lr_cpn_meta.impl_param
def hetero_lr_param():
    from federatedml.param.logistic_regression_param import HeteroLogisticParam

    return HeteroLogisticParam


@hetero_lr_cpn_meta.impl_runner("guest")
def hetero_lr_runner_guest():
    from federatedml.linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_guest import (
        HeteroLRGuest,
    )

    return HeteroLRGuest


@hetero_lr_cpn_meta.impl_runner("host")
def hetero_lr_runner_host():
    from federatedml.linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_host import (
        HeteroLRHost,
    )

    return HeteroLRHost


@hetero_lr_cpn_meta.impl_runner("arbiter")
def hetero_lr_runner_arbiter():
    from federatedml.linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_arbiter import (
        HeteroLRArbiter,
    )

    return HeteroLRArbiter


evaluation_cpn_meta = ComponentMeta("Evaluation")


@evaluation_cpn_meta.impl_param
def evaluation_param():
    from federatedml.param.evaluation_param import EvaluateParam

    return EvaluateParam


@evaluation_cpn_meta.impl_runner("guest", "host", "arbiter")
def evaluation_runner():
    from federatedml.evaluation.evaluation import Evaluation

    return Evaluation
