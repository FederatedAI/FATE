#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

import copy
import logging
import math
import random

import numpy as np
import pandas as pd

from fate.arch import Context
from ..abc.module import HeteroModule, Module

logger = logging.getLogger(__name__)

DEFAULT_METRIC = {"iv": ["iv"], "statistics": ["mean"]}


class HeteroSelectionModuleGuest(HeteroModule):
    def __init__(
        self,
        method=None,
        select_col=None,
        input_models=None,
        iv_param=None,
        statistic_param=None,
        manual_param=None,
        keep_one=True,
    ):
        self.method = method
        self.select_col = select_col
        self.iv_param = iv_param
        self.statistic_param = statistic_param
        self.manual_param = manual_param
        self.keep_one = keep_one

        # keep selection history
        self._inner_method = []
        self._selection_obj = []

        self.isometric_model_dict = None
        if input_models:
            isometric_model_dict = {}
            for model in input_models:
                model_type = model["meta"].get("model_type")
                if model_type is None:
                    raise ValueError(f"Missing 'model_type' in input model")
                isometric_model_dict[model_type] = model
            self.isometric_model_dict = isometric_model_dict

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        # logger.info(f"isometric_model_dict: {self.isometric_model_dict}")
        if self.select_col is None:
            self.select_col = train_data.schema.columns.to_list()

        select_data = train_data[self.select_col]
        header = select_data.schema.columns.to_list()
        for i, filter_type in enumerate(self.method):
            if filter_type == "manual":
                selection_obj = ManualSelection(
                    method=filter_type, header=header, param=self.manual_param, keep_one=self.keep_one
                )
            elif filter_type == "iv":
                model = self.isometric_model_dict.get("binning", None)
                if model is None:
                    raise ValueError(f"Cannot find binning model in input, please check")
                selection_obj = StandardSelection(
                    method=filter_type, header=header, param=self.iv_param, model=model, keep_one=self.keep_one
                )
            elif filter_type == "statistics":
                model = self.isometric_model_dict.get("statistics", None)
                if model is None:
                    raise ValueError(f"Cannot find statistics model in input, please check")
                selection_obj = StandardSelection(
                    method=filter_type, header=header, param=self.statistic_param, model=model, keep_one=self.keep_one
                )
            else:
                raise ValueError(f"{filter_type} selection method not supported, please check")
            self._selection_obj.append(selection_obj)
            self._inner_method.append(filter_type)

        prev_selection_obj = None
        for method, selection_obj in zip(self._inner_method, self._selection_obj):
            if prev_selection_obj:
                selection_obj.set_prev_selected_mask(copy.deepcopy(prev_selection_obj.selected_mask))
                if isinstance(selection_obj, StandardSelection) and isinstance(prev_selection_obj, StandardSelection):
                    selection_obj.set_host_prev_selected_mask(copy.deepcopy(prev_selection_obj._host_selected_mask))
            selection_obj.fit(ctx, select_data)
            if method == "iv":
                if self.iv_param.get("select_federated"):
                    HeteroSelectionModuleGuest.sync_select_federated(ctx, selection_obj)
            prev_selection_obj = selection_obj

    @staticmethod
    def sync_select_federated(ctx: Context, selection_obj):
        logger.info(f"Sync federated selection.")
        for i, host in enumerate(ctx.hosts):
            federated_mask = selection_obj._host_selected_mask[host.name]
            ctx.hosts[i].put(f"selected_mask_{selection_obj.method}", federated_mask)

    def transform(self, ctx: Context, test_data):
        transformed_data = self._selection_obj[-1].transform(ctx, test_data)
        return transformed_data

    def get_model(self):
        # all selection obj need to be recorded for display of cascade order
        selection_obj_list = []
        for selection_obj in self._selection_obj:
            selection_obj_list.append(selection_obj.to_model())
        data = {"selection_obj_list": selection_obj_list, "inner_method": self._inner_method}
        meta = {"method": self.method, "select_col": self.select_col, "keep_one": self.keep_one}
        return {"data": data, "meta": meta}

    def restore(self, model):
        selection_obj_list = []
        selection_obj_model_list = model["selection_obj_list"]
        for i, selection_model in enumerate(selection_obj_model_list):
            if selection_model["method"] in ["manual"]:
                selection_obj = ManualSelection(method=self._inner_method[i])
            else:
                selection_obj = StandardSelection(method=self._inner_method[i])
            selection_obj.restore(selection_model)
            selection_obj_list.append(selection_obj)
        self._selection_obj = selection_obj_list

    @classmethod
    def from_model(cls, model) -> "HeteroSelectionModuleGuest":
        selection_obj = HeteroSelectionModuleGuest(model["meta"]["method"], model["meta"]["select_col"])
        selection_obj._inner_method = model["data"]["inner_method"]
        selection_obj.restore(model["data"])
        return selection_obj


class HeteroSelectionModuleHost(HeteroModule):
    def __init__(
        self,
        method=None,
        select_col=None,
        input_models=None,
        iv_param=None,
        statistic_param=None,
        manual_param=None,
        keep_one=True,
    ):
        self.method = method
        self.iv_param = iv_param
        self.statistic_param = statistic_param
        self.manual_param = manual_param
        self.keep_one = keep_one
        self.select_col = select_col
        # keep selection history
        self._inner_method = []
        self._selection_obj = []

        self.isometric_model_dict = None
        if input_models:
            isometric_model_dict = {}
            for model in input_models:
                model_type = model["meta"].get("model_type")
                if model_type is None:
                    raise ValueError(f"Missing 'model_type' in input model")
                isometric_model_dict[model_type] = model
            self.isometric_model_dict = isometric_model_dict

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        if self.select_col is None:
            self.select_col = train_data.schema.columns.to_list()
        select_data = train_data[self.select_col]
        header = select_data.schema.columns.to_list()
        for i, filter_type in enumerate(self.method):
            if filter_type == "manual":
                selection_obj = ManualSelection(
                    method=filter_type, header=header, param=self.manual_param, keep_one=self.keep_one
                )
            elif filter_type == "iv":
                model = self.isometric_model_dict.get("binning", None)
                if model is None:
                    raise ValueError(f"Cannot find binning model in input, please check")
                selection_obj = StandardSelection(
                    method=filter_type, header=header, param=self.iv_param, model=model, keep_one=self.keep_one
                )
            elif filter_type == "statistics":
                model = self.isometric_model_dict.get("statistics", None)
                if model is None:
                    raise ValueError(f"Cannot find statistics model in input, please check")
                selection_obj = StandardSelection(
                    method=filter_type, header=header, param=self.statistic_param, model=model, keep_one=self.keep_one
                )

            else:
                raise ValueError(f"{type} selection method not supported, please check")
            self._selection_obj.append(selection_obj)
            self._inner_method.append(filter_type)

        prev_selection_obj = None
        for method, selection_obj in zip(self._inner_method, self._selection_obj):
            if prev_selection_obj:
                selection_obj.set_prev_selected_mask(copy.deepcopy(prev_selection_obj.selected_mask))
            selection_obj.fit(ctx, train_data, validate_data)
            if method == "iv":
                if self.iv_param.get("select_federated"):
                    HeteroSelectionModuleHost.sync_select_federated(ctx, selection_obj, train_data)
            prev_selection_obj = selection_obj

    @staticmethod
    def sync_select_federated(ctx: Context, selection_obj, data):
        cur_selected_mask = ctx.guest.get(f"selected_mask_{selection_obj.method}")
        # logger.info(f"cur_selected_mask: {cur_selected_mask}")
        columns, anonymous_columns = data.schema.columns, data.schema.anonymous_columns
        # logger.info(f"anonymous columns: {data.schema.anonymous_columns}")
        new_index = [columns[anonymous_columns.get_loc(col)] for col in cur_selected_mask.index]
        cur_selected_mask.index = new_index
        prev_selected_mask = selection_obj._prev_selected_mask[selection_obj._prev_selected_mask]
        missing_col = set(prev_selected_mask.index).difference(set(new_index))
        if missing_col:
            raise ValueError(f"results for columns: {missing_col} not found in received selection result.")
        cur_selected_mask = [cur_selected_mask.get(col, False) for col in selection_obj._header]
        selected_mask = selection_obj._prev_selected_mask & cur_selected_mask
        selection_obj.set_selected_mask(selected_mask)

    def transform(self, ctx: Context, test_data):
        transformed_data = self._selection_obj[-1].transform(ctx, test_data)
        return transformed_data

    def get_model(self):
        # all selection history need to be recorded for display
        selection_obj_list = []
        for selection_obj in self._selection_obj:
            selection_obj_list.append(selection_obj.to_model())

        data = {"selection_obj_list": selection_obj_list, "inner_method": self._inner_method}
        meta = {"method": self.method, "select_col": self.select_col, "keep_one": self.keep_one}
        return {"data": data, "meta": meta}

    def restore(self, model):
        selection_obj_list = []
        selection_obj_model_list = model["selection_obj_list"]
        for i, selection_model in enumerate(selection_obj_model_list):
            if selection_model["method"] in ["manual"]:
                selection_obj = ManualSelection(method=self._inner_method[i])
            else:
                selection_obj = StandardSelection(method=self._inner_method[i])
            selection_obj.restore(selection_model)
            selection_obj_list.append(selection_obj)
        self._selection_obj = selection_obj_list

    @classmethod
    def from_model(cls, model) -> "HeteroSelectionModuleHost":
        selection_obj = HeteroSelectionModuleHost(model["meta"]["method"], model["meta"]["select_col"])
        selection_obj._inner_method = model["data"]["inner_method"]
        selection_obj.restore(model["data"])
        return selection_obj


class ManualSelection(Module):
    def __init__(self, method, param=None, header=None, model=None, keep_one=True):
        assert method == "manual", f"Manual Selection only accepts 'manual' as `method`, received {method} instead."
        self.method = method
        self.param = param
        self.model = model
        self.keep_one = keep_one
        self._header = header
        self._prev_selected_mask = None
        self._selected_mask = None
        if header is None:
            self._prev_selected_mask = None
        else:
            self._prev_selected_mask = pd.Series(np.ones(len(header)), dtype=bool, index=header)

    @property
    def selected_mask(self):
        return self._selected_mask

    def set_selected_mask(self, mask):
        self._selected_mask = mask

    def set_prev_selected_mask(self, mask):
        self._prev_selected_mask = mask

    def fit(self, ctx: Context, train_data, validate_data=None):
        header = train_data.schema.columns.to_list()
        if self._header is None:
            self._header = header
            self._prev_selected_mask = pd.Series(np.ones(len(header)), dtype=bool, index=header)

        filter_out_col = self.param.get("filter_out_col", None)
        keep_col = self.param.get("keep_col", None)
        if filter_out_col is None:
            filter_out_col = []
        if keep_col is None:
            keep_col = []
        if len(filter_out_col) >= len(header):
            raise ValueError("`filter_out_col` should not be all columns")
        filter_out_col = set(filter_out_col)
        # keep_col = set(keep_col)
        missing_col = (filter_out_col.union(set(keep_col))).difference(set(self._prev_selected_mask.index))
        if missing_col:
            raise ValueError(
                f"columns {missing_col} given in `filter_out_col` & `keep_col` " f"not found in `select_col` or header"
            )
        filter_out_mask = pd.Series(
            [False if col in filter_out_col else True for col in self._header], index=self._header
        )
        # keep_mask = [True if col in keep_col else False for col in self._header]
        selected_mask = self._prev_selected_mask & filter_out_mask
        selected_mask.loc[keep_col] = True
        self._selected_mask = selected_mask
        if self.keep_one:
            StandardSelection._keep_one(self._selected_mask, self._header)

    def transform(self, ctx: Context, transform_data):
        logger.debug(f"Start transform")
        drop_cols = set(self._selected_mask[~self._selected_mask].index)
        select_cols = [col for col in transform_data.schema.columns.to_list() if col not in drop_cols]
        return transform_data[select_cols]

    def to_model(self):
        return dict(method=self.method, keep_one=self.keep_one, selected_mask=self._selected_mask.to_dict())

    def restore(self, model):
        self.method = model["method"]
        self.keep_one = model["keep_one"]
        self._selected_mask = pd.Series(model["selected_mask"], dtype=bool)


class StandardSelection(Module):
    def __init__(self, method, header=None, param=None, model=None, keep_one=True):
        self.method = method
        self.param = param
        self.filter_conf = {}

        if param is not None:
            for metric_name, filter_type, threshold, take_high in zip(
                self.param.get("metrics", DEFAULT_METRIC.get(method)),
                self.param.get("filter_type", ["threshold"]),
                self.param.get("threshold", [1.0]),
                self.param.get("take_high", [True]),
            ):
                metric_conf = self.filter_conf.get(metric_name, {})
                metric_conf["filter_type"] = metric_conf.get("filter_type", []) + [filter_type]
                metric_conf["threshold"] = metric_conf.get("threshold", []) + [threshold]
                metric_conf["take_high"] = metric_conf.get("take_high", []) + [take_high]
                self.filter_conf[metric_name] = metric_conf

        self.model = self.convert_model(model)
        self.keep_one = keep_one
        self._header = header
        self._selected_mask = None
        self._all_selected_mask = None
        if header is None:
            self._prev_selected_mask = None
        else:
            self._prev_selected_mask = pd.Series(np.ones(len(header)), dtype=bool, index=header)
        self._host_selected_mask = {}
        self._all_host_selected_mask = {}
        self._host_prev_selected_mask = {}
        self._all_metrics = None
        self._all_host_metrics = {}

    @staticmethod
    def convert_model(input_model):
        return input_model

    @property
    def selected_mask(self):
        return self._selected_mask

    def set_selected_mask(self, mask):
        self._selected_mask = mask

    def set_host_prev_selected_mask(self, mask):
        self._host_prev_selected_mask = mask

    def set_prev_selected_mask(self, mask):
        self._prev_selected_mask = mask

    def fit(self, ctx: Context, train_data, validate_data=None):
        if self._header is None:
            header = train_data.schema.columns.to_list()
            self._header = header
            self._prev_selected_mask = pd.Series(np.ones(len(header)), dtype=bool, index=header)

        metric_names = self.param.get("metrics", [])
        # local only
        if self.method in ["statistics"]:
            for metric_name in metric_names:
                if metric_name not in self.model.get("meta", {}).get("metrics", {}):
                    raise ValueError(
                        f"metric {metric_name} not found in given statistic model with metrics: "
                        f"{self.model.get('metrics', {})}, please check"
                    )
            model_data = self.model.get("data", {})
            metrics_all = pd.DataFrame(model_data.get("metrics_summary", {})).loc[metric_names]
            self._all_metrics = metrics_all
            missing_col = set(self._prev_selected_mask[self._prev_selected_mask].index).difference(
                set(metrics_all.columns)
            )
            if missing_col:
                raise ValueError(
                    f"metrics for columns {missing_col} from `select_col` or header not found in given model."
                )

            mask_all = self.apply_filter(metrics_all, self.filter_conf)
            self._all_selected_mask = mask_all
            cur_selected_mask = mask_all.all(axis=0)
            cur_selected_mask = [cur_selected_mask.get(col, False) for col in self._header]
            self._selected_mask = self._prev_selected_mask & cur_selected_mask
            if self.keep_one:
                self._keep_one(self._selected_mask, self._prev_selected_mask, self._header)
        # federated selection possible
        elif self.method == "iv":
            # host does not perform local iv selection
            if ctx.is_on_host:
                return
            model_data = self.model.get("data", {})
            iv_metrics = pd.Series(model_data["metrics_summary"]["iv"])
            # metrics_all = pd.DataFrame(iv_metrics).T.rename({0: "iv"}, axis=0)
            metrics_all = StandardSelection.convert_series_metric_to_dataframe(iv_metrics, "iv")
            self._all_metrics = metrics_all
            mask_all = self.apply_filter(metrics_all, self.filter_conf)
            self._all_selected_mask = mask_all
            cur_selected_mask = mask_all.all(axis=0)
            cur_selected_mask = [cur_selected_mask.get(col, False) for col in self._header]
            self._selected_mask = self._prev_selected_mask & cur_selected_mask
            if self.keep_one:
                self._keep_one(self._selected_mask, self._prev_selected_mask, self._header)
            if self.param.get("select_federated", True):
                host_metrics_summary = self.model.get("data", {}).get("host_metrics_summary")
                # logger.info(f"host metrics summary: {host_metrics_summary}")
                if host_metrics_summary is None:
                    raise ValueError(f"Host metrics not found in provided model, please check.")
                for host, host_metrics in host_metrics_summary.items():
                    iv_metrics = pd.Series(host_metrics["iv"])
                    # metrics_all = pd.DataFrame(iv_metrics).T.rename({0: "iv"}, axis=0)
                    metrics_all = StandardSelection.convert_series_metric_to_dataframe(iv_metrics, "iv")
                    self._all_host_metrics[host] = metrics_all
                    host_mask_all = self.apply_filter(metrics_all, self.filter_conf)
                    self._all_host_selected_mask[host] = host_mask_all
                    host_selected_mask = host_mask_all.all(axis=0)
                    if self.keep_one:
                        self._keep_one(host_selected_mask)
                    self._host_selected_mask[host] = host_selected_mask

    @staticmethod
    def _keep_one(cur_mask, prev_mask=None, select_col=None):
        if sum(cur_mask) > 0:
            return cur_mask
        else:
            if prev_mask is not None:
                idx = random.choice(prev_mask[prev_mask].index)
            elif select_col is not None:
                idx = random.choice(select_col)
            else:
                idx = random.choice(cur_mask.index)
            cur_mask[idx] = True

    @staticmethod
    def convert_series_metric_to_dataframe(metrics, metric_name):
        # logger.info(f"metrics: {metrics}")
        return pd.DataFrame(metrics).T.rename({0: metric_name}, axis=0)

    @staticmethod
    def apply_filter(metrics_all, filter_conf):
        return metrics_all.apply(lambda r: StandardSelection.filter_multiple_metrics(r, filter_conf[r.name]), axis=1)

    @staticmethod
    def filter_multiple_metrics(metrics, metric_conf):
        filter_type_list = metric_conf["filter_type"]
        threshold_list = metric_conf["threshold"]
        take_high_list = metric_conf["take_high"]
        result = pd.Series(np.ones(len(metrics.index)), index=metrics.index, dtype=bool)
        for idx in range(len(filter_type_list)):
            result &= StandardSelection.filter_metrics(
                metrics, filter_type_list[idx], threshold_list[idx], take_high_list[idx]
            )
        return result

    @staticmethod
    def filter_metrics(metrics, filter_type, threshold, take_high=True):
        if filter_type == "top_k":
            return StandardSelection.filter_by_top_k(metrics, threshold, take_high)
        elif filter_type == "threshold":
            return StandardSelection.filter_by_threshold(metrics, threshold, take_high)
        elif filter_type == "top_percentile":
            return StandardSelection.filter_by_percentile(metrics, threshold, take_high)
        else:
            raise ValueError(f"filter_type {filter_type} not supported, please check")

    @staticmethod
    def filter_by_top_k(metrics, k, take_high=True):
        # strict top k
        if k == 0:
            return pd.Series(np.ones(len(metrics)), dtype=bool)
        # stable sort
        ordered_metrics = metrics.sort_values(ascending=not take_high, kind="mergesort")
        select_k = ordered_metrics.index[:k]
        return metrics.index.isin(select_k)

    @staticmethod
    def filter_by_threshold(metrics, threshold, take_high=True):
        if take_high:
            return metrics >= threshold
        else:
            return metrics <= threshold

    @staticmethod
    def filter_by_percentile(metrics, percentile, take_high=True):
        top_k = math.ceil(len(metrics) * percentile)
        return StandardSelection.filter_by_top_k(metrics, top_k, take_high)

    def transform(self, ctx: Context, transform_data):
        logger.debug(f"Start transform")
        drop_cols = set(self._selected_mask[~self._selected_mask].index)
        cols = transform_data.schema.columns.to_list()
        select_cols = [col for col in cols if col not in drop_cols]
        return transform_data[select_cols]

    def to_model(self):
        return dict(
            method=self.method,
            keep_one=self.keep_one,
            all_selected_mask=self._all_selected_mask.to_dict() if self._all_selected_mask is not None else None,
            all_metrics=self._all_metrics.to_dict() if self._all_metrics is not None else None,
            all_host_metrics={k: v.to_dict() for k, v in self._all_host_metrics.items()}
            if self._all_host_metrics is not None
            else None,
            selected_mask=self._selected_mask.to_dict(),
            host_selected_mask={k: v.to_dict() for k, v in self._host_selected_mask.items()}
            if self._host_selected_mask is not None
            else None,
            all_host_selected_mask={k: v.to_dict() for k, v in self._all_host_selected_mask.items()}
            if self._all_host_selected_mask is not None
            else None,
        )

    def restore(self, model):
        self.method = model["method"]
        self.keep_one = model["keep_one"]
        self._selected_mask = pd.Series(model["selected_mask"], dtype=bool)
        self._all_selected_mask = pd.DataFrame(model["all_selected_mask"], dtype=bool)
        self._all_metrics = pd.DataFrame(model["all_metrics"])
        self._host_selected_mask = {k: pd.Series(v, dtype=bool) for k, v in model["host_selected_mask"].items()}
        self._all_host_selected_mask = {
            k: pd.DataFrame(v, dtype=bool) for k, v in model["all_host_selected_mask"].items()
        }
        self._all_host_metrics = {k: pd.DataFrame(v) for k, v in model["all_host_metrics"].items()}
