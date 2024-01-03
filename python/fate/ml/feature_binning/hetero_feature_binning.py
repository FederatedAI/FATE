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

import logging

import numpy as np
import pandas as pd

from fate.arch import Context
from fate.arch.histogram import HistogramBuilder
from ..abc.module import HeteroModule, Module

logger = logging.getLogger(__name__)


class HeteroBinningModuleGuest(HeteroModule):
    def __init__(
        self,
        method="quantile",
        n_bins=10,
        split_pt_dict=None,
        bin_col=None,
        transform_method=None,
        category_col=None,
        local_only=False,
        error_rate=1e-6,
        adjustment_factor=0.5,
    ):
        self.method = method
        self.bin_col = bin_col
        self.category_col = category_col
        self.n_bins = n_bins
        self._federation_bin_obj = None
        # param check
        if self.method in ["quantile", "bucket", "manual"]:
            self._bin_obj = StandardBinning(
                method, n_bins, split_pt_dict, bin_col, transform_method, category_col, error_rate, adjustment_factor
            )
        else:
            raise ValueError(f"{self.method} binning method not supported, please check")
        self.local_only = local_only
        self.column_anonymous_map = None

    def set_transform_method(self, transform_method):
        self._bin_obj.transform_method = transform_method

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        logger.info("Enter HeteroBinning fit.")
        self.column_anonymous_map = dict(zip(train_data.schema.columns, train_data.schema.anonymous_columns))
        train_data_binarized_label = train_data.label.get_dummies()
        label_count = train_data_binarized_label.shape[1]
        if label_count > 2:
            raise ValueError(
                f"More than 2 classes found in label column. "
                f"HeteroBinning currently only supports binary data. Please check."
            )

        self._bin_obj.fit(ctx, train_data)

    def compute_metrics(self, ctx: Context, binned_data):
        # label_tensor = binned_data.label.as_tensor()
        self._bin_obj.compute_metrics(binned_data)
        if not self.local_only:
            self.compute_federated_metrics(ctx, binned_data)

    def compute_federated_metrics(self, ctx: Context, binned_data):
        logger.info(f"Start computing federated metrics.")
        kit = ctx.cipher.phe.setup()
        encryptor = kit.get_tensor_encryptor()
        sk, pk, evaluator, coder = kit.sk, kit.pk, kit.evaluator, kit.coder

        label_tensor = binned_data.label.as_tensor()
        ctx.hosts.put("enc_y", encryptor.encrypt_tensor(label_tensor))
        ctx.hosts.put("pk", pk)
        ctx.hosts.put("evaluator", evaluator)
        ctx.hosts.put("coder", coder)
        host_col_bin = ctx.hosts.get("anonymous_col_bin")
        host_event_non_event_count = ctx.hosts.get("event_non_event_count")
        host_bin_sizes = ctx.hosts.get("feature_bin_sizes")
        for i, (col_bin_list, bin_sizes, en_host_count_res) in enumerate(
            zip(host_col_bin, host_bin_sizes, host_event_non_event_count)
        ):
            host_event_non_event_count_hist = en_host_count_res.decrypt(
                {"event_count": sk, "non_event_count": sk},
                {"event_count": (coder, None), "non_event_count": (coder, None)},
            )
            host_event_non_event_count_hist = host_event_non_event_count_hist.reshape(bin_sizes)
            summary_metrics, _ = self._bin_obj.compute_all_col_metrics(host_event_non_event_count_hist, col_bin_list)
            self._bin_obj.set_host_metrics(ctx.hosts[i], summary_metrics)

    def transform(self, ctx: Context, test_data):
        self.column_anonymous_map = dict(zip(test_data.schema.columns, test_data.schema.anonymous_columns))
        transformed_data = self._bin_obj.transform(ctx, test_data)
        return transformed_data

    def get_model(self):
        model_info = self._bin_obj.to_model()
        model = {
            "data": model_info,
            "meta": {
                "method": self.method,
                "metrics": ["iv"] if model_info.get("metrics_summary") else [],
                "local_only": self.local_only,
                "bin_col": self.bin_col,
                "category_col": self.category_col,
                "model_type": "binning",
                "n_bins": self.n_bins,
                "column_anonymous_map": self.column_anonymous_map,
            },
        }
        return model

    def restore(self, model):
        self._bin_obj.restore(model)

    @classmethod
    def from_model(cls, model) -> "HeteroBinningModuleGuest":
        bin_obj = HeteroBinningModuleGuest(
            method=model["meta"]["method"],
            bin_col=model["meta"]["bin_col"],
            category_col=model["meta"]["category_col"],
            n_bins=model["meta"]["n_bins"],
        )
        bin_obj.restore(model["data"])
        return bin_obj


class HeteroBinningModuleHost(HeteroModule):
    def __init__(
        self,
        method="quantile",
        n_bins=10,
        split_pt_dict=None,
        bin_col=None,
        transform_method=None,
        category_col=None,
        local_only=False,
        error_rate=1e-6,
        adjustment_factor=0.5,
    ):
        self.method = method
        self.n_bins = n_bins
        self._federation_bin_obj = None
        if self.method in ["quantile", "bucket", "manual"]:
            self._bin_obj = StandardBinning(
                method, n_bins, split_pt_dict, bin_col, transform_method, category_col, error_rate, adjustment_factor
            )
        self.local_only = local_only
        self.bin_col = bin_col
        self.category_col = category_col
        self.anonymous_col_bin = None
        self.column_anonymous_map = None

    def set_transform_method(self, new_transform_method):
        self._bin_obj.transform_method = new_transform_method

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        logger.info("Enter HeteroBinning fit.")
        self.column_anonymous_map = dict(zip(train_data.schema.columns, train_data.schema.anonymous_columns))
        self._bin_obj.fit(ctx, train_data)

    def compute_metrics(self, ctx: Context, binned_data):
        if not self.local_only:
            self.compute_federated_metrics(ctx, binned_data)

    def compute_federated_metrics(self, ctx: Context, binned_data):
        logger.info(f"Start computing federated metrics.")
        pk = ctx.guest.get("pk")
        evaluator = ctx.guest.get("evaluator")
        coder = ctx.guest.get("coder")
        columns = binned_data.schema.columns.to_list()
        # logger.info(f"self.bin_col: {self.bin_col}")
        to_compute_col = self.bin_col + self.category_col
        anonymous_col_bin = [binned_data.schema.anonymous_columns[columns.index(col)] for col in to_compute_col]

        ctx.guest.put("anonymous_col_bin", anonymous_col_bin)
        encrypt_y = ctx.guest.get("enc_y")
        # event count:
        feature_bin_sizes = [self._bin_obj._bin_count_dict[col] for col in self.bin_col]
        if self.category_col:
            for col in self.category_col:
                category_bin_size = binned_data[col].get_dummies().shape[1]
                feature_bin_sizes.append(category_bin_size)
        to_compute_data = binned_data[to_compute_col]
        to_compute_data.rename(
            columns=dict(zip(to_compute_data.schema.columns, to_compute_data.schema.anonymous_columns))
        )
        hist_targets = binned_data.create_frame()
        hist_targets["event_count"] = encrypt_y
        hist_targets["non_event_count"] = 1
        dtypes = hist_targets.dtypes

        hist_schema = {
            "event_count": {
                "type": "ciphertext",
                "stride": 1,
                "pk": pk,
                "evaluator": evaluator,
                "coder": coder,
                "dtype": dtypes["event_count"],
            },
            "non_event_count": {"type": "plaintext", "stride": 1, "dtype": dtypes["non_event_count"]},
        }
        hist = HistogramBuilder(
            num_node=1, feature_bin_sizes=feature_bin_sizes, value_schemas=hist_schema, enable_cumsum=False
        )
        event_non_event_count_hist = to_compute_data.distributed_hist_stat(
            histogram_builder=hist, targets=hist_targets
        )
        event_non_event_count_hist.i_sub_on_key("non_event_count", "event_count")
        ctx.guest.put("event_non_event_count", (event_non_event_count_hist))
        ctx.guest.put("feature_bin_sizes", feature_bin_sizes)

    def transform(self, ctx: Context, test_data):
        self.column_anonymous_map = dict(zip(test_data.schema.columns, test_data.schema.anonymous_columns))
        return self._bin_obj.transform(ctx, test_data)

    def get_model(self):
        model_info = self._bin_obj.to_model()
        model = {
            "data": model_info,
            "meta": {
                "method": self.method,
                "bin_col": self.bin_col,
                "category_col": self.category_col,
                "n_bins": self.n_bins,
                "model_type": "binning",
                "column_anonymous_map": self.column_anonymous_map,
            },
        }
        return model

    def restore(self, model):
        self._bin_obj.restore(model)

    @classmethod
    def from_model(cls, model) -> "HeteroBinningModuleHost":
        bin_obj = HeteroBinningModuleHost(
            method=model["meta"]["method"],
            bin_col=model["meta"]["bin_col"],
            category_col=model["meta"]["category_col"],
            n_bins=model["meta"]["n_bins"],
        )
        bin_obj.restore(model["data"])
        return bin_obj


class StandardBinning(Module):
    def __init__(
        self, method, n_bins, split_pt_dict, bin_col, transform_method, category_col, error_rate, adjustment_factor
    ):
        self.method = method
        self.n_bins = n_bins
        # cols to be binned
        self.bin_col = bin_col
        # cols to be treated as categorical feature
        self.category_col = category_col
        self.transform_method = transform_method
        self.relative_error = error_rate
        self.adjustment_factor = adjustment_factor
        self._manual_split_pt_dict = split_pt_dict
        # {col_name: [split_pts]}, ordered by split_pts
        self._split_pt_dict = None
        self._bin_idx_dict = None
        # {col_name: [bin_num]}, ordered by split_pts
        self._bin_count_dict = None
        # {col_name: [woe]}, ordered by split_pts, for transform
        self._woe_dict = None
        # for prediction transform
        self._train_woe_dict = None
        # {col_name: {"iv_array": [], "woe": [], "event_count": []...}}
        self._metrics_summary = None
        self._host_metrics_summary = None
        self._train_metrics_summary = None
        self._train_host_metrics_summary = None

    def set_host_metrics(self, host, metrics_summary):
        if self._host_metrics_summary is None:
            self._host_metrics_summary = {}
        self._host_metrics_summary[host.name] = metrics_summary

    def fit(self, ctx: Context, train_data, validate_data=None, skip_none=False):
        # only bin given `col_bin` cols
        if self.bin_col is None:
            self.bin_col = train_data.schema.columns.to_list()
        select_data = train_data[self.bin_col]

        if self.method == "quantile":
            q = list(np.arange(0, 1, 1 / self.n_bins)) + [1.0]
            split_pt_df = select_data.quantile(q=q, relative_error=self.relative_error).drop(0)
        elif self.method == "bucket":
            split_pt_df = select_data.qcut(q=self.n_bins)
        elif self.method == "manual":
            split_pt_df = pd.DataFrame.from_dict(self._manual_split_pt_dict)
        else:
            raise ValueError(f"Unknown binning method {self.method} encountered. Please check")
        # self._split_pt_dict = split_pt_df.to_dict()
        self._split_pt_dict = split_pt_df

        def __get_col_bin_count(col):
            count = len(col.unique())
            return count

        bin_count = split_pt_df.apply(__get_col_bin_count, axis=0)
        self._bin_count_dict = bin_count.to_dict()

    def bucketize_data(self, train_data):
        binned_df = train_data.bucketize(boundaries=self._split_pt_dict)
        return binned_df

    def compute_all_col_metrics(self, event_non_event_count_hist, columns):
        event_non_event_count = event_non_event_count_hist.to_dict(columns)[0]
        non_event_count_dict = event_non_event_count.get("non_event_count")
        event_count_dict = event_non_event_count.get("event_count")

        event_count, non_event_count = {}, {}
        event_rate, non_event_rate = {}, {}
        bin_woe, bin_iv, is_monotonic, iv = {}, {}, {}, {}
        total_event_count, total_non_event_count = None, None
        for col_name in event_count_dict.keys():
            col_event_count = pd.Series(
                {bin_num: int(bin_count.data) for bin_num, bin_count in event_count_dict[col_name].items()}
            )
            col_non_event_count = pd.Series(
                {bin_num: int(bin_count.data) for bin_num, bin_count in non_event_count_dict[col_name].items()}
            )
            if total_event_count is None:
                total_event_count = col_event_count.sum() or 1
                total_non_event_count = col_non_event_count.sum() or 1
            col_event_rate = (col_event_count == 0) * self.adjustment_factor + col_event_count / total_event_count
            col_non_event_rate = (
                col_non_event_count == 0
            ) * self.adjustment_factor + col_non_event_count / total_non_event_count
            col_rate_ratio = col_event_rate / col_non_event_rate
            col_bin_woe = col_rate_ratio.apply(lambda v: np.log(v))
            col_bin_iv = (col_event_rate - col_non_event_rate) * col_bin_woe

            event_count[col_name] = col_event_count.to_dict()
            non_event_count[col_name] = col_non_event_count.to_dict()
            event_rate[col_name] = col_event_rate.to_dict()
            non_event_rate[col_name] = col_non_event_rate.to_dict()
            bin_woe[col_name] = col_bin_woe.to_dict()
            bin_iv[col_name] = col_bin_iv.to_dict()
            is_monotonic[col_name] = col_bin_woe.is_monotonic_increasing or col_bin_woe.is_monotonic_decreasing
            iv[col_name] = col_bin_iv[1:].sum()

        metrics_summary = {}

        metrics_summary["event_count"] = event_count
        metrics_summary["non_event_count"] = non_event_count
        metrics_summary["event_rate"] = event_rate
        metrics_summary["non_event_rate"] = non_event_rate
        metrics_summary["woe"] = bin_woe
        metrics_summary["iv_array"] = bin_iv
        metrics_summary["is_monotonic"] = is_monotonic
        metrics_summary["iv"] = iv
        return metrics_summary, bin_woe

    def compute_metrics(self, binned_data):
        to_compute_col = self.bin_col + self.category_col
        to_compute_data = binned_data[to_compute_col]

        feature_bin_sizes = [self._bin_count_dict[col] for col in self.bin_col]
        if self.category_col:
            for col in self.category_col:
                category_bin_size = binned_data[col].get_dummies().shape[1]
                feature_bin_sizes.append(category_bin_size)
        hist_targets = binned_data.create_frame()
        hist_targets["event_count"] = binned_data.label
        hist_targets["non_event_count"] = 1
        dtypes = hist_targets.dtypes
        hist_schema = {
            "event_count": {"type": "plaintext", "stride": 1, "dtype": dtypes["event_count"]},
            "non_event_count": {"type": "plaintext", "stride": 1, "dtype": dtypes["non_event_count"]},
        }
        hist = HistogramBuilder(
            num_node=1, feature_bin_sizes=feature_bin_sizes, value_schemas=hist_schema, enable_cumsum=False
        )
        event_non_event_count_hist = to_compute_data.distributed_hist_stat(
            histogram_builder=hist, targets=hist_targets
        )
        event_non_event_count_hist.i_sub_on_key("non_event_count", "event_count")
        event_non_event_count_hist = event_non_event_count_hist.decrypt({}, {}).reshape(feature_bin_sizes)
        self._metrics_summary, self._woe_dict = self.compute_all_col_metrics(
            event_non_event_count_hist, to_compute_col
        )

    def transform(self, ctx: Context, binned_data):
        logger.debug(f"Given transform method: {self.transform_method}.")
        if self.transform_method == "bin_idx" and self._bin_idx_dict:
            return binned_data
        elif self.transform_method == "woe":
            if ctx.is_on_host:
                raise ValueError(f"host does not support 'woe' transform method, please use 'bin_idx'.")
            # predict: replace with woe from train phase
            to_transform_data = binned_data[self.bin_col]
            if self._train_woe_dict:
                logger.debug(f"`train_woe_dict` provided, will transform to woe values from training phase.")
                binned_data[self.bin_col] = to_transform_data.replace(self._train_woe_dict)
                # return binned_data.replace(self._train_woe_dict, self.bin_col)
            elif self._woe_dict:
                binned_data[self.bin_col] = to_transform_data.replace(self._woe_dict)
                # return binned_data.replace(self._woe_dict, self.bin_col)
        else:
            logger.warning(
                f"to transform type {self.transform_method} encountered, but no bin tag dict provided. "
                f"Please check"
            )
        return binned_data

    def to_model(self):
        return dict(
            method=self.method,
            bin_col=self.bin_col,
            split_pt_dict=self._split_pt_dict.to_dict(),
            bin_idx_dict=self._bin_idx_dict,
            bin_count_dict=self._bin_count_dict,
            metrics_summary=self._metrics_summary,
            train_metrics_summary=self._train_metrics_summary,
            host_metrics_summary=self._host_metrics_summary,
            train_host_metrics_summary=self._train_host_metrics_summary,
            woe_dict=self._woe_dict,
            category_col=self.category_col,
            adjustment_factor=self.adjustment_factor
            # transform_method = self.transform_method,
        )

    def restore(self, model):
        self.method = model["method"]
        self.bin_col = model["bin_col"]
        # self.transform_method = model["transform_method"]
        self._split_pt_dict = pd.DataFrame.from_dict(model["split_pt_dict"])
        self._bin_idx_dict = model["bin_idx_dict"]
        self._bin_count_dict = model["bin_count_dict"]
        # load predict model
        if model.get("train_metrics_summary"):
            self._metrics_summary = model["metrics_summary"]
            self._train_metrics_summary = model["train_metrics_summary"]
        else:
            self._train_metrics_summary = model["metrics_summary"]
        if model.get("train_host_metrics_summary"):
            self._host_metrics_summary = model["host_metrics_summary"]
            self._train_host_metrics_summary = model["train_host_metrics_summary"]
        else:
            self._train_host_metrics_summary = model["host_metrics_summary"]
        if model.get("train_woe_dict"):
            self._woe_dict = model["woe_dict"]
            self._train_woe_dict = model["train_woe_dict"]
        else:
            self._train_woe_dict = model["woe_dict"]

        self.category_col = model["category_col"]
        self.adjustment_factor = model["adjustment_factor"]
