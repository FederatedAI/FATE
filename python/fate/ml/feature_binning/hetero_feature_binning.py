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

import json
import logging

import numpy as np
import pandas as pd
import torch

from fate.interface import Context
from ..abc.module import Module

logger = logging.getLogger(__name__)


class HeteroBinningModuleGuest(Module):
    def __init__(self, method="quantile", n_bins=10, split_pt_dict=None, bin_col=None, transform_method=None,
                 category_col=None, local_only=False, error_rate=1e-3, adjustment_factor=0.5):
        self.method = method
        self._federation_bin_obj = None
        # param check
        if self.method in ["quantile", "bucket", "manual"]:
            self._bin_obj = StandardBinning(method, n_bins, split_pt_dict, bin_col, transform_method,
                                            category_col, error_rate, adjustment_factor)
        else:
            raise ValueError(f"{self.method} binning method not supported, please check")
        self.local_only = local_only

    def set_transform_method(self, transform_method):
        self._bin_obj.transform_method = transform_method

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        logger.info("Enter HeteroBinning fit.")
        y_vals = train_data.label.unique()
        if len(y_vals) > 2:
            raise ValueError(f"More than 2 classes found in label column. "
                             f"HeteroBinning currently only supports binary data. Please check.")
        self._bin_obj.fit(ctx, train_data)

    def compute_metrics(self, ctx: Context, binned_data):
        label_tensor = binned_data.label.as_tensor()
        self._bin_obj.compute_metrics(binned_data, label_tensor)
        if not self.local_only:
            self.compute_federated_metrics(ctx, binned_data)

    def compute_federated_metrics(self, ctx: Context, binned_data):
        logger.info(f"Start computing federated metrics.")
        encryptor, decryptor = ctx.cipher.phe.keygen(options=dict(key_length=2048))
        label_tensor = binned_data.label.as_tensor()
        encryptor.encrypt(label_tensor).to(ctx.hosts, "enc_y")
        host_col_bin = ctx.hosts.get("anonymous_col_bin")
        host_event_non_event_count = ctx.hosts.get("event_non_event_count")
        for i, (col_bin_list, en_host_count_res) in enumerate(zip(host_col_bin, host_event_non_event_count)):
            host_event_count_hist = en_host_count_res[0].decrypt()
            host_non_event_count_hist = en_host_count_res[1].decrypt()
            summary_metrics, _ = self._bin_obj.compute_all_col_metrics(host_event_count_hist,
                                                                       host_non_event_count_hist)
            self._bin_obj.set_host_metrics(ctx.hosts[i], summary_metrics)

    def transform(self, ctx: Context, test_data):
        transformed_data = self._bin_obj.transform(ctx, test_data)
        return transformed_data

    def to_model(self):
        bin_info = self._bin_obj.to_model()
        return dict(bin_info=bin_info, method=self.method)

    def restore(self, model):
        self._bin_obj.restore(model)

    @classmethod
    def from_model(cls, model) -> "HeteroBinningModuleGuest":
        bin_obj = HeteroBinningModuleGuest(model["method"])
        bin_obj.restore(model)
        return bin_obj


class HeteroBinningModuleHost(Module):
    def __init__(self, method="quantile", n_bins=10, bin_col=None, split_pt_dict=None, transform_method=None,
                 category_col=None, local_only=False, error_rate=1e-3, adjustment_factor=0.5):
        self.method = method
        self._federation_bin_obj = None
        if self.method in ["quantile", "bucket", "manual"]:
            self._bin_obj = StandardBinning(method, n_bins, split_pt_dict, bin_col, transform_method,
                                            category_col, error_rate, adjustment_factor)
        self.local_only = local_only
        self.bin_col = bin_col
        self.category_col = category_col

    def set_transform_method(self, new_transform_method):
        self._bin_obj.transform_method = new_transform_method

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        logger.info("Enter HeteroBinning fit.")
        self._bin_obj.fit(ctx, train_data)

    def compute_metrics(self, ctx: Context, binned_data):
        if not self.local_only:
            self.compute_federated_metrics(ctx, binned_data)

    def compute_federated_metrics(self, ctx: Context, binned_data):
        logger.info(f"Start computing federated metrics.")
        anonymous_col_bin = [binned_data.schema.anonymous_header[binned_data.schema.columns.index(col)]
                             for col in self.bin_col]
        ctx.guest.put("anonymous_col_bin", anonymous_col_bin)
        encrypt_y = ctx.guest.get("enc_y")
        # event count:
        to_compute_col = self.bin_col + self.category_col
        event_count_hist = binned_data.histogram2D(columns=to_compute_col, target=encrypt_y)
        # bin count(entries per bin):
        bin_count = binned_data.histogram2D(columns=to_compute_col, target=torch.tensor([1] * binned_data.count()))
        non_event_count_hist = bin_count - event_count_hist
        ctx.guest.put("event_non_event_count", (event_count_hist, non_event_count_hist))

    def transform(self, ctx: Context, test_data):
        return self._bin_obj.transform(ctx, test_data)

    def to_model(self):
        bin_info = self._bin_obj.to_model()
        return dict(bin_info=bin_info, method=self.method)

    def restore(self, model):
        self._bin_obj.restore(model)

    @classmethod
    def from_model(cls, model) -> "HeteroBinningModuleHost":
        bin_obj = HeteroBinningModuleHost(method=model["method"])
        bin_obj.restore(model)
        return bin_obj


class StandardBinning(Module):
    def __init__(self, method, n_bins, split_pt_dict, bin_col, transform_method,
                 category_col, error_rate, adjustment_factor):
        self.method = method
        self.n_bins = n_bins
        # cols to be binned
        self.bin_col = bin_col
        # cols to be treated as categorical feature
        self.category_col = category_col
        self.transform_method = transform_method
        self.error_rate = error_rate
        self.adjustment_factor = adjustment_factor
        self._manual_split_pt_dict = split_pt_dict
        # {col_name: [split_pts]}, ordered by split_pts
        self._split_pt_dict = None
        # {col_name: [bin_num]}, ordered by split_pts
        self._bin_idx_dict = None
        self._bin_count_dict = None
        # {col_name: [woe]}, ordered by split_pts, for transform
        self._woe_dict = None
        # for prediction transform
        self._train_woe_dict = None
        # {col_name: {"iv_array": [], "woe_array": [], "event_count": []...}}
        self._metrics_summary = None
        self._host_metrics_summary = None
        self._train_metrics_summary = None
        self._host_train_metrics_summary = None

    def set_host_metrics(self, host, metrics_summary):
        self._host_metrics_summary[host] = metrics_summary

    def fit(self, ctx: Context, train_data, validate_data=None, skip_none=False):
        # only bin given `col_bin` cols
        if self.bin_col is None:
            self.bin_col = train_data.schema.columns

        if self.method == "quantile":
            q = np.arange(0, 1, 1 / self.n_bins)
            # maybe api will return df/special tensor
            split_pt_df = train_data.quantile(q=q,
                                              column=self.bin_col,
                                              error_rate=self.error_rate,
                                              skip_none=skip_none)
            """split_pt_dict = {}
            for col in split_pt_df.schema.columns:
                split_pt_dict[col] = list(split_pt_df[col])
            self._split_pt_dict = split_pt_dict"""
            # pd.DataFrame
            # @todo: maybe convert to format {col_name: List[split_pt0, split_pt1]}
            # self._split_pt_dict = split_pt_df.to_dict()
        elif self.method == "bucket":
            split_pt_df = train_data.bucket_split(q=self.n_bins, column=self.bin_col, skip_none=skip_none)

            """split_pt_dict = {}
            for col in split_pt_df.schema.columns:
                split_pt_dict[col] = split_pt_df[col]
            self._split_pt_dict = split_pt_dict"""
            # self._split_pt_dict = split_pt_df.to_dict()
        elif self.method == "manual":
            # self._split_pt_dict = self._manual_split_pt_dict
            split_pt_df = pd.DataFrame.from_dict(self._manual_split_pt_dict)
        else:
            raise ValueError(f"Unknown binning method {self.method} encountered. Please check")
        self._split_pt_dict = split_pt_df.to_dict()

        def __get_col_bin_count(col):
            count = len(col.unique())
            return count

        bin_count = split_pt_df.apply(__get_col_bin_count, axis=0)

        if not skip_none:
            # add extra bin for columns with nan
            bin_count = bin_count + (train_data.nan_count() > 0)
        self._bin_count_dict = bin_count.to_dict()

        """self._split_pt_dict = split_pt_df.to_dict()
        for col_name in self.bin_col:
            bin_count = len(self._split_pt_dict[col_name])
            self._bin_idx_dict[col_name] = list(range(bin_count))
            if not skip_none:
                if train_data[col_name].nan_count() > 0:
                    bin_count += 1
            self._bin_count_dict[col_name] = bin_count"""

    def bucketize_data(self, train_data):
        binned_df = train_data.bucketize(n_bins=self._split_pt_dict, col=self.bin_col, bucket_none=True)
        return binned_df

    """@staticmethod
    def is_monotonic(woe_array):
        # Check the woe is monotonic or not
        
        if len(woe_array) <= 1:
            return torch.tensor([True])

        is_increasing = torch.all(woe_array[1:] > woe_array[:-1])
        is_decreasing = torch.all(woe_array[1:] < woe_array[:-1])
        return is_increasing or is_decreasing"""

    @staticmethod
    def is_monotonic(woe_array):
        """
        Check the woe is monotonic or not
        """
        if len(woe_array) <= 1:
            return True

        is_increasing = all(woe_array[1:] > woe_array[:-1])
        is_decreasing = all(woe_array[1:] < woe_array[:-1])
        return is_increasing or is_decreasing

    def compute_metrics_from_count(self, event_count_array, non_event_count_array,
                                   total_event_count, total_non_event_count):
        event_rate = (event_count_array == 0) * self.adjustment_factor + event_count_array / total_event_count
        non_event_rate = (non_event_count_array == 0) * self.adjustment_factor + non_event_count_array \
                         / total_non_event_count
        bin_woe = torch.log(event_rate / non_event_rate)
        bin_iv = (event_rate - non_event_rate) * bin_woe
        return event_rate, non_event_rate, bin_woe, bin_iv

    """def compute_all_col_metrics(self, col_list, event_count_hist, non_event_count_hist):
        total_event_count, total_non_event_count = None, None
        metrics_summary = {}
        woe_dict = {}
        for col in col_list:
            col_summary = {}
            event_count_array = event_count_hist[col].as_tensor()
            non_event_count_array = non_event_count_hist[col].as_tensor()
            if total_event_count is None:
                total_event_count = torch.max(torch.sum(event_count_array), torch.tensor(1))
                total_non_event_count = torch.max(torch.sum(non_event_count_array), torch.tensor(1))

            event_rate, non_event_rate, bin_woe, bin_iv = self.compute_metrics_from_count(event_count_array,
                                                                                          non_event_count_array,
                                                                                          total_event_count,
                                                                                          total_non_event_count)
            col_summary["event_count"] = event_count_array.tolist()
            col_summary["non_event_count"] = non_event_count_array.tolist()
            col_summary["event_rate"] = event_rate.tolist()
            col_summary["non_event_rate"] = non_event_rate.tolist()
            col_summary["woe_array"] = bin_woe.tolist()
            col_summary["iv_array"] = bin_iv.tolist()
            col_summary["is_monotonic"] = StandardBinning.is_monotonic(bin_woe).tolist()
            col_summary["col_iv"] = bin_iv.sum().tolist()
            metrics_summary[col] = col_summary
            woe_dict[col] = bin_woe.tolist()
        return metrics_summary, woe_dict"""

    def compute_all_col_metrics(self, event_count_hist, non_event_count_hist):
        # pd.DataFrame ver
        total_event_count, total_non_event_count = event_count_hist.sum(), non_event_count_hist.sum()
        total_non_event_count[total_event_count < 1] = 1
        total_non_event_count[total_non_event_count < 1] = 1
        event_rate = (event_count_hist == 0) * self.adjustment_factor + event_count_hist / total_event_count
        non_event_rate = (non_event_count_hist == 0) * self.adjustment_factor + non_event_count_hist \
                         / total_non_event_count
        rate_ratio = event_rate / non_event_rate
        bin_woe = rate_ratio.apply(lambda v: np.log(v))
        bin_iv = (event_rate - non_event_rate) * bin_woe

        metrics_summary = {}

        metrics_summary["event_count"] = event_count_hist.to_dict()
        metrics_summary["non_event_count"] = non_event_count_hist.to_dict()
        metrics_summary["event_rate"] = event_rate.to_dict()
        metrics_summary["non_event_rate"] = non_event_rate.to_dictt()
        metrics_summary["woe_array"] = bin_woe.to_dict()
        metrics_summary["iv_array"] = bin_iv.to_dict()
        metrics_summary["is_monotonic"] = bin_woe.apply(StandardBinning.is_monotonic, axis=0).to_dict()
        metrics_summary["col_iv"] = bin_iv.sum().to_dict()
        # @todo: maybe convert to {col_name: List[woe_0, woe_1]}
        woe_dict = bin_woe.to_dict()
        return metrics_summary, woe_dict

    def compute_metrics(self, binned_data, label_col):
        to_compute_col = self.bin_col + self.category_col
        event_count_hist = binned_data.histogram2D(columns=to_compute_col, target=label_col)
        bin_count_hist = binned_data.histogram2D(columns=to_compute_col,
                                                 target=torch.tensor([[1]] * binned_data.count()))
        non_event_count_hist = bin_count_hist - event_count_hist
        self._metrics_summary, self._woe_dict = self.compute_all_col_metrics(event_count_hist,
                                                                             non_event_count_hist)

    def transform(self, ctx: Context, binned_data):
        logger.debug(f"Given transform method: {self.transform_method}.")
        if self.transform_method == "bin_idx" and self._bin_idx_dict:
            return binned_data
        elif self.transform_method == "woe":
            # predict: replace with woe from train phase
            if self._train_woe_dict:
                logger.debug(f"`train_woe_dict` provided, will transform to woe values from training phase.")
                return binned_data.replace(self._train_woe_dict, self.bin_col)
            elif self._woe_dict:
                return binned_data.replace(self._woe_dict, self.bin_col)
        else:
            logger.warning(f"to transform type {self.transform_method} encountered, but no bin tag dict provided. "
                           f"Please check")
        return binned_data

    def to_model(self):
        return dict(
            method=self.method,
            split_pt_dict=json.dumps(self._split_pt_dict),
            bin_idx_dict=json.dumps(self._bin_idx_dict),
            bin_count=json.dumps(self._bin_count),
            metrics_summary=json.dumps(self._metrics_summary),
            host_metrics_summary=json.dumps(self._host_metrics_summary),
            woe_dict=json.dumps(self._woe_dict),
            category_col=self.category_col,
            adjustment_factor=self.adjustment_factor
        )

    def restore(self, model):
        self.method = model["method"]
        self._split_pt_dict = json.loads(model["split_pt_dict"])
        self._bin_idx_dict = json.loads(model["bin_idx_dict"])
        self._bin_count = json.load(model["bin_count"])
        self._train_metrics_summary = json.loads(model["metrics_summary"])
        self._train_woe_dict = json.loads(model["woe_dict"])
        self._train_host_metrics_summary = json.loads(model["host_train_metrics_summary"])
        self.category_col = model["category_col"]
        self.adjustment_factor = model["adjustment_factor"]
