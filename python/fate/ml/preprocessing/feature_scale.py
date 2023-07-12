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

import pandas as pd

from fate.arch import Context
from ..abc.module import Module

logger = logging.getLogger(__name__)


class FeatureScale(Module):
    def __init__(self, method="standard", scale_col=None, feature_range=None, strict_range=True):
        self.method = method
        self._scaler = None
        if self.method == "standard":
            self._scaler = StandardScaler(scale_col)
        elif self.method == "min_max":
            self._scaler = MinMaxScaler(scale_col, feature_range, strict_range)
        else:
            raise ValueError(f"Unknown scale method {self.method} given. Please check.")

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        self._scaler.fit(ctx, train_data)

    def transform(self, ctx: Context, test_data):
        return self._scaler.transform(ctx, test_data)

    def get_model(self):
        scaler_info = self._scaler.to_model()
        model_data = dict(scaler_info=scaler_info)
        return {"data": model_data, "meta": {"method": self.method,
                                             "model_type": "feature_scale"}}

    def restore(self, model):
        self._scaler.from_model(model)

    @classmethod
    def from_model(cls, model) -> "FeatureScale":
        scaler = FeatureScale(model["meta"]["method"])
        scaler.restore(model["data"]["scaler_info"])
        return scaler


class StandardScaler(Module):
    def __init__(self, select_col):
        self._mean = None
        self._std = None
        self.select_col = select_col

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        if self.select_col is None:
            self.select_col = train_data.schema.columns.to_list()
        train_data_select = train_data[self.select_col]
        self._mean = train_data_select.mean()
        self._std = train_data_select.std()

    def transform(self, ctx: Context, test_data):
        test_data_select = test_data[self.select_col]
        test_data[self.select_col] = (test_data_select - self._mean) / self._std
        return test_data

    def to_model(self):
        return dict(
            mean=self._mean.to_dict(),
            mean_dtype=self._mean.dtype.name,
            std=self._std.to_dict(),
            std_dtype=self._std.dtype.name,
            select_col=self.select_col,
        )

    def from_model(self, model):
        self._mean = pd.Series(model["mean"], dtype=model["mean_dtype"])
        self._std = pd.Series(model["std"], dtype=model["std_dtype"])
        self.select_col = model["select_col"]


class MinMaxScaler(Module):
    """
    Transform data by scaling features to given feature range.
    Note that if `strict_range` is set, transformed values will always be within given range,
    regardless whether transform data exceeds training data value range (as in 1.x ver)

    The transformation is given by::
        X_scaled = (X - X.min()) / (X.max() - X.min()) * feature_range + feature_range_min
                 = (X - X.min()) * (feature_range / (X.max() - X.min()) + feature_range_min
    """

    def __init__(self, select_col, feature_range, strict_range):
        self.feature_range = feature_range
        self.select_col = select_col
        self.strict_range = strict_range
        self._scale = None
        self._scale_min = None
        self._range_min = None
        self._range_max = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        if self.select_col is None:
            self.select_col = train_data.schema.columns.to_list()
        train_data_select = train_data[self.select_col]
        data_max = train_data_select.max()
        data_min = train_data_select.min()

        # select_col has same keys as feature_range
        self._range_min = pd.Series({col: self.feature_range[col][0] for col in self.select_col})
        self._range_max = pd.Series({col: self.feature_range[col][1] for col in self.select_col})

        data_range = data_max - data_min
        # for safe division
        data_range[data_range < 1e-6] = 1.0
        self._scale = (self._range_max - self._range_min) / data_range
        self._scale_min = data_min * self._scale

    def transform(self, ctx: Context, test_data):
        """
        Transformation is given by:
            X_scaled = (X * scale - scale_min) + feature_range_min
        where scale = feature_range / (X_train.max() - X_train.min()) and scale_min = X_train.min() * scale

        """
        test_data_select = test_data[self.select_col]

        data_scaled = test_data_select * self._scale - (self._scale_min + self._range_min)
        if self.strict_range:
            # restrict feature output within given feature value range
            data_scaled = data_scaled[data_scaled >= self._range_min].fillna(self._range_min)
            data_scaled = data_scaled[data_scaled <= self._range_max].fillna(self._range_max)
        test_data[self.select_col] = data_scaled
        return test_data

    def to_model(self):
        return dict(
            scale=self._scale.to_dict(),
            scale_dtype=self._scale.dtype.name,
            scale_min=self._scale_min.to_dict(),
            scale_min_dtype=self._scale_min.dtype.name,
            range_min=self._range_min.to_dict(),
            range_min_dtype=self._range_min.dtype.name,
            range_max=self._range_max.to_dict(),
            range_max_dtype=self._range_max.dtype.name,
            strict_range=self.strict_range,
            select_col=self.select_col,
        )

    def from_model(self, model):
        self._scale = pd.Series(model["scale"], dtype=model["scale_dtype"])
        self._scale_min = pd.Series(model["scale_min"], dtype=model["scale_min_dtype"])
        self._range_min = pd.Series(model["range_min"], dtype=model["range_min_dtype"])
        self._range_max = pd.Series(model["range_max"], dtype=model["range_max_dtype"])
        self.strict_range = model["strict_range"]
        self.select_col = model["select_col"]
