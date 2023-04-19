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

from fate.interface import Context
from ..abc.module import Module

logger = logging.getLogger(__name__)


class FeatureScale(Module):
    def __init__(self, method="standard", scale_col=None, feature_range=None):
        self.method = method
        self._scaler = None
        if self.method == "standard":
            self._scaler = StandardScaler(scale_col)
        elif self.method == "min_max":
            self._scaler = MinMaxScaler(scale_col, feature_range)
        else:
            raise ValueError(f"Unknown scale method {self.method} given. Please check.")

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        self._scaler.fit(ctx, train_data)

    def transform(self, ctx: Context, test_data):
        return self._scaler.transform(ctx, test_data)

    def to_model(self):
        scaler_info = self._scaler.to_model()
        return dict(scaler_info=scaler_info, method=self.method)

    def restore(self, model):
        self._scaler.from_model(model)

    @classmethod
    def from_model(cls, model) -> "FeatureScale":
        scaler = FeatureScale(model["method"])
        scaler.restore(model["scaler_info"])
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
            select_col=self.select_col
        )

    def from_model(self, model):
        self._mean = pd.Series(model["mean"], dtype=model["mean_dtype"])
        self._std = pd.Series(model["std"], dtype=model["std_dtype"])
        self.select_col = model["select_col"]


class MinMaxScaler(Module):
    """
       Transform given data by scaling features to given range.
       Adapted from sklearn.preprocessing.minmax_scale

       The transformation is given by::
           X_std = (X - X.min()) / (X.max() - X.min())
           X_scaled = X_std * (range_max - range_min) + range_min
    """

    def __init__(self, select_col, feature_range):
        self.feature_range = feature_range
        self.select_col = select_col
        self._scale = None
        self._scale_min = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        train_data_select = train_data[self.select_col]
        data_max = train_data_select.max()
        data_min = train_data_select.min()
        min_list, max_list = [], []

        # min/max values in the same order as schema.header
        for col in train_data_select.schema.columns:
            if col in self.feature_range:
                min_list.append(self.feature_range[col][0])
                max_list.append(self.feature_range[col][1])
        range_min = pd.Series(min_list, index=train_data_select.schema.columns)
        range_max = pd.Series(max_list, index=train_data_select.schema.columns)

        data_range = data_max - data_min
        # for safe division
        data_range[data_range < 1e-6] = 1.0
        self._scale = (range_max - range_min) / data_range
        self._scale_min = range_min - data_min * self._scale

    def transform(self, ctx: Context, test_data):
        """
        Transformation is given by:
            X_scaled = scale * X + range_min - X_train.min() * scale
        where scale = (range_max - range_min) / (X_train.max() - X_train.min()

        """
        test_data_select = test_data[self.select_col]
        data_scaled = test_data_select * self._scale + self._scale_min
        test_data[self.select_col] = data_scaled
        return test_data

    def to_model(self):
        return dict(
            scale=self._scale.to_dict(),
            scale_dtype=self._scale.dtype.name,
            scale_min=self._scale_min.to_dict(),
            scale_min_dtype=self._scale_min.dtype.name,
            select_col=self.select_col
        )

    def from_model(self, model):
        self._scale = pd.Series(model["scale"], dtype=model["scale_dtype"])
        self._scale_min = pd.Series(model["scale_min"], dtype=model["scale_min_dtype"])
        self.select_col = model["select_col"]
