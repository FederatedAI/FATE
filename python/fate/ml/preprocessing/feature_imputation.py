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

from fate.interface import Context
from ..abc.module import Module

logger = logging.getLogger(__name__)


class FeatureImputation(Module):
    def __init__(self,
                 imputation_col=None,
                 col_missing_fill_method=None,
                 missing_value=None,
                 designated_fill_value=None):
        self._imputer = Imputer(imputation_col, col_missing_fill_method, missing_value, designated_fill_value)

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        self._imputer.fit(ctx, train_data)

    def transform(self, ctx: Context, test_data):
        return self._imputer.transform(ctx, test_data)

    def to_model(self):
        imputer_info = self._imputer.to_model()
        return dict(imputer_info=imputer_info)

    def restore(self, model):
        self._imputer.from_model(model)

    @classmethod
    def from_model(cls, model) -> "FeatureImputation":
        scaler = FeatureImputation()
        scaler.restore(model["imputer_info"])
        return scaler


class Imputer(Module):
    def __init__(self,
                 select_col=None,
                 col_missing_fill_method=None,
                 missing_value=None,
                 designated_fill_value=None):
        self._missing_fill_value = None
        self._mean = None
        self._median = None
        self._min = None
        self._max = None

        self.designated_fill_value = designated_fill_value
        self.col_missing_fill_method = col_missing_fill_method
        self.missing_value = set(missing_value)
        self.select_col = select_col

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        if self.select_col is None:
            self.select_col = train_data.schema.columns.to_list()
        train_data_select = train_data[self.select_col]
        if self.missing_value:
            mask = train_data_select.isin(self.missing_value)
            train_data_select = train_data_select[~mask]
        if self.col_missing_fill_method is None:
            self.col_missing_fill_method = {col: "designated" for col in self.select_col}
            self._missing_fill_value = {col: self.designated_fill_value for col in self.select_col}
        else:
            fill_method = set(self.col_missing_fill_method.values())
            for method in fill_method:
                if method == "min":
                    self._min = train_data_select.nanmin()
                elif method == "max":
                    self._max = train_data_select.nanmax()
                elif method == "mean":
                    self._mean = train_data_select.nanmean()
                elif method == "median":
                    self._median = train_data_select.nanmedian()

            missing_fill_value = {}
            for col in self.select_col:
                if col not in self.col_missing_fill_method:
                    self.col_missing_fill_method[col] = "designated"
                    missing_fill_value[col] = 0
                elif self.col_missing_fill_method[col] == "designated":
                    missing_fill_value[col] = self.designated_fill_value[col]
                elif self.col_missing_fill_method[col] == "mean":
                    missing_fill_value[col] = self._mean[col]
                elif self.col_missing_fill_method[col] == "median":
                    missing_fill_value[col] = self._median[col]
                elif self.col_missing_fill_method[col] == "min":
                    missing_fill_value[col] = self._min[col]
                elif self.col_missing_fill_method[col] == "max":
                    missing_fill_value[col] = self._max[col]
                else:
                    raise ValueError(f"col_missing_fill_method {self.col_missing_fill_method[col]} is not supported")
            self._missing_fill_value = missing_fill_value

    def transform(self, ctx: Context, test_data):
        test_data_select = test_data[self.select_col]
        if self.missing_value:
            mask = test_data_select.isin(self.missing_value)
            test_data_select = test_data_select[~mask]
        test_data[self.select_col] = test_data_select.fillna(self._missing_fill_value)
        return test_data

    def to_model(self):
        return dict(
            select_col=self.select_col,
            col_missing_fill_method=self.col_missing_fill_method,
            missing_fill_value=self._missing_fill_value,
            missing_value=list(self.missing_value)
        )

    def from_model(self, model):
        self._missing_fill_value = model["missing_fill_value"]
        self.col_missing_fill_method = model["col_missing_fill_method"]
        self.select_col = model["select_col"]
        self.missing_value = set(model["missing_value"])
