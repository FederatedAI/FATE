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
import json
import logging

import pandas as pd
from fate.interface import Context

from ..abc.module import Module

logger = logging.getLogger(__name__)


class FeatureScale(Module):
    def __init__(self, method="standard"):
        self.method = method
        self._scaler = None
        if self.method == "standard":
            self._scaler = StandardScaler()

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
    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        self._mean = train_data.mean()
        self._std = train_data.std()

    def transform(self, ctx: Context, test_data):
        return (test_data - self._mean) / self._std

    def to_model(self):
        return dict(
            mean=self._mean.to_json(),
            mean_dtype=self._mean.dtype.name,
            std=self._std.to_json(),
            std_dtype=self._std.dtype.name,
        )

    def from_model(self, model):
        self._mean = pd.Series(json.loads(model["mean"]), dtype=model["mean_dtype"])
        self._std = pd.Series(json.loads(model["std"]), dtype=model["std_dtype"])
