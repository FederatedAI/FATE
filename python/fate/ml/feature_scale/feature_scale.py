import json
import logging
import pandas as pd

from fate.interface import Context

from ..abc.module import Module

logger = logging.getLogger(__name__)


class FeatureScale(Module):
    def __init__(
        self,
        method="standard"
    ):
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
        return dict(
            scaler_info=scaler_info,
            method=self.method
        )

    def restore(self, model):
        self._scaler.from_model(model)

    @classmethod
    def from_model(cls, model) -> 'FeatureScale':
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
        return test_data - self._mean
        # return (test_data - self._mean) / self._std

    def to_model(self):
        return dict(
            mean=self._mean.to_json(),
            std=self._std.to_json()
        )

    def from_model(self, model):
        self._mean = pd.Series(json.loads(model["mean"]))
        self._std = pd.Series(json.loads(model["std"]))
