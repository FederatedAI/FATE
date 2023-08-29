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

from fate.arch import Context
from ..abc.module import Module

logger = logging.getLogger(__name__)


class OneHotEncoder(Module):
    def __init__(self, drop=None, handle_unknown="error", encode_col=None):
        self.drop = drop
        self.handle_unknown = handle_unknown
        self.encode_col = encode_col
        # @todo: use different encoder for sync(homo) vs. async(local)
        self._encoder = LocalOneHotEncoder(self.drop, self.handle_unknown, self.encode_col)

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        self._encoder.fit(ctx, train_data)

    def transform(self, ctx: Context, test_data):
        return self._encoder.transform(ctx, test_data)

    def get_model(self):
        encoder_info = self._encoder.to_model()
        return {"data": encoder_info,
                "meta": {"drop": self.drop,
                         "handle_unknown": self.handle_unknown,
                         "encode_col": self.encode_col,
                         "model_type": "one_hot_encoder"}
                }

    def restore(self, model):
        self._encoder.from_model(model)

    @classmethod
    def from_model(cls, model) -> "OneHotEncoder":
        encoder = OneHotEncoder(model["meta"]["drop"],
                                model["meta"]["handle_unknown"],
                                model["meta"]["encode_col"])
        encoder.restore(model["data"])
        return encoder


class LocalOneHotEncoder(Module):
    def __init__(self, drop, handle_unknown, encode_col):
        self.encode_col = encode_col
        self.raise_if_unknown = handle_unknown == "error"
        self.drop_first = drop == "first"
        self.drop_if_binary = drop == "if_binary"

        self._column_encode_map = {}

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        if self.encode_col is None:
            self.encode_col = train_data.schema.columns.to_list()

    def transform(self, ctx: Context, test_data):
        encode_col = set(self.encode_col)
        non_transform_col = [col for col in test_data.schema.columns if col not in encode_col]
        if len(non_transform_col) > 0:
            result_data = test_data[non_transform_col]
        else:
            result_data = test_data.create_frame(with_label=True, with_weight=True)
        for col in self.encode_col:
            encoded_col = test_data[col].get_dummies()
            n_category = encoded_col.shape[1]
            if self.drop_if_binary and n_category == 2:
                keep_col = encoded_col.schema.columns[1]
                encoded_col = encoded_col[keep_col]
            if self.drop_first:
                if n_category == 1:
                    continue
                keep_col = encoded_col.schema.columns[1:]
                encoded_col = encoded_col[keep_col]
            stored_category_col = self._column_encode_map.get(col)
            if stored_category_col:
                if set(encoded_col.schema.columns).issubset(set(stored_category_col)):
                    if self.raise_if_unknown:
                        raise ValueError(f"unknown categories found in {encoded_col.schema.columns}, "
                                         f"but not in {stored_category_col}")
            else:
                self._column_encode_map[col] = encoded_col.columns.to_list()
            result_data[encoded_col.schema.columns] = encoded_col
        return result_data

    def to_model(self):
        return dict(
            column_encode_map=self._column_encode_map
        )

    def from_model(self, model):
        self._column_encode_map = model["column_encode_map"]
