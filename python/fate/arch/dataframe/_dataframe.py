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
import copy
import operator
import typing
from typing import List, Union

import numpy as np
import pandas as pd

from fate.arch.tensor import DTensor
from .manager import DataManager, Schema
from fate.arch.trace import auto_trace

if typing.TYPE_CHECKING:
    from fate.arch.histogram import DistributedHistogram, HistogramBuilder


class DataFrame(object):
    def __init__(self, ctx, block_table, partition_order_mappings, data_manager: DataManager):
        self._ctx = ctx
        self._block_table = block_table
        self._partition_order_mappings = partition_order_mappings
        self._data_manager = data_manager

        self._sample_id_indexer = None
        self._match_id_indexer = None
        self._sample_id = None
        self._match_id = None
        self._label = None
        self._weight = None

        self._count = None
        self._columns = None

    @property
    def sample_id(self):
        if self._sample_id is None:
            self._sample_id = self.__extract_fields(
                with_sample_id=True, with_match_id=False, with_label=False, with_weight=False
            )
        return self._sample_id

    @property
    def match_id(self):
        if self._match_id is None:
            self._match_id = self.__extract_fields(
                with_sample_id=False, with_match_id=True, with_label=False, with_weight=False
            )

        return self._match_id

    @property
    def values(self):
        """
        as values maybe bigger than match_id/sample_id/weight/label, we will not cached them
        """
        if not len(self.schema.columns):
            return None

        return self.__extract_fields(
            with_sample_id=True, with_match_id=True, with_label=False, with_weight=False, columns=self.columns.tolist()
        )

    @property
    @auto_trace
    def label(self):
        if not self.schema.label_name:
            return None

        if self._label is None:
            self._label = self.__extract_fields(
                with_sample_id=True, with_match_id=True, with_label=True, with_weight=False
            )

        return self._label

    @property
    def weight(self):
        if not self.schema.weight_name:
            return None

        if self._weight is None:
            self._weight = self.__extract_fields(
                with_sample_id=True, with_match_id=True, with_label=False, with_weight=True
            )

        return self._weight

    @property
    def shape(self) -> "tuple":
        if self._count is None:
            items = 0
            for _, v in self._partition_order_mappings.items():
                items += v["end_index"] - v["start_index"] + 1
            self._count = items

        return self._count, len(self._data_manager.schema.columns)

    @property
    def schema(self) -> "Schema":
        return self._data_manager.schema

    @property
    def columns(self):
        return self.schema.columns

    @property
    def block_table(self):
        return self._block_table

    @block_table.setter
    def block_table(self, block_table):
        self._block_table = block_table

    @property
    def partition_order_mappings(self):
        return self._partition_order_mappings

    @property
    def data_manager(self) -> "DataManager":
        return self._data_manager

    @data_manager.setter
    def data_manager(self, data_manager):
        self._data_manager = data_manager

    @property
    def dtypes(self):
        return self._data_manager.dtypes

    @auto_trace
    def as_tensor(self, dtype=None):
        """
        df.weight.as_tensor()
        df.label.as_tensor()
        df.values.as_tensor()
        """
        from .ops._transformer import transform_to_tensor

        return transform_to_tensor(
            self._block_table, self._data_manager, dtype, partition_order_mappings=self.partition_order_mappings
        )

    def as_pd_df(self) -> "pd.DataFrame":
        from .ops._transformer import transform_to_pandas_dataframe

        return transform_to_pandas_dataframe(self._block_table, self._data_manager)

    @auto_trace
    def apply_row(self, func, columns=None, with_label=False, with_weight=False, enable_type_align_checking=False):
        from .ops._apply_row import apply_row

        return apply_row(
            self,
            func,
            columns=columns,
            with_label=with_label,
            with_weight=with_weight,
            enable_type_align_checking=enable_type_align_checking,
        )

    @auto_trace
    def create_frame(self, with_label=False, with_weight=False, columns: Union[list, pd.Index] = None) -> "DataFrame":
        if columns is not None and isinstance(columns, pd.Index):
            columns = columns.tolist()

        return self.__extract_fields(
            with_sample_id=True, with_match_id=True, with_label=with_label, with_weight=with_weight, columns=columns
        )

    @auto_trace
    def empty_frame(self) -> "DataFrame":
        return DataFrame(
            self._ctx,
            self._ctx.computing.parallelize([], include_key=False, partition=self._block_table.num_partitions),
            partition_order_mappings=dict(),
            data_manager=self._data_manager.duplicate(),
        )

    @auto_trace
    def drop(self, index) -> "DataFrame":
        from .ops._dimension_scaling import drop

        return drop(self, index)

    @auto_trace
    def fillna(self, value):
        from .ops._fillna import fillna

        return fillna(self, value)

    @auto_trace
    def get_dummies(self, dtype="int32"):
        from .ops._encoder import get_dummies

        return get_dummies(self, dtype=dtype)

    @auto_trace
    def isna(self):
        from .ops._missing import isna

        return isna(self)

    @auto_trace
    def isin(self, values):
        from .ops._isin import isin

        return isin(self, values)

    @auto_trace
    def na_count(self):
        return self.isna().sum()

    @auto_trace
    def max(self) -> "pd.Series":
        from .ops._stat import max

        return max(self)

    @auto_trace
    def min(self, *args, **kwargs) -> "pd.Series":
        from .ops._stat import min

        return min(self)

    @auto_trace
    def mean(self, *args, **kwargs) -> "pd.Series":
        from .ops._stat import mean

        return mean(self)

    @auto_trace
    def sum(self, *args, **kwargs) -> "pd.Series":
        from .ops._stat import sum

        return sum(self)

    @auto_trace
    def std(self, ddof=1, **kwargs) -> "pd.Series":
        from .ops._stat import std

        return std(self, ddof=ddof)

    @auto_trace
    def var(self, ddof=1, **kwargs):
        from .ops._stat import var

        return var(self, ddof=ddof)

    @auto_trace
    def variation(self, ddof=1):
        from .ops._stat import variation

        return variation(self, ddof=ddof)

    @auto_trace
    def skew(self, unbiased=False):
        from .ops._stat import skew

        return skew(self, unbiased=unbiased)

    @auto_trace
    def kurt(self, unbiased=False):
        from .ops._stat import kurt

        return kurt(self, unbiased=unbiased)

    @auto_trace
    def sigmoid(self) -> "DataFrame":
        from .ops._activation import sigmoid

        return sigmoid(self)

    @auto_trace
    def rename(
        self,
        sample_id_name: str = None,
        match_id_name: str = None,
        label_name: str = None,
        weight_name: str = None,
        columns: dict = None,
    ):
        self._data_manager.rename(
            sample_id_name=sample_id_name,
            match_id_name=match_id_name,
            label_name=label_name,
            weight_name=weight_name,
            columns=columns,
        )

    @auto_trace
    def count(self) -> "int":
        return self.shape[0]

    @auto_trace
    def describe(self, ddof=1, unbiased=False):
        from .ops._stat import describe

        return describe(self, ddof=ddof, unbiased=unbiased)

    @auto_trace
    def quantile(self, q, relative_error: float = 1e-4):
        from .ops._quantile import quantile

        return quantile(self, q, relative_error)

    @auto_trace
    def qcut(self, q: int):
        from .ops._quantile import qcut

        return qcut(self, q)

    @auto_trace
    def bucketize(self, boundaries: Union[dict, pd.DataFrame]) -> "DataFrame":
        from .ops._encoder import bucketize

        return bucketize(self, boundaries)

    @auto_trace
    def distributed_hist_stat(
        self,
        histogram_builder: "HistogramBuilder",
        position: "DataFrame" = None,
        targets: Union[dict, "DataFrame"] = None,
    ) -> "DistributedHistogram":
        from .ops._histogram import distributed_hist_stat

        if targets is None:
            raise ValueError("To use distributed hist stat, targets should not be None")
        if position is None:
            position = self.create_frame()
            position["node_idx"] = 0

        return distributed_hist_stat(self, histogram_builder, position, targets)

    @auto_trace
    def replace(self, to_replace=None) -> "DataFrame":
        from .ops._replace import replace

        return replace(self, to_replace)

    @auto_trace
    def __add__(self, other: Union[int, float, list, "np.ndarray", "DataFrame", "pd.Series"]) -> "DataFrame":
        return self.__arithmetic_operate(operator.add, other)

    @auto_trace
    def __radd__(self, other: Union[int, float, list, "np.ndarray", "pd.Series"]) -> "DataFrame":
        return self + other

    @auto_trace
    def __sub__(self, other: Union[int, float, list, "np.ndarray", "pd.Series"]) -> "DataFrame":
        return self.__arithmetic_operate(operator.sub, other)

    @auto_trace
    def __rsub__(self, other: Union[int, float, list, "np.ndarray", "pd.Series"]) -> "DataFrame":
        return self * (-1) + other

    @auto_trace
    def __mul__(self, other) -> "DataFrame":
        return self.__arithmetic_operate(operator.mul, other)

    @auto_trace
    def __rmul__(self, other) -> "DataFrame":
        return self * other

    @auto_trace
    def __truediv__(self, other) -> "DataFrame":
        return self.__arithmetic_operate(operator.truediv, other)

    @auto_trace
    def __pow__(self, power) -> "DataFrame":
        return self.__arithmetic_operate(operator.pow, power)

    @auto_trace
    def __lt__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.lt, other)

    @auto_trace
    def __le__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.le, other)

    @auto_trace
    def __gt__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.gt, other)

    @auto_trace
    def __ge__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.ge, other)

    @auto_trace
    def __eq__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.eq, other)

    @auto_trace
    def __ne__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.ne, other)

    @auto_trace
    def __invert__(self):
        from .ops._unary_operator import invert

        return invert(self)

    def __arithmetic_operate(self, op, other) -> "DataFrame":
        from .ops._arithmetic import arith_operate

        return arith_operate(self, other, op)

    def __cmp_operate(self, op, other) -> "DataFrame":
        from .ops._cmp import cmp_operate

        return cmp_operate(self, other, op)

    def __setattr__(self, key, value):
        property_attr_mapping = dict(block_table="_block_table", data_manager="_data_manager")
        if key not in ["label", "weight"] and key not in property_attr_mapping:
            self.__dict__[key] = value
            return

        if key in property_attr_mapping:
            self.__dict__[property_attr_mapping[key]] = value
            return

        if key == "label":
            if self._label is not None:
                self.__dict__["_label"] = None
            from .ops._set_item import set_label_or_weight

            set_label_or_weight(self, value, key_type=key)
        else:
            if self._weight is not None:
                self.__dict__["_weight"] = None
            from .ops._set_item import set_label_or_weight

            set_label_or_weight(self, value, key_type=key)

    def __getitem__(self, items) -> "DataFrame":
        if isinstance(items, DataFrame):
            from .ops._where import where

            return where(self, items)

        if isinstance(items, DTensor):
            from .ops._dimension_scaling import retrieval_row

            return retrieval_row(self, items)

        if isinstance(items, pd.Index):
            items = items.tolist()
        elif not isinstance(items, list):
            items = [items]

        for item in items:
            if item not in self._data_manager.schema.columns:
                raise ValueError(f"DataFrame does not has attribute {item}")

        return self.__extract_fields(with_sample_id=True, with_match_id=True, columns=items)

    def __setitem__(self, keys, items):
        if isinstance(keys, str):
            keys = [keys]
        elif isinstance(keys, pd.Series):
            keys = keys.tolist()

        state = 0
        column_set = set(self._data_manager.schema.columns)
        for key in keys:
            if key not in column_set:
                state |= 1
            else:
                state |= 2

        if state == 3:
            raise ValueError(f"setitem operation does not support a mix of old and new columns")

        from .ops._set_item import set_item

        set_item(self, keys, items, state)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state_dict):
        self.__dict__.update(state_dict)

    def __len__(self):
        return self.count()

    def _retrieval_attr(self) -> dict:
        return dict(
            ctx=self._ctx,
            schema=self._schema.dict(),
            index=self._index,
            values=self._values,
            label=self._label,
            weight=self._weight,
        )

    def __get_index_by_column_names(self, column_names):
        if isinstance(column_names, str):
            column_names = [column_names]

        indexes = []
        header_mapping = dict(zip(self._schema.header, range(len(self._schema.header))))
        for col in column_names:
            index = header_mapping.get(col, None)
            if index is None:
                raise ValueError(f"Can not find column: {col}")
            indexes.append(index)

        return indexes

    @auto_trace
    def get_indexer(self, target):
        if target not in ["sample_id", "match_id"]:
            raise ValueError(f"Target should be sample_id or match_id, but {target} found")

        if self.shape[0] == 0:
            return self._ctx.computing.parallelize([], include_key=False, partition=self._block_table.num_partitions)

        target_name = getattr(self.schema, f"{target}_name")
        indexer = self.__convert_to_table(target_name)
        if target == "sample_id":
            self._sample_id_indexer = indexer
        else:
            self._match_id_indexer = indexer

        return indexer

    @auto_trace
    def loc(self, indexer, target="sample_id", preserve_order=False):
        from .ops._indexer import loc

        return loc(self, indexer, target=target, preserve_order=preserve_order)

    @auto_trace
    def iloc(self, indexer: "DataFrame") -> "DataFrame":
        from .ops._dimension_scaling import retrieval_row

        return retrieval_row(self, indexer)

    @auto_trace
    def loc_with_sample_id_replacement(self, indexer):
        """
        indexer: table,
            row: (key=random_key,
            value=(sample_id, (src_block_id, src_block_offset))
        """
        from .ops._indexer import loc_with_sample_id_replacement

        return loc_with_sample_id_replacement(self, indexer)

    @auto_trace
    def flatten(self, key_type="block_id", with_sample_id=True):
        """
        flatten data_frame
        """
        from .ops._indexer import flatten_data

        return flatten_data(self, key_type=key_type, with_sample_id=with_sample_id)

    @auto_trace
    def copy(self) -> "DataFrame":
        return DataFrame(
            self._ctx,
            self._block_table.mapValues(lambda v: v),
            copy.deepcopy(self.partition_order_mappings),
            self._data_manager.duplicate(),
        )

    @classmethod
    def from_flatten_data(cls, ctx, flatten_table, data_manager, key_type) -> "DataFrame":
        from .ops._indexer import transform_flatten_data_to_df

        return transform_flatten_data_to_df(ctx, flatten_table, data_manager, key_type)

    @classmethod
    @auto_trace
    def hstack(cls, stacks: List["DataFrame"]) -> "DataFrame":
        from .ops._dimension_scaling import hstack

        return hstack(stacks)

    @classmethod
    @auto_trace
    def vstack(cls, stacks: List["DataFrame"]) -> "DataFrame":
        from .ops._dimension_scaling import vstack

        return vstack(stacks)

    @auto_trace
    def sample(self, n: int = None, frac: float = None, random_state=None) -> "DataFrame":
        from .ops._dimension_scaling import sample

        return sample(self, n, frac, random_state)

    def nlargest(self, n, columns, keep="first", error=1e-4):
        from .ops._sort import nlargest

        return nlargest(self, n, columns, keep=keep, error=error)

    def __extract_fields(
        self,
        with_sample_id=True,
        with_match_id=True,
        with_label=True,
        with_weight=True,
        columns: Union[str, list] = None,
    ) -> "DataFrame":
        from .ops._field_extract import field_extract

        return field_extract(
            self,
            with_sample_id=with_sample_id,
            with_match_id=with_match_id,
            with_label=with_label,
            with_weight=with_weight,
            columns=columns,
        )

    def __convert_to_table(self, target_name):
        block_loc = self._data_manager.loc_block(target_name)
        assert block_loc[1] == 0, "support only one indexer in current version"

        from .ops._indexer import transform_to_table

        return transform_to_table(self._block_table, block_loc[0], self._partition_order_mappings)

    def data_overview(self, num=100):
        from .ops._data_overview import collect_data

        return collect_data(self, num=num)
