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
from typing import List, Union

import numpy as np
import pandas as pd

from .manager import DataManager, Schema
from .ops import aggregate_indexer, get_partition_order_mappings


class DataFrame(object):
    def __init__(self, ctx, block_table, partition_order_mappings, data_manager: DataManager):
        self._ctx = ctx
        self._block_table = block_table
        self._partition_order_mappings = partition_order_mappings
        self._data_manager = data_manager

        """
        the following is cached
        index: [(id, (partition_id, index_in_block)]
        """
        self._sample_id_indexer = None
        self._match_id_indexer = None
        self._sample_id = None
        self._match_id = None
        self._label = None
        self._weight = None

        self.__count = None
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
                with_sample_id=True, with_match_id=True, with_label=False, with_weight=False
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
                with_sample_id=True, with_match_id=True, with_label=False, with_weight=False
            )

        return self._weight

    @property
    def shape(self) -> "tuple":
        if not self.__count:
            if self._sample_id_indexer:
                items = self._sample_id_indexer.count()
            elif self._match_id_indexer:
                items = self._match_id_indexer.count()
            else:
                items = self._block_table.mapValues(lambda block: 0 if block is None else len(block[0])).reduce(
                    lambda size1, size2: size1 + size2
                )
            self.__count = items

        return self.__count, len(self._data_manager.schema.columns)

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

    def as_tensor(self, dtype=None):
        """
        df.weight.as_tensor()
        df.label.as_tensor()
        df.values.as_tensor()
        """
        from .ops._transformer import transform_to_tensor

        return transform_to_tensor(self._block_table, self._data_manager, dtype)

    def as_pd_df(self) -> "pd.DataFrame":
        from .ops._transformer import transform_to_pandas_dataframe

        return transform_to_pandas_dataframe(self._block_table, self._data_manager)

    def apply_row(self, func, columns=None, with_label=False, with_weight=False, enable_type_align_checking=True):
        from .ops._apply_row import apply_row

        return apply_row(
            self,
            func,
            columns=columns,
            with_label=with_label,
            with_weight=with_weight,
            enable_type_align_checking=enable_type_align_checking,
        )

    def create_frame(self, with_label=False, with_weight=False, columns: list = None) -> "DataFrame":
        return self.__extract_fields(
            with_sample_id=True, with_match_id=True, with_label=with_label, with_weight=with_weight, columns=columns
        )

    def drop(self, index) -> "DataFrame":
        from .ops._dimension_scaling import drop

        return drop(self, index)

    def fillna(self, value):
        from .ops._fillna import fillna

        return fillna(self, value)

    def get_dummies(self, dtype="int32"):
        from .ops._encoder import get_dummies

        return get_dummies(self, dtype=dtype)

    def isna(self):
        from .ops._missing import isna

        return isna(self)

    def isin(self, values):
        from .ops._isin import isin

        return isin(self, values)

    def na_count(self):
        return self.isna().sum()

    def max(self) -> "pd.Series":
        from .ops._stat import max

        return max(self)

    def min(self, *args, **kwargs) -> "pd.Series":
        from .ops._stat import min

        return min(self)

    def mean(self, *args, **kwargs) -> "pd.Series":
        from .ops._stat import mean

        return mean(self)

    def sum(self, *args, **kwargs) -> "pd.Series":
        from .ops._stat import sum

        return sum(self)

    def std(self, ddof=1, **kwargs) -> "pd.Series":
        from .ops._stat import std

        return std(self, ddof=ddof)

    def var(self, ddof=1, **kwargs):
        from .ops._stat import var

        return var(self, ddof=ddof)

    def variation(self, ddof=1):
        from .ops._stat import variation

        return variation(self, ddof=ddof)

    def skew(self, unbiased=False):
        from .ops._stat import skew

        return skew(self, unbiased=unbiased)

    def kurt(self, unbiased=False):
        from .ops._stat import kurt

        return kurt(self, unbiased=unbiased)

    def sigmoid(self) -> "DataFrame":
        from .ops._activation import sigmoid

        return sigmoid(self)

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

    def count(self) -> "int":
        return self.shape[0]

    def describe(self, ddof=1, unbiased=False):
        from .ops._stat import describe

        return describe(self, ddof=ddof, unbiased=unbiased)

    def quantile(self, q, relative_error: float = 1e-4):
        from .ops._quantile import quantile

        return quantile(self, q, relative_error)

    def qcut(self, q: int):
        from .ops._quantile import qcut

        return qcut(self, q)

    def bucketize(self, boundaries: Union[dict, pd.DataFrame]) -> "DataFrame":
        from .ops._encoder import bucketize

        return bucketize(self, boundaries)

    def hist(self, targets):
        from .ops._histogram import hist

        return hist(self, targets)

    def replace(self, to_replace=None) -> "DataFrame":
        from .ops._replace import replace

        return replace(self, to_replace)

    def __add__(self, other: Union[int, float, list, "np.ndarray", "DataFrame", "pd.Series"]) -> "DataFrame":
        return self.__arithmetic_operate(operator.add, other)

    def __radd__(self, other: Union[int, float, list, "np.ndarray", "pd.Series"]) -> "DataFrame":
        return self + other

    def __sub__(self, other: Union[int, float, list, "np.ndarray", "pd.Series"]) -> "DataFrame":
        return self.__arithmetic_operate(operator.sub, other)

    def __rsub__(self, other: Union[int, float, list, "np.ndarray", "pd.Series"]) -> "DataFrame":
        return self * (-1) + other

    def __mul__(self, other) -> "DataFrame":
        return self.__arithmetic_operate(operator.mul, other)

    def __rmul__(self, other) -> "DataFrame":
        return self * other

    def __truediv__(self, other) -> "DataFrame":
        return self.__arithmetic_operate(operator.truediv, other)

    def __pow__(self, power) -> "DataFrame":
        return self.__arithmetic_operate(operator.pow, power)

    def __lt__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.lt, other)

    def __le__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.le, other)

    def __gt__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.gt, other)

    def __ge__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.ge, other)

    def __eq__(self, other) -> "DataFrame":
        return self.__cmp_operate(operator.eq, other)

    def __invert__(self):
        from .ops._unary_operator import invert

        return invert(self)

    def __arithmetic_operate(self, op, other) -> "DataFrame":
        """
        df * 1.5, int -> float
        可能的情况：
        a. columns类型统一：此时，block只有一个
        b. columns类型不一致，多block，但要求单个block里面所有列都是被使用的。

        需要注意的是：int/float可能会统一上升成float，所以涉及到block类型的变化和压缩
        """
        from .ops._arithmetic import arith_operate

        return arith_operate(self, other, op)

    def __cmp_operate(self, op, other) -> "DataFrame":
        from .ops._cmp import cmp_operate

        return cmp_operate(self, other, op)

    def __getattr__(self, attr):
        if attr not in self._data_manager.schema.columns:
            raise ValueError(f"DataFrame does not has attribute {attr}")

        return self.__getitem__(attr)

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

    def get_indexer(self, target):
        if target not in ["sample_id", "match_id"]:
            raise ValueError(f"Target should be sample_id or match_id, but {target} found")

        target_name = getattr(self.schema, f"{target}_name")
        indexer = self.__convert_to_table(target_name)
        if target == "sample_id":
            self._sample_id_indexer = indexer
        else:
            self._match_id_indexer = indexer

        return indexer

    def loc(self, indexer, target="sample_id", preserve_order=False):
        self_indexer = self.get_indexer(target)
        if preserve_order:
            indexer = self_indexer.join(indexer, lambda lhs, rhs: (lhs, rhs))
        else:
            indexer = self_indexer.join(indexer, lambda lhs, rhs: (lhs, lhs))

        agg_indexer = aggregate_indexer(indexer)

        if not preserve_order:

            def _convert_block(blocks, retrieval_indexes):
                row_indexes = [retrieval_index[0] for retrieval_index in retrieval_indexes]
                return [block[row_indexes] for block in blocks]

            block_table = self._block_table.join(agg_indexer, _convert_block)
        else:

            def _convert_to_block(kvs):
                ret_dict = {}
                for block_id, (blocks, block_indexer) in kvs:
                    """
                    block_indexer: row_id, (new_block_id, new_row_id)
                    """
                    for src_row_id, (dst_block_id, dst_row_id) in block_indexer:
                        if dst_block_id not in ret_dict:
                            ret_dict[dst_block_id] = []

                        ret_dict[dst_block_id].append(
                            [
                                block[src_row_id] if isinstance(block, pd.Index) else block[src_row_id].tolist()
                                for block in blocks
                            ]
                        )

                return list(ret_dict.items())

            def _merge_list(lhs, rhs):
                if not lhs:
                    return rhs
                if not rhs:
                    return lhs

                l_len = len(lhs)
                r_len = len(rhs)
                ret = [[] for i in range(l_len + r_len)]
                i, j, k = 0, 0, 0
                while i < l_len and j < r_len:
                    if lhs[i][0] < rhs[j][0]:
                        ret[k] = lhs[i]
                        i += 1
                    else:
                        ret[k] = rhs[j]
                        j += 1

                    k += 1

                while i < l_len:
                    ret[k] = lhs[i]
                    i += 1
                    k += 1

                while j < r_len:
                    ret[k] = rhs[j]
                    j += 1
                    k += 1

                return ret

            from .ops._transformer import transform_list_block_to_frame_block

            block_table = self._block_table.join(agg_indexer, lambda lhs, rhs: (lhs, rhs))
            block_table = block_table.mapReducePartitions(_convert_to_block, _merge_list)
            block_table = transform_list_block_to_frame_block(block_table, self._data_manager)

        partition_order_mappings = get_partition_order_mappings(block_table)
        return DataFrame(self._ctx, block_table, partition_order_mappings, self._data_manager.duplicate())

    def iloc(self, indexes):
        ...

    def copy(self) -> "DataFrame":
        return DataFrame(
            self._ctx,
            self._block_table.mapValues(lambda v: v),
            copy.deepcopy(self.partition_order_mappings),
            self._data_manager.duplicate(),
        )

    @classmethod
    def hstack(cls, stacks: List["DataFrame"]) -> "DataFrame":
        from .ops._dimension_scaling import hstack

        return hstack(stacks)

    @classmethod
    def vstack(cls, stacks: List["DataFrame"]) -> "DataFrame":
        from .ops._dimension_scaling import vstack

        return vstack(stacks)

    def sample(self, n: int = None, frac: float = None, random_state=None) -> "DataFrame":
        from .ops._dimension_scaling import sample

        return sample(self, n, frac, random_state)

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

        return collect_data(self, num=100)
