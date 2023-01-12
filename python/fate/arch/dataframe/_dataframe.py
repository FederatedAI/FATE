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
import copy
import operator

import torch
from fate.arch.computing import is_table

from .ops import arith_method, stat_method, transform_to_predict_result
from .storage import Index, ValueStore


# TODO: record data type, support multiple data types
class DataFrame(object):
    def __init__(self, ctx, schema, index=None, match_id=None, values=None, label=None, weight=None):
        self._ctx = ctx
        self._index = index
        self._match_id = match_id
        self._values = values
        self._label = label
        self._weight = weight
        self._schema = Schema(**schema)

        self.__shape = None
        self._columns = None

        self._tensor_label = None

    @property
    def index(self):
        return self._index

    @property
    def values(self):
        return self._values

    @property
    def label(self):
        return self._label

    @property
    def weight(self):
        return self._weight

    @property
    def match_id(self):
        return self._match_id

    @property
    def shape(self):
        if self.__shape:
            return self.__shape

        if self._values is None:
            self.__shape = (self._index.count(), 0)
        else:
            self.__shape = (self._index.count(), len(self._schema.header))

        return self.__shape

    @property
    def schema(self) -> "Schema":
        return self._schema

    @property
    def columns(self) -> "ColumnObject":
        if not self._columns:
            self._columns = ColumnObject(self._schema.header)
        else:
            return self._columns

    def max(self, *args, **kwargs) -> "DataFrame":
        return stat_method(self._values, "max", *args, index=self._schema.header, **kwargs)

    def min(self, *args, **kwargs) -> "DataFrame":
        return stat_method(self._values, "min", *args, index=self._schema.header, **kwargs)

    def mean(self, *args, **kwargs) -> "DataFrame":
        return stat_method(self._values, "mean", *args, index=self._schema.header, **kwargs)

    def sum(self, *args, **kwargs) -> "DataFrame":
        return stat_method(self._values, "sum", *args, index=self._schema.header, **kwargs)

    def std(self, *args, **kwargs) -> "DataFrame":
        return stat_method(self._values, "std", *args, index=self._schema.header, **kwargs)

    def count(self) -> "int":
        return self.shape[0]

    def __add__(self, other) -> "DataFrame":
        return self._arithmetic_operate(operator.add, other)

    def __sub__(self, other) -> "DataFrame":
        return self._arithmetic_operate(operator.sub, other)

    def __mul__(self, other) -> "DataFrame":
        return self._arithmetic_operate(operator.mul, other)

    def __truediv__(self, other) -> "DataFrame":
        return self._arithmetic_operate(operator.truediv, other)

    def _arithmetic_operate(self, op, other) -> "DataFrame":
        ret_value = arith_method(self._values, other, op)
        attrs_dict = self._retrieval_attr()
        attrs_dict["values"] = ret_value
        return DataFrame(**attrs_dict)

    def __getattr__(self, attr):
        if attr not in self.schema.header:
            raise ValueError(f"DataFrame does not has attribute {attr}")

        if isinstance(self._values, ValueStore):
            value = getattr(self._values, attr)
        else:
            col_idx = self.schema.header.index(attr)
            value = self._values[:, col_idx]

        schema = dict(sid=self.schema.sid, header=[attr])

        return DataFrame(self._ctx, schema=schema, values=value)

    def __getitem__(self, items):
        indexes = self.__get_index_by_column_names(items)
        ret_tensor = self._values[:, indexes]

        header_mapping = dict(zip(self._schema.header, range(len(self._schema.header))))
        new_schema = copy.deepcopy(self._schema)
        new_header = items if isinstance(items, list) else [items]
        new_anonymous_header = []

        for item in items:
            index = header_mapping[item]
            new_anonymous_header.append(self._schema.anonymous_header[index])

        new_schema["header"] = new_header
        new_schema["anonymous__header"] = new_anonymous_header

        return DataFrame(
            self._ctx, index=self._index, values=ret_tensor, label=self._label, weight=self._weight, schema=new_schema
        )

    def __setitem__(self, keys, item):
        if not isinstance(item, DataFrame):
            raise ValueError("Using syntax df[[col1, col2...]] = rhs, rhs should be a dataframe")

        indexes = self.__get_index_by_column_names(keys)
        self._values[:, indexes] = item._values

        return self

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

    def loc(self, ids, with_partition_id=True):
        # this is very costly, use iloc is better
        # TODO: if data is not balance, repartition is need?
        if isinstance(ids, int):
            ids = [ids]

        indexes = self._index.get_indexer(ids, with_partition_id)

        return self.iloc(indexes)

    def iloc(self, indexes):
        # TODO: if data is not balance, repartition is need?
        if self.is_local:
            if is_table(indexes):
                raise ValueError("Local dataframe does not support table indexer")
                # indexes = indexes.reduce(lambda l1, l2: l1 + l2)

            weight = self._weight[indexes] if self._weight else None
            label = self._label[indexes] if self._label else None
            values = self._values[indexes] if self._values else None
            match_id = self._match_id[indexes] if self._match_id else None
            index = self._index[indexes]
        elif isinstance(indexes, (int, list)) or is_table(indexes):
            if isinstance(indexes, int):
                indexes = [indexes]

            """
            indexer: [(old_partition_id, old_block_index), (new_partition_id, new_block_index)]
            note: new_block_index may not be continuous
            """
            if isinstance(indexes, list):
                indexes = self._index.change_index_list_to_indexer(indexes)
            """
            agg_indexer: key=old_partition_id, value=[old_block_index, (new_partition_id, new_block_index)]
            """
            agg_indexer = Index.aggregate_indexer(indexes)

            # TODO: distributed tensor does not provider slice api, need to fix later
            def _iloc_tensor(distributed_tensor):
                blocks = distributed_tensor.storage.blocks
                dtype = blocks.first()[1].dtype.name

                def _retrieval_func(kvs):
                    ret = dict()
                    for partition_id_key, (t, mappings) in kvs:
                        t = t.to_local().data.tolist()
                        for old_block_index, (new_partition_id, new_block_index) in mappings:
                            t_value = t[old_block_index]

                            if new_partition_id not in ret:
                                ret[new_partition_id] = []
                            ret[new_partition_id].append((new_block_index, t_value))

                    return list(ret.items())

                blocks = blocks.join(agg_indexer, lambda ten, block_mapping: (ten, block_mapping))
                blocks = blocks.mapReducePartitions(_retrieval_func, lambda l1, l2: l1 + l2)
                blocks = blocks.mapValues(lambda block: sorted(block, key=lambda buf: buf[0]))
                blocks = blocks.mapValues(
                    lambda block: torch.tensor([value[1] for value in block], dtype=getattr(torch, dtype))
                )
                blocks = [block for pid, block in sorted(list(blocks.collect()))]

                from fate.arch import tensor

                return tensor.distributed_tensor(self._ctx, blocks, partitions=len(blocks))

            weight = _iloc_tensor(self._weight) if self._weight else None
            label = _iloc_tensor(self._label) if self._label else None
            values = _iloc_tensor(self._values) if self._values else None
            match_id = _iloc_tensor(self._match_id) if self._match_id else None
            index = self._index[indexes]
        else:
            raise ValueError(f"iloc function dose not support args type={type(indexes)}")

        return DataFrame(
            self._ctx, self._schema.dict(), index=index, match_id=match_id, label=label, weight=weight, values=values
        )


    @property
    def is_local(self):
        if self._values is not None:
            return not self._values.is_distributed
        if self._weight is not None:
            return not self._weight.is_distributed
        if self.label is not None:
            return not self._label.is_distributed
        if self._match_id is not None:
            return not self._match_id.is_distributed

        return False

    def transform_to_predict_result(
        self, predict_score, data_type="train", task_type="binary", classes=None, threshold=0.5
    ):
        """ """

        ret, header = transform_to_predict_result(
            self._ctx, predict_score, data_type=data_type, task_type=task_type, classes=classes, threshold=threshold
        )

        transform_schema = {"header": header, "sid": self._schema.sid}
        if self._schema.match_id_name:
            transform_schema["match_id_name"] = self._schema.match_id_name

        if self._label:
            transform_schema["label_name"] = self.schema.label_name

        return DataFrame(
            ctx=self._ctx,
            index=self._index,
            match_id=self._match_id,
            label=self.label,
            values=ValueStore(self._ctx, ret, header),
            schema=transform_schema,
        )


class ColumnObject(object):
    def __init__(self, col_names):
        self._col_names = col_names

    def __getitem__(self, items):
        if isinstance(items, int):
            return self._col_names[items]
        else:
            ret_cols = []
            for item in items:
                ret_cols.append(self._col_names[item])

            return ColumnObject(ret_cols)

    def tolist(self):
        return self._col_names

    def __iter__(self):
        return (col_name for col_name in self._col_names)


class Schema(object):
    def __init__(
        self, sid=None, match_id_name=None, weight_name=None, label_name=None, header=None, anonymous_header=None
    ):
        self._sid = sid
        self._match_id_name = match_id_name
        self._weight_name = weight_name
        self._label_name = label_name
        self._header = header
        self._anonymous_header = anonymous_header

    @property
    def sid(self):
        return self._sid

    @property
    def match_id_name(self):
        return self._match_id_name

    @property
    def weight_name(self):
        return self._weight_name

    @property
    def label_name(self):
        return self._label_name

    @property
    def header(self):
        return self._header

    @property
    def anonymous_header(self):
        return self._anonymous_header

    def dict(self):
        schema = dict(sid=self._sid)

        if self._header:
            schema["header"] = self._header
        if self._anonymous_header:
            schema["anonymous_header"] = self._anonymous_header

        if self._weight_name:
            schema["weight_name"] = self._weight_name

        if self._label_name:
            schema["label_name"] = self._label_name

        if self._match_id_name:
            schema["match_id_name"] = self._match_id_name

        return schema
