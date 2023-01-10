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
import functools
import typing

import numpy as np
import pandas as pd
import torch
from fate.arch import tensor

from ._dataframe import DataFrame
from .storage import Index


class RawTableReader(object):
    def __init__(
        self,
        delimiter: str = ",",
        label_name: typing.Union[None, str] = None,
        label_type: str = "int",
        weight_name: typing.Union[None, str] = None,
        dtype: str = "float32",
        input_format: str = "dense",
    ):
        self._delimiter = delimiter
        self._label_name = label_name
        self._label_type = label_type
        self._weight_name = weight_name
        self._dtype = dtype
        self._input_format = input_format

    def to_frame(self, ctx, table):
        if self._input_format != "dense":
            raise ValueError("Only support dense input format in this version.")

        return self._dense_format_to_frame(ctx, table)

    def _dense_format_to_frame(self, ctx, table):
        schema = dict()
        schema["sid"] = table.schema["sid"]
        header = table.schema["header"].split(self._delimiter, -1)

        table = table.mapValues(lambda value: value.split(self._delimiter, -1))
        header_indexes = list(range(len(header)))
        index_table, _block_partition_mapping, _global_ranks = _convert_to_order_indexes(table)

        data_dict = {}
        if self._label_name:
            if self._label_name not in header:
                raise ValueError("Label name does not exist in header, please have a check")
            label_idx = header.index(self._label_name)
            header.remove(self._label_name)
            header_indexes.remove(label_idx)
            label_type = getattr(np, self._label_type)
            label_table = table.mapValues(lambda value: [label_type(value[label_idx])])
            data_dict["label"] = _convert_to_tensor(
                ctx,
                label_table,
                block_partition_mapping=_block_partition_mapping,
                dtype=getattr(torch, self._label_type),
            )
            schema["label_name"] = self._label_name

        if self._weight_name:
            if self._weight_name not in header:
                raise ValueError("Weight name does not exist in header, please have a check")

            weight_idx = header.index(self._weight_name)
            header.remove(self._weight_name)
            header_indexes.remove(weight_idx)
            weight_table = table.mapValues(lambda value: [value[weight_idx]])
            data_dict["weight"] = _convert_to_tensor(
                ctx, weight_table, block_partition_mapping=_block_partition_mapping, dtype=getattr(torch, "float64")
            )

            schema["weight_name"] = self._weight_name

        if header_indexes:
            value_table = table.mapValues(lambda value: np.array(value)[header_indexes].astype(self._dtype).tolist())
            data_dict["values"] = _convert_to_tensor(
                ctx, value_table, block_partition_mapping=_block_partition_mapping, dtype=getattr(torch, self._dtype)
            )
            schema["header"] = header

        data_dict["index"] = _convert_to_index(
            ctx, index_table, block_partition_mapping=_block_partition_mapping, global_ranks=_global_ranks
        )

        return DataFrame(ctx=ctx, schema=schema, **data_dict)


class ImageReader(object):
    """
    Image Reader now support convert image to a 3D tensor, dtype=torch.float64
    """

    def __init__(
        self,
        mode="L",
    ):
        ...


class CSVReader(object):
    # TODO: fast data read
    # TODO: a. support match_id, b. more id type
    def __init__(
        self,
        id_name: typing.Union[None, str] = None,
        delimiter: str = ",",
        label_name: typing.Union[None, str] = None,
        label_type: str = "int",
        weight_name: typing.Union[None, str] = None,
        dtype: str = "float32",
        partition: int = 4,
    ):
        self._id_name = id_name
        self._delimiter = delimiter
        self._label_name = label_name
        self._label_type = label_type
        self._weight_name = weight_name
        self._dtype = dtype
        self._partition = partition

    def to_frame(self, ctx, path):
        # TODO: use table put data instead of read all data
        df = pd.read_csv(path, delimiter=self._delimiter)

        return PandasReader(
            id_name=self._id_name,
            label_name=self._label_name,
            label_type=self._label_type,
            weight_name=self._weight_name,
            partition=self._partition,
        ).to_frame(ctx, df)


class HiveReader(object):
    ...


class MysqlReader(object):
    ...


class TextReader(object):
    ...


class TorchDataSetReader(object):
    # TODO: this is for Torch DataSet Reader, the passing object has attributes __len__ and __get_item__
    def __init__(
        self,
    ):
        ...

    def to_frame(self, ctx, dataset):
        ...


class PandasReader(object):
    def __init__(
        self,
        id_name: typing.Union[None, str] = None,
        label_name: str = None,
        label_type: str = "int",
        weight_name: typing.Union[None, str] = None,
        dtype: str = "float32",
        partition: int = 4,
    ):
        self._id_name = id_name
        self._label_name = label_name
        self._label_type = label_type
        self._weight_name = weight_name
        self._dtype = dtype
        self._partition = partition

    def to_frame(self, ctx, df: "pd.DataFrame"):
        schema = dict()
        if not self._id_name:
            self._id_name = df.columns[0]
        df = df.set_index(self._id_name)

        # TODO: need to ensure id's type is str?
        df.index = df.index.astype("str")

        id_list = df.index.tolist()

        index_table = ctx.computing.parallelize(
            zip(id_list, range(df.shape[0])), include_key=True, partition=self._partition
        )

        index_table, _block_partition_mapping, _global_ranks = _convert_to_order_indexes(index_table)

        data_dict = {}
        if self._label_name:
            label_list = [[label] for label in df[self._label_name].tolist()]
            label_table = ctx.computing.parallelize(
                zip(id_list, label_list), include_key=True, partition=self._partition
            )
            data_dict["label"] = _convert_to_tensor(
                ctx,
                label_table,
                block_partition_mapping=_block_partition_mapping,
                dtype=getattr(torch, self._label_type),
            )
            df = df.drop(columns=self._label_name)
            schema["label_name"] = self._label_name

        if self._weight_name:
            weight_list = df[self._weight_name].tolist()
            weight_table = ctx.computing.parallelize(
                zip(id_list, weight_list), include_key=True, partition=self._partition
            )
            data_dict["weight"] = _convert_to_tensor(
                ctx, weight_table, block_partition_mapping=_block_partition_mapping, dtype=getattr(torch, "float64")
            )

            df = df.drop(columns=self._weight_name)
            schema["weight_name"] = self._weight_name

        if df.shape[1]:
            value_table = ctx.computing.parallelize(
                zip(id_list, df.values), include_key=True, partition=self._partition
            )
            data_dict["values"] = _convert_to_tensor(
                ctx, value_table, block_partition_mapping=_block_partition_mapping, dtype=getattr(torch, self._dtype)
            )
            schema["header"] = df.columns.to_list()

        data_dict["index"] = _convert_to_index(
            ctx, index_table, block_partition_mapping=_block_partition_mapping, global_ranks=_global_ranks
        )

        schema["sid"] = self._id_name

        return DataFrame(ctx=ctx, schema=schema, **data_dict)


def _convert_to_order_indexes(table):
    def _get_block_summary(kvs):
        key = next(kvs)[0]
        block_size = 1 + sum(1 for kv in kvs)
        return {key: block_size}

    def _order_indexes(kvs, rank_dict: dict = None):
        bid = None
        order_indexes = []
        for idx, (k, v) in enumerate(kvs):
            if bid is None:
                bid = rank_dict[k]["block_id"]

            order_indexes.append((k, (bid, idx)))

        return order_indexes

    block_summary = table.mapPartitions(_get_block_summary).reduce(lambda blk1, blk2: {**blk1, **blk2})

    start_index, block_id = 0, 0
    block_partition_mapping = dict()
    global_ranks = []
    for blk_key, blk_size in block_summary.items():
        block_partition_mapping[blk_key] = dict(
            start_index=start_index, end_index=start_index + blk_size - 1, block_id=block_id
        )
        global_ranks.append(block_partition_mapping[blk_key])

        start_index += blk_size
        block_id += 1

    order_func = functools.partial(_order_indexes, rank_dict=block_partition_mapping)
    order_table = table.mapPartitions(order_func, use_previous_behavior=False)

    return order_table, block_partition_mapping, global_ranks


def _convert_to_index(ctx, table, block_partition_mapping, global_ranks):
    return Index(ctx, table, block_partition_mapping=block_partition_mapping, global_ranks=global_ranks)


def _convert_to_tensor(ctx, table, block_partition_mapping, dtype):
    # TODO: in mini-demo stage, distributed tensor only accept list, in future, replace this with distributed table.
    convert_func = functools.partial(_convert_block, block_partition_mapping=block_partition_mapping, dtype=dtype)
    blocks_with_id = list(table.mapPartitions(convert_func, use_previous_behavior=False).collect())
    blocks = [block_with_id[1] for block_with_id in sorted(blocks_with_id)]

    return tensor.distributed_tensor(ctx, blocks, partitions=len(blocks))


def _convert_block(kvs, block_partition_mapping, dtype, convert_type="tensor"):
    ret = []
    block_id = None
    for key, value in kvs:
        if block_id is None:
            block_id = block_partition_mapping[key]["block_id"]

        ret.append(value)

    if convert_type == "tensor":
        return [(block_id, torch.tensor(ret, dtype=dtype))]
    else:
        return [(block_id, pd.Index(ret, dtype=dtype))]
