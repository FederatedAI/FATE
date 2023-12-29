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
import pandas as pd
from typing import Union


from .conf.default_config import DATAFRAME_BLOCK_ROW_SIZE
from .entity import types
from ._dataframe import DataFrame
from .manager import DataManager
from fate.arch.trace import auto_trace


class TableReader(object):
    def __init__(
        self,
        sample_id_name: str = None,
        match_id_name: str = None,
        match_id_list: list = None,
        match_id_range: int = 0,
        label_name: Union[None, str] = None,
        label_type: str = "int",
        weight_name: Union[None, str] = None,
        weight_type: str = "float32",
        header: str = None,
        delimiter: str = ",",
        dtype: Union[str, dict] = "float32",
        anonymous_site_name: str = None,
        na_values: Union[str, list, dict] = None,
        input_format: str = "dense",
        tag_with_value: bool = False,
        tag_value_delimiter: str = ":",
        block_row_size: int = None,
    ):
        self._sample_id_name = sample_id_name
        self._match_id_name = match_id_name
        self._match_id_list = match_id_list
        self._match_id_range = match_id_range
        self._label_name = label_name
        self._label_type = label_type
        self._weight_name = weight_name
        self._weight_type = weight_type
        self._delimiter = delimiter
        self._header = header
        self._dtype = dtype
        self._anonymous_site_name = anonymous_site_name
        self._na_values = na_values
        self._input_format = input_format
        self._tag_with_value = tag_with_value
        self._tag_value_delimiter = tag_value_delimiter
        self._block_row_size = block_row_size if block_row_size is not None else DATAFRAME_BLOCK_ROW_SIZE

        self.check_params()

    def check_params(self):
        if not self._sample_id_name:
            raise ValueError("Please provide sample_id_name")

        if not isinstance(self._block_row_size, int) or self._block_row_size < 0:
            raise ValueError("block_row_size should be positive integer")

    @auto_trace
    def to_frame(self, ctx, table):
        if self._input_format != "dense":
            raise ValueError("Only support dense input format in this version.")

        return self._dense_format_to_frame(ctx, table)

    def _dense_format_to_frame(self, ctx, table):
        data_manager = DataManager(block_row_size=self._block_row_size)
        columns = self._header.split(self._delimiter, -1)
        columns.remove(self._sample_id_name)
        retrieval_index_dict = data_manager.init_from_local_file(
            sample_id_name=self._sample_id_name,
            columns=columns,
            match_id_list=self._match_id_list,
            match_id_name=self._match_id_name,
            label_name=self._label_name,
            weight_name=self._weight_name,
            label_type=self._label_type,
            weight_type=self._weight_type,
            dtype=self._dtype,
            default_type=types.DEFAULT_DATA_TYPE,
        )

        from .ops._indexer import get_partition_order_by_raw_table

        partition_order_mappings = get_partition_order_by_raw_table(table, data_manager.block_row_size)
        # partition_order_mappings = _get_partition_order(table)
        table = table.mapValues(lambda value: value.split(self._delimiter, -1))
        to_block_func = functools.partial(
            _to_blocks,
            data_manager=data_manager,
            retrieval_index_dict=retrieval_index_dict,
            partition_order_mappings=partition_order_mappings,
        )
        block_table = table.mapPartitions(to_block_func, use_previous_behavior=False)

        return DataFrame(
            ctx=ctx,
            block_table=block_table,
            partition_order_mappings=partition_order_mappings,
            data_manager=data_manager,
        )


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
    def __init__(
        self,
        sample_id_name: Union[None, str] = None,
        match_id_list: Union[None, list] = None,
        match_id_name: Union[None, str] = None,
        delimiter: str = ",",
        label_name: Union[None, str] = None,
        label_type: str = "int",
        weight_name: Union[None, str] = None,
        weight_type: str = "float32",
        dtype: str = "float32",
        na_values: Union[None, str, list, dict] = None,
        partition: int = 4,
        block_row_size: int = None,
    ):
        self._sample_id_name = sample_id_name
        self._match_id_list = match_id_list
        self._match_id_name = match_id_name
        self._delimiter = delimiter
        self._label_name = label_name
        self._label_type = label_type
        self._weight_name = weight_name
        self._weight_type = weight_type
        self._dtype = dtype
        self._na_values = na_values
        self._partition = partition
        self._block_row_size = block_row_size if block_row_size is not None else DATAFRAME_BLOCK_ROW_SIZE

    @auto_trace
    def to_frame(self, ctx, path):
        # TODO: use table put data instead of read all data
        df = pd.read_csv(path, delimiter=self._delimiter, na_values=self._na_values)

        return PandasReader(
            sample_id_name=self._sample_id_name,
            match_id_list=self._match_id_list,
            match_id_name=self._match_id_name,
            label_name=self._label_name,
            label_type=self._label_type,
            weight_name=self._weight_name,
            dtype=self._dtype,
            partition=self._partition,
            block_row_size=self._block_row_size,
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
        sample_id_name: Union[None, str] = None,
        match_id_list: Union[None, list] = None,
        match_id_name: Union[None, str] = None,
        label_name: str = None,
        label_type: str = "int32",
        weight_name: Union[None, str] = None,
        weight_type: str = "float32",
        dtype: str = "float32",
        partition: int = 4,
        block_row_size: int = None,
    ):
        self._sample_id_name = sample_id_name
        self._match_id_list = match_id_list
        self._match_id_name = match_id_name
        self._label_name = label_name
        self._label_type = label_type
        self._weight_name = weight_name
        self._weight_type = weight_type
        self._dtype = dtype
        self._partition = partition
        self._block_row_size = block_row_size if block_row_size is not None else DATAFRAME_BLOCK_ROW_SIZE

        if self._sample_id_name and not self._match_id_name:
            raise ValueError(f"As sample_id {self._sample_id_name} is given, match_id should be given too")

    @auto_trace
    def to_frame(self, ctx, df: "pd.DataFrame"):
        if not self._sample_id_name:
            self._sample_id_name = types.DEFAULT_SID_NAME
            df.index.name = self._sample_id_name
        else:
            df = df.set_index(self._sample_id_name)

        data_manager = DataManager(block_row_size=self._block_row_size)
        retrieval_index_dict = data_manager.init_from_local_file(
            sample_id_name=self._sample_id_name,
            columns=df.columns.tolist(),
            match_id_list=self._match_id_list,
            match_id_name=self._match_id_name,
            label_name=self._label_name,
            weight_name=self._weight_name,
            label_type=self._label_type,
            weight_type=self._weight_type,
            dtype=self._dtype,
            default_type=types.DEFAULT_DATA_TYPE,
        )

        site_name = ctx.local.name
        local_role = ctx.local.party[0]

        if local_role != "local":
            data_manager.fill_anonymous_site_name(site_name=site_name)

        buf = zip(df.index.tolist(), df.values.tolist())
        table = ctx.computing.parallelize(buf, include_key=True, partition=self._partition)

        from .ops._indexer import get_partition_order_by_raw_table

        partition_order_mappings = get_partition_order_by_raw_table(table, data_manager.block_row_size)
        # partition_order_mappings = _get_partition_order(table)
        to_block_func = functools.partial(
            _to_blocks,
            data_manager=data_manager,
            retrieval_index_dict=retrieval_index_dict,
            partition_order_mappings=partition_order_mappings,
        )

        block_table = table.mapPartitions(to_block_func, use_previous_behavior=False)

        return DataFrame(
            ctx=ctx,
            block_table=block_table,
            partition_order_mappings=partition_order_mappings,
            data_manager=data_manager,
        )


def _to_blocks(kvs, data_manager=None, retrieval_index_dict=None, partition_order_mappings=None, na_values=None):
    """
    sample_id/match_id,label(maybe missing),weight(maybe missing),X
    """
    block_id = None

    schema = data_manager.schema

    splits = [[] for _ in range(data_manager.block_num)]
    sample_id_block = (
        data_manager.loc_block(schema.sample_id_name, with_offset=False) if schema.sample_id_name else None
    )

    match_id_block = data_manager.loc_block(schema.match_id_name, with_offset=False) if schema.match_id_name else None
    match_id_column_index = retrieval_index_dict["match_id_index"]

    label_block = data_manager.loc_block(schema.label_name, with_offset=False) if schema.label_name else None
    label_column_index = retrieval_index_dict["label_index"]

    weight_block = data_manager.loc_block(schema.weight_name, with_offset=False) if schema.weight_name else None
    weight_column_index = retrieval_index_dict["weight_index"]

    column_indexes = retrieval_index_dict["column_indexes"]

    columns = schema.columns
    column_blocks_mapping = dict()
    for col_id, col_name in zip(column_indexes, columns):
        bid = data_manager.loc_block(col_name, with_offset=False)
        if bid not in column_blocks_mapping:
            column_blocks_mapping[bid] = []

        column_blocks_mapping[bid].append(col_id)

    block_row_size = data_manager.block_row_size

    lid = 0
    for key, value in kvs:
        if block_id is None:
            block_id = partition_order_mappings[key]["start_block_id"]
        lid += 1

        # columns = value.split(",", -1)
        splits[sample_id_block].append(key)
        if match_id_block:
            splits[match_id_block].append(value[match_id_column_index])
        if label_block:
            splits[label_block].append([value[label_column_index]])
        if weight_block:
            splits[weight_block].append([value[weight_column_index]])

        for bid, col_id_list in column_blocks_mapping.items():
            splits[bid].append([value[col_id] for col_id in col_id_list])

        if lid % block_row_size == 0:
            converted_blocks = data_manager.convert_to_blocks(splits)
            yield block_id, converted_blocks
            block_id += 1
            splits = [[] for _ in range(data_manager.block_num)]

    if lid % block_row_size:
        converted_blocks = data_manager.convert_to_blocks(splits)
        yield block_id, converted_blocks
