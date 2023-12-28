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
from ._indexer import get_partition_order_mappings_by_block_table
from ..manager.block_manager import Block
from .._dataframe import DataFrame


FLOATING_ZERO = 1e-8


def nlargest(df: DataFrame, n, columns, keep, error) -> DataFrame:
    if keep not in ["first", "last"]:
        raise ValueError(f"keep={keep} is not supported, only first or last is accepted")

    if n > len(df):
        raise ValueError(f"n={n} > dataframe's row size={len(df)}")

    if n == 0:
        return df.empty_frame()

    if isinstance(columns, str):
        columns = [columns]

    if error > FLOATING_ZERO and len(columns) == 1:
        return _nlargest_by_quantile(df, n, columns[0], keep, error)
    else:
        return _nlargest_exactly(df, n, columns, keep)


def _nlargest_by_quantile(df: DataFrame, n, column, keep, error) -> DataFrame:
    ret_frame = df.empty_frame()
    left = n
    offset = 0

    while True:
        if left + offset >= len(df):
            tmp_df = df
        else:
            rate = (left + offset) / len(df)
            split_value = df.quantile([rate], relative_error=error)[column].tolist()[0]
            tmp_df = df.iloc(df[column] >= split_value)

        if len(tmp_df) >= left:
            if keep == "first":
                tmp_df = tmp_df.sample(n=left)

            ret_frame = DataFrame.vstack([ret_frame, tmp_df])
            break

        ret_frame = DataFrame.vstack([ret_frame, tmp_df])
        left -= len(tmp_df)

        if not offset:
            offset = 1
        else:
            offset <<= 1

    return ret_frame


def _nlargest_exactly(df: DataFrame, n, columns, keep) -> DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    fields_loc = df.data_manager.loc_block(columns)
    block_num = df.data_manager.block_num

    values = []
    for k, blocks in df.block_table.collect():
        flat_blocks = [Block.transform_block_to_list(block) for block in blocks]
        if not values:
            values = flat_blocks
        else:
            for _bid in range(block_num):
                values[_bid].extend(flat_blocks[_bid])

    indexes = [i for i in range(len(df))]

    def _extract_columns(r_id):
        elem = [values[bid][r_id][_offset] for bid, _offset in fields_loc]
        return tuple(elem)

    indexes.sort(key=_extract_columns, reverse=True)

    if keep == "last":
        while n + 1 < len(indexes):
            if _extract_columns(indexes[n]) == _extract_columns(indexes[n + 1]):
                n += 1

    indexes = indexes[:n]

    block_row_size = df.data_manager.block_row_size
    blocks_with_id = []

    for block_id in range((n + block_row_size - 1) // block_row_size):
        block_indexes = indexes[block_id * block_row_size : (block_id + 1) * block_row_size]

        blocks = [[] for _ in range(block_num)]

        for idx in block_indexes:
            for j in range(block_num):
                blocks[j].append(values[j][idx])

        blocks_with_id.append((block_id, df.data_manager.convert_to_blocks(blocks)))

    block_table = df._ctx.computing.parallelize(
        blocks_with_id, include_key=True, partition=df.block_table.num_partitions
    )

    partition_order_mappings = get_partition_order_mappings_by_block_table(block_table, block_row_size=block_row_size)

    return DataFrame(df._ctx, block_table, partition_order_mappings, data_manager=df.data_manager.duplicate())
