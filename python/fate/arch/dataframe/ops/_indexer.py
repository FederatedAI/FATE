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
import functools
import uuid

from ..manager import Block, DataManager
from .._dataframe import DataFrame


def transform_to_table(block_table, block_index, partition_order_mappings):
    def _convert_to_order_index(kvs):
        for block_id, blocks in kvs:
            for _idx, _id in enumerate(blocks[block_index]):
                yield _id, (block_id, _idx)

    return block_table.mapPartitions(_convert_to_order_index,
                                     use_previous_behavior=False)


def get_partition_order_mappings_by_block_table(block_table, block_row_size):
    def _block_counter(kvs):
        partition_key = None
        size = 0
        first_block_id = ''

        for k, v in kvs:
            if size == 0 and len(v[0]):
                partition_key = v[0][0]

            if first_block_id == '':
                first_block_id = k

            size += len(v[0])

        if size == 0:
            partition_key = str(first_block_id) + "_" + str(uuid.uuid1())

        return partition_key, size

    block_info = sorted([summary[1] for summary in block_table.applyPartitions(_block_counter).collect()])
    block_order_mappings = dict()
    start_index = 0
    acc_block_num = 0
    for block_key, block_size in block_info:
        block_num = (block_size + block_row_size - 1) // block_row_size
        block_order_mappings[block_key] = dict(
            start_index=start_index,
            end_index=start_index + block_size - 1,
            start_block_id=acc_block_num,
            end_block_id=acc_block_num + block_num - 1
        )
        start_index += block_size
        acc_block_num += block_num

    return block_order_mappings


def get_partition_order_by_raw_table(table, block_row_size, key_type="sample_id"):
    def _get_block_summary(kvs):
        try:
            if key_type == "sample_id":
                key = next(kvs)[0]
            else:
                key = next(kvs)[1][0]

            block_size = 1 + sum(1 for kv in kvs)
        except StopIteration:
            key, block_size = 0, 0

        return {key: block_size}

    block_summary = table.applyPartitions(_get_block_summary).reduce(lambda blk1, blk2: {**blk1, **blk2})

    start_index, acc_block_num = 0, 0
    block_order_mappings = dict()

    if not block_summary:
        return block_order_mappings

    for blk_key, blk_size in sorted(block_summary.items()):
        block_num = (blk_size + block_row_size - 1) // block_row_size
        block_order_mappings[blk_key] = dict(
            start_index=start_index,
            end_index=start_index + blk_size - 1,
            start_block_id=acc_block_num,
            end_block_id=acc_block_num + block_num - 1
        )

        start_index += blk_size
        acc_block_num += block_num

    return block_order_mappings


def regenerated_sample_id(block_table, regenerated_sample_id_table, data_manager):
    """
    regenerated_sample_id_table: (sample_id, ([new_id_list]))
    """
    from ._dimension_scaling import _flatten_partition

    _flatten_func = functools.partial(_flatten_partition, block_num=data_manager.block_num)
    raw_table = block_table.mapPartitions(_flatten_func, use_previous_behavior=False)
    regenerated_table = raw_table.join(regenerated_sample_id_table, lambda lhs, rhs: (lhs, rhs))

    def _flat_id(key, value):
        content, id_list = value
        flat_ret = []
        for _id in id_list:
            flat_ret.append((_id, content))

        return flat_ret

    _flat_id_func = functools.partial(_flat_id)
    regenerated_table = regenerated_table.flatMap(_flat_id_func)

    return regenerated_table


def _merge_list(lhs, rhs):
    if not lhs:
        return rhs
    if not rhs:
        return lhs

    l_len = len(lhs)
    r_len = len(rhs)
    ret = [None] * (l_len + r_len)
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


def flatten_data(df: DataFrame, key_type="block_id", with_sample_id=True):
    """
    key_type="block_id":
        key=(block_id, block_offset), value=data_row
    key_type="sample_id":
        key=sample_id, value=data_row
    """
    sample_id_index = df.data_manager.loc_block(
        df.data_manager.schema.sample_id_name, with_offset=False
    ) if (with_sample_id or key_type == "sample_id") else  None

    def _flatten(kvs):
        for block_id, blocks in kvs:
            flat_blocks = [Block.transform_block_to_list(block) for block in blocks]
            block_num = len(flat_blocks)
            if key_type == "block_id":
                for row_id in range(len(blocks[0])):
                    if with_sample_id:
                        yield (block_id, row_id), (
                            flat_blocks[sample_id_index][row_id],
                            [flat_blocks[i][row_id] for i in range(block_num)]
                        )
                    else:
                        yield (block_id, row_id), [flat_blocks[i][row_id] for i in range(block_num)]
            else:
                for row_id in range(len(blocks[0])):
                    yield flat_blocks[sample_id_index][row_id], [flat_blocks[i][row_id] for i in range(block_num)]

    if key_type in ["block_id", "sample_id"]:
        return df.block_table.mapPartitions(_flatten, use_previous_behavior=False)
    else:
        raise ValueError(f"Not Implement key_type={key_type} of flatten_data.")


def transform_flatten_data_to_df(ctx, flatten_table, data_manager: DataManager, key_type, value_with_sample_id=True):
    partition_order_mappings = get_partition_order_by_raw_table(flatten_table,
                                                                data_manager.block_row_size,
                                                                key_type=key_type)
    block_num = data_manager.block_num

    def _convert_to_blocks(kvs):
        bid = None
        ret_blocks = [[] for _ in range(block_num)]

        lid = 0
        for _, value in kvs:
            if value_with_sample_id:
                data = value[1]
            else:
                data = value
            lid += 1
            if bid is None:
                sample_id = data[0]
                bid = partition_order_mappings[sample_id]["start_block_id"]

            for i in range(block_num):
                ret_blocks[i].append(data[i])

            if lid % data_manager.block_row_size == 0:
                ret_blocks = [data_manager.blocks[i].convert_block(block) for i, block in enumerate(ret_blocks)]
                yield bid, ret_blocks
                bid += 1
                ret_blocks = [[] for _ in range(block_num)]

        if lid % data_manager.block_row_size:
            ret_blocks = [data_manager.blocks[i].convert_block(block) for i, block in enumerate(ret_blocks)]
            yield bid, ret_blocks

    block_table = flatten_table.mapPartitions(_convert_to_blocks, use_previous_behavior=False)

    return DataFrame(
        ctx=ctx,
        block_table=block_table,
        partition_order_mappings=partition_order_mappings,
        data_manager=data_manager.duplicate()
    )


def loc(df: DataFrame, indexer, target="sample_id", preserve_order=False):
    """
    indexer: table, key=sample_id, value=(block_id, block_offset)
    """
    if target != "sample_id":
        raise ValueError(f"Only target=sample_id is supported, but target={target} is found")
    flatten_table = flatten_data(df, key_type="sample_id")
    if not preserve_order:
        flatten_table = flatten_table.join(indexer, lambda v1, v2: v1)
        if not flatten_table.count():
            return df.empty_frame()
        return transform_flatten_data_to_df(df._ctx, flatten_table, df.data_manager,
                                            key_type="sample_id", value_with_sample_id=False)
    else:
        flatten_table_with_dst_indexer = flatten_table.join(indexer, lambda v1, v2: (v2[0], (v2[1], v1)))
        if not flatten_table_with_dst_indexer.count():
            return df.empty_frame()

        def _aggregate(kvs):
            values = [value for key, value in kvs]
            values.sort()
            i = 0
            l = len(values)
            while i < l:
                j = i + 1
                while j < l and values[j][0] == values[i][0]:
                    j += 1

                yield values[i][0], [values[k][1] for k in range(i, j)]

                i = j

        data_manager = df.data_manager.duplicate()
        block_num = data_manager.block_num

        def _to_blocks(values):
            block_size = len(values)
            ret_blocks = [[None] * block_size for _ in range(block_num)]

            for row_id, row_data in values:
                for j in range(block_num):
                    ret_blocks[j][row_id] = row_data[j]

            for idx, block_schema in enumerate(data_manager.blocks):
                ret_blocks[idx] = block_schema.convert_block(ret_blocks[idx])

            return ret_blocks

        agg_data = flatten_table_with_dst_indexer.mapReducePartitions(_aggregate, lambda v1, v2: v1 + v2)
        block_table = agg_data.mapValues(_to_blocks)

        partition_order_mappings = get_partition_order_mappings_by_block_table(
            block_table,
            block_row_size=data_manager.block_row_size
        )

        return DataFrame(
            df._ctx,
            block_table=block_table,
            partition_order_mappings=partition_order_mappings,
            data_manager=data_manager.duplicate()
        )


def loc_with_sample_id_replacement(df: DataFrame, indexer):
    """
    indexer: table,
            row: (key=random_key,
            value=(sample_id, (src_block_id, src_offset))
    """
    if indexer.count() == 0:
        return df.empty_frame()

    data_manager = df.data_manager
    partition_order_mappings = get_partition_order_by_raw_table(indexer,
                                                                data_manager.block_row_size,
                                                                key_type="block_id")
    
    def _aggregate(kvs):
        bid, offset = None, 0
        flat_ret = []
        for k, values in kvs:
            sample_id, (src_block_id, src_offset) = values
            if bid is None:
                bid = partition_order_mappings[sample_id]["start_block_id"]

            flat_ret.append((src_block_id, src_offset, sample_id, bid, offset))

            offset += 1
            if offset == data_manager.block_row_size:
                offset = 0
                bid += 1

        flat_ret.sort()
        i = 0
        l = len(flat_ret)
        while i < l:
            j = i
            while j < l and flat_ret[i][0] == flat_ret[j][0]:
                j += 1

            agg_ret = [flat_ret[k][1:] for k in range(i, j)]
            yield  flat_ret[i][0], agg_ret

            i = j

    sample_id_index = data_manager.loc_block(data_manager.schema.sample_id_name, with_offset=False)
    block_num = data_manager.block_num
    
    def _convert_to_row(kvs):
        ret_dict = {}
        for block_id, (blocks, block_indexer) in kvs:
            flat_blocks = [Block.transform_block_to_list(block) for block in blocks]
            for src_row_id, sample_id, dst_block_id, dst_row_id in block_indexer:
                if dst_block_id not in ret_dict:
                    ret_dict[dst_block_id] = []

                row_data = [flat_blocks[i][src_row_id] for i in range(block_num)]
                row_data[sample_id_index] = sample_id

                ret_dict[dst_block_id].append(
                    (dst_row_id, row_data)
                )

        for dst_block_id, value_list in ret_dict.items():
            yield dst_block_id, sorted(value_list)

    agg_indexer = indexer.mapReducePartitions(_aggregate, lambda l1, l2: l1 + l2)
    block_table = df.block_table.join(agg_indexer, lambda v1, v2: (v1, v2))
    block_table = block_table.mapReducePartitions(_convert_to_row, _merge_list)

    def _convert_to_frame_block(blocks, data_manager):
        convert_blocks = []
        for idx, block_schema in enumerate(data_manager.blocks):
            block_content = [row_data[1][idx] for row_data in blocks]
            convert_blocks.append(block_schema.convert_block(block_content))

        return convert_blocks

    _convert_to_frame_block_func = functools.partial(_convert_to_frame_block, data_manager=data_manager)
    block_table = block_table.mapValues(_convert_to_frame_block_func)

    return DataFrame(
        ctx=df._ctx,
        block_table=block_table,
        partition_order_mappings=partition_order_mappings,
        data_manager=data_manager
    )
