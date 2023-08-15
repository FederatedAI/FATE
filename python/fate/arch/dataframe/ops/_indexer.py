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
import pandas as pd

from ..manager import Block, BlockType, DataManager
from .._dataframe import DataFrame


def aggregate_indexer(indexer):
    """
    indexer: table, each row is ((old_block_id, old_row_id), (new_block_id, new_row_id))

    Returns:

    agg_indexer: key=old_block_id, value=(old_row_id, (new_block_id, new_row_id))
    """

    def _aggregate(kvs):
        aggregate_ret = dict()
        for k, values in kvs:
            old_msg, new_msg = values
            if old_msg[0] not in aggregate_ret:
                aggregate_ret[old_msg[0]] = []

            aggregate_ret[old_msg[0]].append([old_msg[1], new_msg])

        return list(aggregate_ret.items())

    agg_indexer = indexer.mapReducePartitions(_aggregate, lambda l1, l2: l1 + l2)

    return agg_indexer


def transform_to_table(block_table, block_index, partition_order_mappings):
    def _convert_to_order_index(kvs):
        order_indexes = []

        for block_id, blocks in kvs:
            for _idx, _id in enumerate(blocks[block_index]):
                order_indexes.append((_id, (block_id, _idx)))

        return order_indexes

    return block_table.mapPartitions(_convert_to_order_index,
                                     use_previous_behavior=False)


def get_partition_order_mappings_by_block_table(block_table, block_row_size):
    def _block_counter(kvs):
        partition_key = None
        size = 0
        first_block_id = 0
        for k, v in kvs:
            if partition_key is None:
                partition_key = k

            size += len(v[0])

        return first_block_id, (partition_key, size)

    block_info = sorted([summary[1] for summary in block_table.applyPartitions(_block_counter).collect()])
    block_order_mappings = dict()
    start_index = 0
    acc_block_num = 0
    for block_id, (block_key, block_size) in block_info:
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

    block_summary = table.mapPartitions(_get_block_summary).reduce(lambda blk1, blk2: {**blk1, **blk2})

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


def loc(df: DataFrame, indexer, target, preserve_order=False):
    self_indexer = df.get_indexer(target)
    if preserve_order:
        indexer = self_indexer.join(indexer, lambda lhs, rhs: (lhs, rhs))
    else:
        indexer = self_indexer.join(indexer, lambda lhs, rhs: (lhs, lhs))

    if indexer.count() == 0:
        return df.empty_frame()

    agg_indexer = aggregate_indexer(indexer)

    if not preserve_order:

        def _convert_block(blocks, retrieval_indexes):
            row_indexes = [retrieval_index[0] for retrieval_index in retrieval_indexes]
            return [Block.retrieval_row(block, row_indexes) for block in blocks]

        block_table = df.block_table.join(agg_indexer, _convert_block)
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
                        (dst_row_id, [Block.transform_row_to_raw(block, src_row_id) for block in blocks])
                    )

            for dst_block_id, value_list in ret_dict.items():
                yield dst_block_id, sorted(value_list)

        from ._transformer import transform_list_block_to_frame_block

        block_table = df.block_table.join(agg_indexer, lambda lhs, rhs: (lhs, rhs))
        block_table = block_table.mapReducePartitions(_convert_to_block, _merge_list)
        block_table = block_table.mapValues(lambda values: [v[1] for v in values])
        block_table = transform_list_block_to_frame_block(block_table, df.data_manager)

    partition_order_mappings = get_partition_order_mappings_by_block_table(block_table, df.data_manager.block_row_size)
    return DataFrame(
        df._ctx,
        block_table,
        partition_order_mappings,
        df.data_manager.duplicate())


def flatten_data(df: DataFrame, key_type="block_id", with_sample_id=True):
    """
    key_type="block_id":
        key=(block_id, block_offset), value=data_row
    """
    sample_id_index = df.data_manager.loc_block(
        df.data_manager.schema.sample_id_name, with_offset=False
    ) if with_sample_id else  None

    def _flatten_with_block_id_key(kvs):
        for block_id, blocks in kvs:
            for row_id in range(len(blocks[0])):
                if with_sample_id:
                    yield (block_id, row_id), (
                        blocks[sample_id_index][row_id],
                        [Block.transform_row_to_raw(block, row_id) for block in blocks]
                    )
                else:
                    yield (block_id, row_id), [Block.transform_row_to_raw(block, row_id) for block in blocks]

    """
    def _flatten_with_block_id_key(block_id, blocks):
        for row_id in range(len(blocks[0])):
            if with_sample_id:
                yield (block_id, row_id), (
                    blocks[sample_id_index][row_id],
                    [Block.transform_row_to_raw(block, row_id) for block in blocks]
                )
            else:
                yield (block_id, row_id), [Block.transform_row_to_raw(block, row_id) for block in blocks]
    """

    if key_type == "block_id":
        return df.block_table.mapPartitions(_flatten_with_block_id_key, use_previous_behavior=False)
        # return df.block_table.flatMap(_flatten_with_block_id_key)
    else:
        raise ValueError(f"Not Implement key_type={key_type} of flatten_data.")


def transform_flatten_data_to_df(ctx, flatten_table, data_manager: DataManager):
    partition_order_mappings = get_partition_order_by_raw_table(flatten_table,
                                                                data_manager.block_row_size,
                                                                key_type="block_id")
    block_num = data_manager.block_num

    def _convert_to_blocks(kvs):
        bid = None
        ret_blocks = [[] for _ in range(block_num)]

        lid = 0
        for _, (sample_id, data) in kvs:
            lid += 1
            if bid is None:
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
        data_manager=data_manager
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

        flat_ret.sort(key=lambda value: value[0])
        i = 0
        l = len(flat_ret)
        while i < l:
            j = i
            while j < l and flat_ret[i][0] == flat_ret[j][0]:
                j += 1

            agg_ret = [(flat_ret[k][1], flat_ret[k][2], flat_ret[k][3], flat_ret[k][4])
                       for k in range(i, j)]
            yield  flat_ret[i][0], agg_ret

            i = j
        """
        aggregate_ret = dict()
        offset = 0
        bid = None
        for k, values in kvs:
            sample_id, (src_block_id, src_offset) = values
            if bid is None:
                bid = partition_order_mappings[sample_id]["start_block_id"]
                
            if src_block_id not in aggregate_ret:
                aggregate_ret[src_block_id] = []
                
            aggregate_ret[src_block_id].append((src_offset, sample_id, bid, offset))
            
            offset += 1
            if offset == data_manager.block_row_size:
                bid += 1
                offset = 0

        return list(aggregate_ret.items())
        """
    
    sample_id_index = data_manager.loc_block(data_manager.schema.sample_id_name, with_offset=False)
    block_num = data_manager.block_num
    
    def _convert_to_row(kvs):
        ret_dict = {}
        for block_id, (blocks, block_indexer) in kvs:
            for src_row_id, sample_id, dst_block_id, dst_row_id in block_indexer:
                if dst_block_id not in ret_dict:
                    ret_dict[dst_block_id] = []

                row_data = []
                for i in range(block_num):
                    if i == sample_id_index:
                        row_data.append(sample_id)
                    elif data_manager.blocks[i].block_type == BlockType.index:
                        row_data.append(blocks[i][src_row_id])
                    else:
                        row_data.append(blocks[i][src_row_id].tolist())

                ret_dict[dst_block_id].append(
                    (dst_row_id, row_data)
                )

        for dst_block_id, value_list in ret_dict.items():
            yield dst_block_id, sorted(value_list)

    def _convert_to_frame_block(blocks):
        convert_blocks = []
        for idx, block_schema in enumerate(data_manager.blocks):
            block_content = [row_data[1][idx] for row_data in blocks]
            convert_blocks.append(block_schema.convert_block(block_content))

        return convert_blocks

    agg_indexer = indexer.mapReducePartitions(_aggregate, lambda l1, l2: l1 + l2)
    block_table = df.block_table.join(agg_indexer, lambda v1, v2: (v1, v2))
    block_table = block_table.mapReducePartitions(_convert_to_row, _merge_list)
    block_table = block_table.mapValues(_convert_to_frame_block)
    """
    block_table = block_table.mapValues(lambda values: [v[1] for v in values])
    
    from ._transformer import transform_list_block_to_frame_block
    block_table = transform_list_block_to_frame_block(block_table, df.data_manager)
    """
    
    return DataFrame(
        ctx=df._ctx,
        block_table=block_table,
        partition_order_mappings=partition_order_mappings,
        data_manager=data_manager
    )
