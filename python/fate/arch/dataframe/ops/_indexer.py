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

from ..manager import Block, BlockType
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
    # agg_indexer = agg_indexer.mapValues(lambda v: sorted(v, key=lambda x: x[1]))

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


def get_partition_order_mappings(block_table):
    block_info = sorted(list(block_table.mapValues(lambda blocks: (blocks[0][0], len(blocks[0]))).collect()))
    block_order_mappings = dict()
    start_index = 0
    for block_id, (block_key, block_size) in block_info:
        block_order_mappings[block_key] = dict(
            start_index=start_index, end_index=start_index + block_size - 1, block_id=block_id)
        start_index += block_size

    return block_order_mappings


def get_partition_order_by_raw_table(table):
    def _get_block_summary(kvs):
        try:
            key = next(kvs)[0]
            block_size = 1 + sum(1 for kv in kvs)
        except StopIteration:
            key, block_size = 0, 0

        return {key: block_size}

    block_summary = table.mapPartitions(_get_block_summary).reduce(lambda blk1, blk2: {**blk1, **blk2})

    start_index, block_id = 0, 0
    block_order_mappings = dict()
    for blk_key, blk_size in block_summary.items():
        block_order_mappings[blk_key] = dict(
            start_index=start_index, end_index=start_index + blk_size - 1, block_id=block_id
        )

        start_index += blk_size
        block_id += 1

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

    partition_order_mappings = get_partition_order_mappings(block_table)
    return DataFrame(
        df._ctx,
        block_table,
        partition_order_mappings,
        df.data_manager.duplicate())


def iloc(df: DataFrame, indexer, return_new_indexer=True):
    """
    indexer: (random_key, [(pid, offset), ..., ])
    """
    def _agg_mapper(kvs):
        agg_dict = dict()
        for _, id_loc_list in kvs:
            for pid, offset in id_loc_list:
                if pid not in agg_dict:
                    agg_dict[pid] = []
                if return_new_indexer:
                    agg_dict[pid].append((offset, _))
                else:
                    agg_dict[pid].append(offset)

        return list(agg_dict.items())

    def _agg_reducer(v1, v2):
        if not v1:
            return v2
        if not v2:
            return v1
        return v1 + v2

    data_manager = df.data_manager.duplicate()
    sample_id_block_id = df.data_manager.loc_block(df.schema.sample_id_name)[0]
    block_num = data_manager.block_num

    def _retrieval_mapper(key, value):
        retrieval_ret = []
        blocks, retrieval_list = value
        for _info in retrieval_list:
            offset = _info if not return_new_indexer else _info[0]
            sample_id = blocks[sample_id_block_id][offset]
            data = []
            for i in range(block_num):
                data.append([blocks[i][offset]] if isinstance(blocks[i], pd.Index) else blocks[i][offset].tolist())

            retrieval_ret.append((sample_id, (data, _info[1])) if return_new_indexer else (sample_id, data))

        return retrieval_ret

    agg_indexer = indexer.mapReducePartitions(_agg_mapper, _agg_reducer)

    raw_table = df.block_table.join(agg_indexer, lambda v1, v2: (v1, v2)).flatMap(_retrieval_mapper)

    partition_order_mappings = get_partition_order_by_raw_table(raw_table)

    def _convert_to_blocks(kvs):
        bid = None
        ret_blocks = [[] for _ in range(block_num)]

        for offset, (sample_id, data) in enumerate(kvs):
            if bid is None:
                bid = partition_order_mappings[sample_id]["block_id"]

            if return_new_indexer:
                data = data[0]

            for i in range(block_num):
                if data_manager.blocks[i].block_type == BlockType.index:
                    ret_blocks[i].append(data[i][0])
                else:
                    ret_blocks[i].append(data[i])

        ret_blocks = [data_manager.blocks[i].convert_block(block) for i, block in enumerate(ret_blocks)]

        return [(bid, ret_blocks)]

    block_table = raw_table.mapPartitions(_convert_to_blocks, use_previous_behavior=False)

    ret_df = DataFrame(
        df._ctx,
        block_table,
        partition_order_mappings,
        data_manager
    )

    if not return_new_indexer:
        return ret_df
    else:
        def _mapper(kvs):
            bid = None
            for offset, (sample_id, (_, k)) in enumerate(kvs):
                if bid is None:
                    bid = partition_order_mappings[sample_id]["block_id"]

                yield k, [(sample_id, bid, offset)]

        new_indexer = raw_table.mapReducePartitions(_mapper, lambda v1, v2: v1 + v2)

        return ret_df, new_indexer


def loc_with_sample_id_replacement(df: DataFrame, indexer):
    """
    indexer: table,
            row: (key=random_key,
            value=((src_partition_id, src_offset), [(sample_id, dst_partition_id, dst_offset) ...])
    """
    agg_indexer = aggregate_indexer(indexer)

    sample_id_index = df.data_manager.loc_block(df.schema.sample_id_name, with_offset=False)

    def _convert_to_block(kvs):
        ret_dict = {}
        for block_id, (blocks, block_indexer) in kvs:
            """
            block_indexer: row_id, [(sample_id, new_block_id, new_row_id)...]
            """

            for src_row_id, dst_indexer_list in block_indexer:
                for sample_id, dst_block_id, dst_row_id in dst_indexer_list:
                    if dst_block_id not in ret_dict:
                        ret_dict[dst_block_id] = []

                    data = []
                    for idx, block in enumerate(blocks):
                        if idx == sample_id_index:
                            data.append(sample_id)
                        else:
                            data.append(
                                block[src_row_id] if isinstance(block, pd.Index) else block[src_row_id].tolist()
                            )

                    ret_dict[dst_block_id].append((dst_row_id, data))

        for dst_block_id, value_list in ret_dict.items():
            yield dst_block_id, sorted(value_list)

    from ._transformer import transform_list_block_to_frame_block

    block_table = df.block_table.join(agg_indexer, lambda lhs, rhs: (lhs, rhs))
    block_table = block_table.mapReducePartitions(_convert_to_block, _merge_list)
    block_table = block_table.mapValues(lambda values: [v[1] for v in values])
    block_table = transform_list_block_to_frame_block(block_table, df.data_manager)

    partition_order_mappings = get_partition_order_mappings(block_table)
    return DataFrame(
        df._ctx,
        block_table,
        partition_order_mappings,
        df.data_manager.duplicate()
    )
