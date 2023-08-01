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
    agg_indexer = agg_indexer.mapValues(lambda v: sorted(v, key=lambda x: x[1]))

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
