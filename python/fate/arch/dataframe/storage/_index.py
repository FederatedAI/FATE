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
import functools

import pandas as pd
from fate.arch.computing import is_table


class Index(object):
    def __init__(self, ctx, distributed_index, block_partition_mapping, global_ranks):
        self._ctx = ctx
        self._index_table = distributed_index
        self._block_partition_mapping = block_partition_mapping
        self._global_ranks = global_ranks
        self._count = None

    @property
    def global_ranks(self):
        return self._global_ranks

    @property
    def block_partition_mapping(self):
        return self._block_partition_mapping

    @property
    def values(self):
        return self._index_table

    def count(self):
        if self._count is not None:
            return self._count

        self._count = self._index_table.count()

        return self._count

    def __len__(self):
        return self.count()

    def tolist(self):
        indexes_with_partition_id = sorted(self._index_table.collect(), key=lambda kv: kv[1])
        id_list = [k for k, v in indexes_with_partition_id]

        return id_list

    def to_local(self):
        """
        index_table: id, (partition_id, block_index)
        """
        index_table = self._index_table.mapValues(
            lambda order_tuple: (0, self._global_ranks[order_tuple[0]]["start_index"] + order_tuple[1])
        )

        global_ranks = [dict(start_index=0, end_index=self.count(), block_id=0)]
        block_partition_mapping = copy.deepcopy(self._block_partition_mapping)
        for block_id in self._block_partition_mapping:
            if block_id != 0:
                block_partition_mapping.pop(block_id)

        return Index(self._ctx, index_table, block_partition_mapping, global_ranks)

    def __getitem__(self, items):
        if isinstance(items, int):
            items = [items]

        # NOTE: it will not repartition automatically, user should call it in DataFrame if need
        # TODO: make sure that items is non-overlapped in this version
        if isinstance(items, list):
            items_set = set(items)
            index_table = self._index_table.filter(lambda kv: kv[0] in items_set)

        elif is_table(items):
            index_table = self._index_table.join(items, lambda v1, v2: v2[1])
        else:
            raise ValueError(f"get item does not support {type(items)}")

        agg_table = self.aggregate(index_table)
        global_ranks = self.regenerate_global_ranks(agg_table, self._global_ranks)
        block_partition_mapping = self.regenerate_block_partition_mapping(agg_table, global_ranks)

        def _flat_partition(k, values, ranks=None):
            start_index = ranks[k]["start_index"]
            _flat_ret = []
            for idx, (_id, block_index) in enumerate(values):
                _flat_ret.append((_id, (k, start_index + idx)))

            return _flat_ret

        _flat_func = functools.partial(_flat_partition, ranks=global_ranks)
        index_table = agg_table.flatMap(_flat_func)

        return Index(self._ctx, index_table, block_partition_mapping, global_ranks)

    def get_indexer(self, ids, with_partition_id=True):
        if isinstance(ids, list):

            def _filter_id(key, value, ids_set=None):
                return key in ids_set

            filter_func = functools.partial(_filter_id, ids_set=set(ids))
            indexer = self._index_table.filter(filter_func)
            indexer = indexer.mapValues(lambda v: [v, v])

        elif is_table(ids):
            """ """
            if with_partition_id:
                indexer = self._index_table.join(ids, lambda v1, v2: [v1, v2])
            else:
                indexer = self._index_table.join(ids, lambda v1, v2: v1)
                indexer = indexer.mapValues(lambda v: [v, v])
        else:
            raise ValueError(f"get_indexer's args type is {type(ids)}, is not supported")

        return indexer

    def change_indexes_to_indexer(self, indexes):
        def _filter(k, v, index_set=None, global_ranks=None):
            partition_id, block_index = v
            return global_ranks[partition_id]["start_index"] + block_index in index_set

        filter_func = functools.partial(_filter, index_set=set(indexes), global_ranks=self._global_ranks)
        indexer = self._index_table.filter(filter_func, use_previous_behavior=False)
        indexer = indexer.mapValues(lambda v: [v, v])
        return indexer

    @classmethod
    def aggregate(cls, table):
        """
        agg_table: key=partition_id, value=(id, block_index), block_index may be not continuous
        """

        def _aggregate_ids(kvs):
            aggregate_ret = dict()

            for k, v in kvs:
                partition_id, block_index = v
                if partition_id not in aggregate_ret:
                    aggregate_ret[partition_id] = []

                aggregate_ret[partition_id].append((k, block_index))

            return list(aggregate_ret.items())

        agg_table = table.mapReducePartitions(_aggregate_ids, lambda l1, l2: l1 + l2)
        agg_table = agg_table.mapValues(
            lambda id_list: sorted(id_list, key=lambda block_index_with_key: block_index_with_key[1])
        )

        return agg_table

    @classmethod
    def aggregate_indexer(cls, indexer):
        """
        key=id, value=[(old_partition_id, old_block_index), (new_partition_id, new_block_index)]
        =>
        key=old_partition_id, value=[old_block_index, (new_partition_id, new_block_index)]
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
        agg_indexer = agg_indexer.mapValues(lambda v: sorted(v))
        return agg_indexer

    @classmethod
    def regenerate_global_ranks(cls, agg_table, old_global_ranks):
        """
        input should be agg_table instead of index_table
        """
        block_counts = sorted(list(agg_table.mapValues(lambda v: len(v)).collect()))
        global_ranks = []

        idx = 0
        for block_id, block_count in block_counts:
            if global_ranks and global_ranks[-1]["block_id"] + 1 != block_id:
                last_bid = global_ranks[-1]["block_id"]
                for bid in range(last_bid + 1, block_id):
                    global_ranks.append(dict(start_index=idx, end_index=idx - 1, block_id=bid))

            global_ranks.append(dict(start_index=idx, end_index=idx + block_count - 1, block_id=block_id))
            idx += block_count

        if len(global_ranks) < len(old_global_ranks):
            last_bid = len(global_ranks)
            for bid in range(last_bid, len(old_global_ranks)):
                global_ranks.append(dict(start_index=idx, end_index=idx - 1, block_id=bid))

        return global_ranks

    @classmethod
    def regenerate_block_partition_mapping(cls, agg_table, global_ranks):
        """
        input should be agg_table instead of index_table
        """
        blocks = agg_table.mapValues(lambda v: v[0]).collect()
        block_partition_mapping = dict()
        for partition_id, (key, block_index) in blocks:
            block_partition_mapping[key] = global_ranks[partition_id]

        return block_partition_mapping
