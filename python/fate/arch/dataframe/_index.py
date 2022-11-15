import functools
import pandas as pd


class Index(object):
    def __init__(self, ctx, distributed_index, global_ranks):
        self._ctx = ctx
        self._index_table = distributed_index
        self._global_ranks = global_ranks
        self._count = None

    @property
    def global_ranks(self):
        return self._global_ranks

    @property
    def values(self):
        return self._index_table

    def count(self):
        if self._count is not None:
            return self._count

        self._count = self._index_table.mapValues(lambda index: len(index)).reduce(lambda x, y: x + y)
        return self._count

    def __len__(self):
        return self.count()

    def tolist(self):
        indexes_with_partition_id = sorted(self._index_table.collect())
        id_list = []
        for part_index in indexes_with_partition_id:
            id_list.extend(part_index[1].tolist())

        return id_list

    def to_local(self):
        id_list = self.tolist()
        index_table = self._ctx.computing.parallelize(
            [(0, pd.Index(id_list))],
            include_key=True,
            partition=1
        )
        return Index(
            self._ctx,
            index_table,
            [dict(start_index=0,
                  end_index=self.count(),
                  block_id=0)
             ]
        )

    def __getitem__(self, items):
        if isinstance(items, int):
            items = [items]

        def _filter_id(kvs, index_set=None, global_ranks=None):
            bid, indexer = list(kvs)[0]
            start_index = global_ranks[bid]["start_index"]
            end_index = global_ranks[bid]["end_index"]
            filter_indexes = []
            for _idx in range(start_index, end_index + 1):
                if _idx in index_set:
                    filter_indexes.append(_idx - start_index)

            if filter_indexes:
                indexer = indexer[filter_indexes]

            return [(bid, indexer)]

        filter_func = functools.partial(_filter_id,
                                        index_set=set(items),
                                        global_ranks=self._global_ranks)
        filter_index_table = self._index_table.mapPartitions(filter_func,
                                                             use_previous_behavior=False)
        block_counts = sorted(list(filter_index_table.mapValues(lambda v: len(v)).collect()))
        filter_global_ranks = []
        idx = 0
        for block_id, block_count in block_counts:
            filter_global_ranks.append(
                dict(
                    start_index=idx,
                    end_index=idx + block_count - 1,
                    block_id=block_id
                )
            )
            idx += block_count

        return Index(self._ctx,
                     filter_index_table,
                     filter_global_ranks
                     )

    def get_indexer(self, ids):
        def _get_indexer(kvs, ids_set=None, global_ranks=None):
            _indexer = []
            block_id, block_index = list(kvs)[0]
            start_idx = global_ranks[block_id]["start_index"]
            for offset, _id in enumerate(block_index):
                if _id in ids_set:
                    _indexer.append(start_idx + offset)

            return _indexer

        get_indexer_func = functools.partial(_get_indexer,
                                             ids_set=set(ids),
                                             global_ranks=self._global_ranks)

        return self._index_table.mapPartitions(get_indexer_func).reduce(lambda l1, l2: l1 + l2)
