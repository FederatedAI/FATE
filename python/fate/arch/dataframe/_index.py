class Index(object):
    def __init__(self, ctx, distributed_index, block_partition_mapping):
        self._ctx = ctx
        self._index_table = distributed_index
        self._block_partition_mapping = block_partition_mapping

        self._count = None

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
            [(0, id_list)],
            include_key=True,
            partition=1
        )
        return Index(
            self._ctx,
            index_table,
            block_partition_mapping={
                id_list[0]: dict(
                    start_index=0,
                    end_index=self.count() - 1,
                    block_id=0
                )
            }
        )
