import functools
import pandas as pd


class ValueStore(object):
    def __init__(self, ctx, distributed_table, header):
        self._ctx = ctx
        self._header = header
        self._data = distributed_table

    def to_local(self, keep_table=False):
        if self._data.partitions == 1 and keep_table:
            return self

        frames = [frame for partition_id, frame in sorted(self._data.collect())]
        concat_frame = pd.concat(frames)

        if not keep_table:
            return concat_frame
        else:
            table = self._ctx.computing.parallelize(
                [(0, concat_frame)],
                include_key=True,
                partition=1
            )

            return ValueStore(
                self._ctx,
                table,
                self._header
            )

    def __getattr__(self, attr):
        if attr not in self._header:
            raise ValueError(f"ValueStore does not has attribute: {attr}")

        return ValueStore(
            self._ctx,
            self._data.mapValues(lambda df: df[attr]),
            [attr]
        )

    def tolist(self):
        return self.to_local().tolist()

    @staticmethod
    def values(self):
        return self._data
