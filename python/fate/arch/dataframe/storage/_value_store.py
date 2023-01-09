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


class ValueStore(object):
    def __init__(self, ctx, distributed_table, header):
        self._ctx = ctx
        self._header = header
        self._data = distributed_table
        self._dtypes = None

    def to_local(self, keep_table=False):
        if self._data.partitions == 1 and keep_table:
            return self

        frames = [frame for partition_id, frame in sorted(self._data.collect())]
        concat_frame = pd.concat(frames)

        if not keep_table:
            return concat_frame
        else:
            table = self._ctx.computing.parallelize([(0, concat_frame)], include_key=True, partition=1)

            return ValueStore(self._ctx, table, self._header)

    def __getattr__(self, attr):
        if attr not in self._header:
            raise ValueError(f"ValueStore does not has attribute: {attr}")

        return ValueStore(self._ctx, self._data.mapValues(lambda df: df[attr]), [attr])

    def tolist(self):
        return self.to_local().tolist()

    @property
    def dtypes(self):
        if self._dtypes is None:
            self._dtypes = self._data.first()[1].dtypes

        return self._dtypes

    @property
    def values(self):
        return self._data
