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
from .._dataframe import DataFrame
from sklearn.model_selection import KFold as sk_KFold


class KFold(object):
    def __init__(self, ctx, mode="hetero", role="guest", n_splits=5, shuffle=False, random_state=None):
        self._ctx = ctx
        self._mode = mode
        self._role = role
        self._n_splits = n_splits
        self._shuffle = shuffle
        self._random_state = random_state

        self._check_param()

    def split(self, df: DataFrame):
        if self._mode == "hetero":
            return self._hetero_split(df)
        else:
            return self._homo_split(df, return_indexer=False)

    def _hetero_split(self, df: DataFrame):
        if self._role == "guest":
            homo_splits = self._homo_split(df, return_indexer=True)
            for _, iter_ctx in self._ctx.sub_ctx("KFold").ctxs_range(self._n_splits):
                train_frame, test_frame, train_indexer, test_indexer = next(homo_splits)

                iter_ctx.hosts.put("fold_indexes", (train_indexer, test_indexer))

                yield train_frame, test_frame
        else:
            for _, iter_ctx in self._ctx.sub_ctx("KFold").ctxs_range(self._n_splits):
                train_indexer, test_indexer = iter_ctx.guest.get("fold_indexes")
                train_frame = df.loc(train_indexer, preserve_order=True)
                test_frame = df.loc(test_indexer, preserve_order=True)

                yield train_frame, test_frame

    def _homo_split(self, df: DataFrame, return_indexer):
        kf = sk_KFold(n_splits=self._n_splits, shuffle=self._shuffle, random_state=self._random_state)
        indexer = list(df.get_indexer(target="sample_id").collect())

        for train, test in kf.split(indexer):
            train_indexer = [indexer[idx] for idx in train]
            test_indexer = [indexer[idx] for idx in test]

            train_indexer = self._ctx.computing.parallelize(
                train_indexer, include_key=True, partition=df.block_table.num_partitions
            )

            test_indexer = self._ctx.computing.parallelize(
                test_indexer, include_key=True, partition=df.block_table.num_partitions
            )

            train_frame = df.loc(train_indexer)
            test_frame = df.loc(test_indexer)

            if return_indexer:
                yield train_frame, test_frame, train_frame.get_indexer(target="sample_id"), test_frame.get_indexer(
                    target="sample_id"
                )
            else:
                yield train_frame, test_frame

    def _check_param(self):
        if not isinstance(self._n_splits, int) or self._n_splits < 2:
            raise ValueError("n_splits should be positive integer >= 2")
