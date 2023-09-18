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
import random


class DataLoader(object):
    def __init__(
        self,
        dataset,
        ctx=None,
        mode="homo",
        role="guest",
        need_align=False,
        sync_arbiter=False,
        batch_size=-1,
        shuffle=False,
        batch_strategy="full",
        random_state=None,
    ):
        self._ctx = ctx
        self._dataset = dataset
        self._batch_size = batch_size
        if dataset:
            if batch_size is None:
                self._batch_size = len(dataset)
            else:
                self._batch_size = min(batch_size, len(dataset))
        self._shuffle = shuffle
        self._batch_strategy = batch_strategy
        self._random_state = random_state
        self._need_align = need_align
        self._mode = mode
        self._role = role
        self._sync_arbiter = sync_arbiter

        self._init_settings()

    def _init_settings(self):
        if self._batch_strategy == "full":
            self._batch_generator = FullBatchDataLoader(
                self._dataset,
                self._ctx,
                mode=self._mode,
                role=self._role,
                batch_size=self._batch_size,
                shuffle=self._shuffle,
                random_state=self._random_state,
                need_align=self._need_align,
                sync_arbiter=self._sync_arbiter,
            )
        else:
            raise ValueError(f"batch strategy {self._batch_strategy} is not support")

    @staticmethod
    def batch_num(self):
        return self._batch_generator.batch_num

    def __next__(self):
        for batch in self._batch_generator:
            yield batch

    def __iter__(self):
        for batch in self._batch_generator:
            yield batch


class FullBatchDataLoader(object):
    def __init__(self, dataset, ctx, mode, role, batch_size, shuffle, random_state, need_align, sync_arbiter):
        self._dataset = dataset
        self._ctx = ctx
        self._mode = mode
        self._role = role
        self._batch_size = batch_size
        if self._batch_size is None and self._role != "arbiter":
            self._batch_size = len(self._dataset)
        self._shuffle = shuffle
        self._random_state = random_state
        self._need_align = need_align
        self._sync_arbiter = sync_arbiter

        self._batch_num = None
        self._batch_splits = []  # list of DataFrame
        self._prepare()

    def _prepare(self):
        if self._mode == "homo":
            if self._role == "arbiter":
                batch_info = self._ctx.arbiter.get("batch_info")
                self._batch_size = batch_info["batch_size"]
                self._batch_num = batch_info["batch_num"]
            elif self._role == "guest":
                self._batch_num = (len(self._dataset) + self._batch_size - 1) // self._batch_size
                self._ctx.arbiter.put("batch_num", self._batch_num)
        elif self._mode == "local":
            self._batch_num = (len(self._dataset) + self._batch_size - 1) // self._batch_size
        elif self._mode == "hetero":
            # NOTE: index should be align first, using after doing psi
            if self._role != "arbiter":
                self._batch_num = (len(self._dataset) + self._batch_size - 1) // self._batch_size
                if self._role == "guest" and self._sync_arbiter:
                    self._ctx.arbiter.put("batch_num", self._batch_num)
            elif self._sync_arbiter:
                self._batch_num = self._ctx.guest.get("batch_num")

        if self._role == "arbiter":
            return

        if self._batch_size == len(self._dataset):
            self._batch_splits.append(BatchEncoding(self._dataset, batch_id=0))
        else:
            if self._mode in ["homo", "local"] or self._role == "guest":
                indexer = sorted(list(self._dataset.get_indexer(target="sample_id").collect()))
                if self._shuffle:
                    random.seed = self._random_state
                    random.shuffle(indexer)

                for i, iter_ctx in self._ctx.sub_ctx("dataloader_batch").ctxs_range(self._batch_num):
                    batch_indexer = indexer[self._batch_size * i: self._batch_size * (i + 1)]
                    batch_indexer = self._ctx.computing.parallelize(batch_indexer,
                                                                    include_key=True,
                                                                    partition=self._dataset.block_table.partitions)

                    sub_frame = self._dataset.loc(batch_indexer, preserve_order=False)

                    if self._mode == "hetero" and self._role == "guest":
                        iter_ctx.hosts.put("batch_indexes", sub_frame.get_indexer(target="sample_id"))

                    self._batch_splits.append(BatchEncoding(sub_frame, batch_id=i))

            elif self._mode == "hetero" and self._role == "host":
                for i, iter_ctx in self._ctx.sub_ctx("dataloader_batch").ctxs_range(self._batch_num):
                    batch_indexes = iter_ctx.guest.get("batch_indexes")
                    sub_frame = self._dataset.loc(batch_indexes, preserve_order=True)
                    self._batch_splits.append(BatchEncoding(sub_frame, batch_id=i))

    def __next__(self):
        if self._role == "arbiter":
            for batch_id in range(self._batch_num):
                yield BatchEncoding(batch_id=batch_id)
            return

        for batch in self._batch_splits:
            yield batch

    def __iter__(self):
        return self.__next__()

    @property
    def batch_num(self):
        return self._batch_num


class BatchEncoding(object):
    def __init__(self, batch_df=None, batch_id=None):
        if batch_df:
            self._x = batch_df.values.as_tensor()
            self._label = batch_df.label.as_tensor() if batch_df.label else None
            self._weight = batch_df.weight.as_tensor() if batch_df.weight else None
        else:
            self._x = None
            self._label = None
            self._weight = None

        self._batch_id = batch_id

    @property
    def x(self):
        return self._x

    @property
    def label(self):
        return self._label

    @property
    def weight(self):
        return self._weight

    @property
    def batch_id(self):
        return self._batch_id
