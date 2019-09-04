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

import uuid
from typing import Iterable

# noinspection PyProtectedMember
from arch.api.standalone.eggroll import _DTable
from arch.api.table.table import Table
from arch.api.utils.profile_util import log_elapsed


# noinspection PyProtectedMember,SpellCheckingInspection,PyPep8Naming
class DTable(Table):

    def __init__(self, dtable: _DTable, job_id):
        self._dtable = dtable
        self._partitions = self._dtable._partitions
        self.schema = {}
        self._name = self._dtable._name or str(uuid.uuid1())
        self._namespace = self._dtable._namespace
        self._job_id = job_id

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    @log_elapsed
    def save_as(self, name, namespace, partition=None, use_serialize=True, **kwargs):
        return self._dtable.save_as(name=name, namespace=namespace, partition=partition, use_serialize=use_serialize)

    def put(self, k, v, use_serialize=True, maybe_large_value=False):
        if not maybe_large_value:
            return self._dtable.put(k=k, v=v, use_serialize=use_serialize)
        else:
            from arch.api.table.storage_enhance import split_put
            return split_put(k, v, use_serialize=use_serialize, put_call_back_func=self._dtable.put)

    def put_all(self, kv_list: Iterable, use_serialize=True, chunk_size=100000):
        return self._dtable.put_all(kv_list=kv_list, use_serialize=use_serialize, chunk_size=chunk_size)

    def get(self, k, use_serialize=True, maybe_large_value=False):
        if not maybe_large_value:
            return self._dtable.get(k, use_serialize)
        else:
            from arch.api.table.storage_enhance import split_get
            return split_get(k=k, use_serialize=use_serialize, get_call_back_func=self._dtable.get)

    @log_elapsed
    def collect(self, min_chunk_size=0, use_serialize=True, **kwargs) -> list:
        return self._dtable.collect(min_chunk_size=min_chunk_size, use_serialize=use_serialize)

    def delete(self, k, use_serialize=True):
        return self._dtable.delete(k=k, use_serialize=use_serialize)

    def destroy(self):
        return self._dtable.destroy()

    @log_elapsed
    def count(self, **kwargs):
        return self._dtable.count()

    def put_if_absent(self, k, v, use_serialize=True):
        return self._dtable.put_if_absent(k=k, v=v, use_serialize=use_serialize)

    def take(self, n=1, keysOnly=False, use_serialize=True):
        return self._dtable.take(n=n, keysOnly=keysOnly, use_serialize=use_serialize)

    def first(self, keysOnly=False, use_serialize=True):
        return self._dtable.first(keysOnly=keysOnly, use_serialize=use_serialize)

    # noinspection PyProtectedMember
    def get_partitions(self):
        return self._dtable._partitions

    """
    computing apis
    """

    @log_elapsed
    def map(self, func, **kwargs):
        return DTable(self._dtable.map(func), job_id=self._job_id)

    @log_elapsed
    def mapValues(self, func, **kwargs):
        return DTable(self._dtable.mapValues(func), job_id=self._job_id)

    @log_elapsed
    def mapPartitions(self, func, **kwargs):
        return DTable(self._dtable.mapPartitions(func), job_id=self._job_id)

    @log_elapsed
    def reduce(self, func, **kwargs):
        return self._dtable.reduce(func)

    @log_elapsed
    def join(self, other, func, **kwargs):
        return DTable(self._dtable.join(other._dtable, func=func), job_id=self._job_id)

    @log_elapsed
    def glom(self, **kwargs):
        return DTable(self._dtable.glom(), job_id=self._job_id)

    @log_elapsed
    def sample(self, fraction, seed=None, **kwargs):
        return DTable(self._dtable.sample(fraction=fraction, seed=seed), job_id=self._job_id)

    @log_elapsed
    def subtractByKey(self, other, **kwargs):
        return DTable(self._dtable.subtractByKey(other._dtable), job_id=self._job_id)

    @log_elapsed
    def filter(self, func, **kwargs):
        return DTable(self._dtable.filter(func), job_id=self._job_id)

    @log_elapsed
    def union(self, other, func=lambda v1, v2: v1, **kwargs):
        return DTable(self._dtable.union(other._dtable, func=func), job_id=self._job_id)

    @log_elapsed
    def flatMap(self, func, **kwargs):
        return DTable(self._dtable.flatMap(func), job_id=self._job_id)
