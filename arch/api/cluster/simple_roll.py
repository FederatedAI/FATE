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

import hashlib
import time
import uuid
from collections import Iterable
from functools import partial
from heapq import heapify, heappop, heapreplace
from itertools import tee
from operator import is_not

import grpc
from cachetools import cached, TTLCache

from arch.api.proto import kv_pb2_grpc, kv_pb2, processor_pb2_grpc, processor_pb2, storage_basic_pb2
from arch.api.proto.storage_basic_pb2 import StorageLocator
from arch.api.utils import cloudpickle as pickle, eggroll_serdes
from arch.api.utils import file_utils
from arch.api.utils.metric_utils import record_metrics

current_milli_time = lambda: int(round(time.time() * 1000))


def init(job_id=None, mode=None):
    EggRoll(job_id)


class EggRoll(object):
    __instance = None
    _serdes = eggroll_serdes.get_serdes()
    egg_list = []
    init_flag = False
    proc_list = []
    proc_egg_map = {}

    @staticmethod
    def get_instance():
        if EggRoll.__instance is None:
            raise EnvironmentError("eggroll should be initialized before use")
        return EggRoll.__instance

    def __init__(self, job_id):
        if EggRoll.__instance is not None:
            raise Exception("This class is a singleton!")
        EggRoll.init()
        self.job_id = str(uuid.uuid1()) if job_id is None else job_id
        self._meta_table = _DTable(self, storage_basic_pb2.LMDB, "__META__", "__META__", 10)
        EggRoll.__instance = self

    def table(self, name, namespace, partition=1, create_if_missing=True, error_if_exist=False, persistent=True):
        _type = storage_basic_pb2.LMDB if persistent else storage_basic_pb2.IN_MEMORY
        _table_key = "{}.{}.{}".format(_type, namespace, name)
        _old_partition = self._meta_table.put_if_absent(_table_key, partition)
        return _DTable(EggRoll.get_instance(), _type, namespace, name,
                       partition if _old_partition is None else _old_partition)

    def parallelize(self, data: Iterable, include_key=False, name=None, partition=1, namespace=None,
                    create_if_missing=True,
                    error_if_exist=False, persistent=False):
        eggroll = EggRoll.get_instance()
        if name is None:
            name = str(uuid.uuid1())
        if namespace is None and persistent:
            raise ValueError("namespace cannot be None for persistent table")
        elif namespace is None:
            namespace = eggroll.job_id
        _table = self.table(name, namespace, partition, persistent)
        _iter = data if include_key else enumerate(data)
        eggroll.put(_table, _iter)
        return _table

    def _merge(self, iters):
        ''' Merge sorted iterators. '''
        entries = []
        for _id, it in enumerate(map(iter, iters)):
            try:
                op = next(it)
                entries.append([op.key, op.value, _id, it])
            except StopIteration:
                pass
        heapify(entries)
        while entries:
            key, value, _, it = entry = entries[0]
            yield self._serdes.deserialize(key), self._serdes.deserialize(value)
            try:
                op = next(it)
                entry[0], entry[1] = op.key, op.value
                heapreplace(entries, entry)
            except StopIteration:
                heappop(entries)

    @staticmethod
    def init():
        if EggRoll.init_flag:
            return
        config = file_utils.load_json_conf('arch/conf/mock_roll.json')
        egg_ids = config.get('eggs')

        for egg_id in egg_ids:
            target = config.get('storage').get(egg_id)
            channel = grpc.insecure_channel(target, options=[('grpc.max_send_message_length', -1),
                                                             ('grpc.max_receive_message_length', -1)])
            EggRoll.egg_list.append(kv_pb2_grpc.KVServiceStub(channel))
            procs = config.get('procs').get(egg_id)
            for proc in procs:
                _channel = grpc.insecure_channel(proc, options=[('grpc.max_send_message_length', -1),
                                                                ('grpc.max_receive_message_length', -1)])
                _stub = processor_pb2_grpc.ProcessServiceStub(_channel)
                proc_info = (_channel, _stub)
                i = len(EggRoll.proc_list)
                EggRoll.proc_egg_map[i] = int(egg_id) - 1
                EggRoll.proc_list.append(proc_info)
        EggRoll.init_flag = True

    def serialize_and_hash_func(self, func):
        pickled_function = pickle.dumps(func)
        func_id = str(uuid.uuid1())
        return func_id, pickled_function

    @record_metrics
    def map(self, _table, func):
        func_id, func_bytes = self.serialize_and_hash_func(func)
        results = []

        for partition in range(_table.partition):
            operand = EggRoll.__get_storage_locator(_table, partition)
            unary_p = processor_pb2.UnaryProcess(operand=operand,
                                                 info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                             function_id=func_id + "_inter",
                                                                             function_bytes=func_bytes))

            proc_id = partition % len(self.proc_list)
            channel, stub = self.proc_list[proc_id]
            results.append(stub.map.future(unary_p))
        for r in results:
            result = r.result()

        return _DTable(self, result.type, result.namespace, result.name, _table.partition).save_as(func_id,
                                                                                                   result.namespace,
                                                                                                   _table.partition)

    @record_metrics
    def mapPartitions(self, _table, func):
        func_id, func_bytes = self.serialize_and_hash_func(func)
        results = []

        for partition in range(_table.partition):
            operand = EggRoll.__get_storage_locator(_table, partition)
            unary_p = processor_pb2.UnaryProcess(operand=operand,
                                                 info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                             function_id=func_id,
                                                                             function_bytes=func_bytes))

            proc_id = partition % len(self.proc_list)
            channel, stub = self.proc_list[proc_id]
            results.append(stub.mapPartitions.future(unary_p))
        for r in results:
            result = r.result()
        return _DTable(self, result.type, result.namespace, result.name, _table.partition)

    @record_metrics
    def mapValues(self, _table, func):
        func_id, func_bytes = self.serialize_and_hash_func(func)
        results = []
        for partition in range(_table.partition):
            operand = EggRoll.__get_storage_locator(_table, partition)
            unary_p = processor_pb2.UnaryProcess(operand=operand, info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                                              function_id=func_id,
                                                                                              function_bytes=func_bytes))

            proc_id = partition % len(self.proc_list)
            channel, stub = self.proc_list[proc_id]
            results.append(stub.mapValues.future(unary_p))

        for r in results:
            result = r.result()
        return _DTable(self, result.type, result.namespace, result.name, _table.partition)

    @record_metrics
    def glom(self, _table):
        results = []
        func_id = str(uuid.uuid1())
        for p in range(_table.partition):
            operand = EggRoll.__get_storage_locator(_table, p)
            unary_p = processor_pb2.UnaryProcess(operand=operand, info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                                              function_id=func_id))
            proc_id = p % len(self.proc_list)
            channel, stub = self.proc_list[proc_id]
            results.append(stub.glom.future(unary_p))
        for r in results:
            result = r.result()
        return _DTable(self, result.type, result.namespace, result.name, _table.partition)

    @record_metrics
    def sample(self, _table, fraction, seed):
        if fraction < 0 or fraction > 1:
            raise ValueError("fraction must be in [0, 1]")
        func_bytes = self._serdes.serialize((fraction, seed))
        results = []
        func_id = str(uuid.uuid1())
        for p in range(_table.partition):
            operand = EggRoll.__get_storage_locator(_table, p)
            unary_p = processor_pb2.UnaryProcess(operand=operand, info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                                              function_id=func_id,
                                                                                              function_bytes=func_bytes))
            proc_id = p % len(self.proc_list)
            channel, stub = self.proc_list[proc_id]
            results.append(stub.sample.future(unary_p))
        for r in results:
            result = r.result()
        return _DTable(self, result.type, result.namespace, result.name, _table.partition)

    @record_metrics
    def reduce(self, _table, func):
        func_id, func_bytes = self.serialize_and_hash_func(func)
        rtn = None
        results = []
        for partition in range(_table.partition):
            operand = EggRoll.__get_storage_locator(_table, partition)
            proc_id = partition % len(self.proc_list)
            channel, stub = self.proc_list[proc_id]
            unary_p = processor_pb2.UnaryProcess(operand=operand, info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                                              function_id=func_id,
                                                                                              function_bytes=func_bytes))
            results = results + list(stub.reduce(unary_p))
        rs = []
        for val in results:
            if len(val.value) > 0:
                rs.append(self._serdes.deserialize(val.value))
        rs = [r for r in filter(partial(is_not, None), rs)]
        if len(results) <= 0:
            return rtn
        rtn = rs[0]
        for r in rs[1:]:
            rtn = func(rtn, r)
        return rtn

    @record_metrics
    def join(self, left, right, func):
        func_id, func_bytes = self.serialize_and_hash_func(func)

        results = []
        res = None
        for partition in range(left.partition):
            l_op = EggRoll.__get_storage_locator(left, partition)
            r_op = EggRoll.__get_storage_locator(right, partition)
            binary_p = processor_pb2.BinaryProcess(left=l_op, right=r_op,
                                                   info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                               function_id=func_id,
                                                                               function_bytes=func_bytes))
            proc_id = partition % len(self.proc_list)
            channel, stub = self.proc_list[proc_id]
            results.append(stub.join.future(binary_p))
        for r in results:
            res = r.result()
        return _DTable(self, res.type, res.namespace, res.name, left.partition)

    @staticmethod
    def __get_storage_locator(_table, fragment=None):
        if fragment is None:
            fragment = _table.partition
        return StorageLocator(name=_table.name, namespace=_table.namespace, type=_table.type, fragment=fragment)

    def split_gen(self, _iter: Iterable, num):
        gens = tee(_iter, num)
        return (self.dispatch_gen(gen, i, num) for i, gen in enumerate(gens))

    def dispatch_gen(self, _iter: Iterable, p, total):
        for k, v in _iter:
            _p, i = self.__get_index(k, total)
            if _p == p:
                yield kv_pb2.Operand(key=self._serdes.serialize(k), value=self._serdes.serialize(v))

    def put(self, _table, kv_list):

        gens = self.split_gen(kv_list, _table.partition)
        results = []

        for p, gen in enumerate(gens):
            i = p % len(self.proc_list)
            stub = self.egg_list[i]
            meta = self.__get_meta(_table, str(p))
            stub.putAll(gen, metadata=meta)
        for r in results:
            r.result()
        return True

    def put_if_absent(self, _table, k, v):
        p, i = self.__get_index(k, _table.partition)
        stub = self.egg_list[i]
        meta = self.__get_meta(_table, str(p))
        rtn = stub.putIfAbsent(kv_pb2.Operand(key=self._serdes.serialize(k), value=self._serdes.serialize(v)),
                               metadata=meta).value
        rtn = self._serdes.deserialize(rtn) if len(rtn) > 0 else None
        return rtn

    def get(self, _table, k_list):
        res = []
        for k in k_list:
            p, i = self.__get_index(k, _table.partition)
            stub = self.egg_list[i]
            op = stub.get(kv_pb2.Operand(key=self._serdes.serialize(k)),
                          metadata=self.__get_meta(_table, str(p)))
            res.append(self.__get_pair(op))
        return res

    def delete(self, _table, k):
        p, i = self.__get_index(k, _table.partition)
        stub = self.egg_list[i]
        op = stub.delOne(kv_pb2.Operand(key=self._serdes.serialize(k)),
                         metadata=self.__get_meta(_table, str(p)))
        return self.__get_pair(op)

    def iterate(self, _table):
        iters = []
        for p in range(_table.partition):
            proc_id = p % len(EggRoll.proc_list)
            i = self.__get_index_by_proc(proc_id)
            stub = self.egg_list[i]
            iters.append(_PartitionIterator(stub, self.__get_meta(_table, str(p))))
        return self._merge(iters)

    def destroy(self, _table):
        for p in range(_table.partition):
            proc_id = p % len(EggRoll.proc_list)
            i = self.__get_index_by_proc(proc_id)
            stub = self.egg_list[i]
            stub.destroy(kv_pb2.Empty(), metadata=self.__get_meta(_table, str(p)))

    def count(self, _table):
        count = 0
        for p in range(_table.partition):
            proc_id = p % len(EggRoll.proc_list)
            i = self.__get_index_by_proc(proc_id)
            stub = self.egg_list[i]
            count += stub.count(kv_pb2.Empty(), metadata=self.__get_meta(_table, str(p))).value
        return count

    @staticmethod
    def __get_meta(_table, fragment):
        return ('store_type', _table.type), ('table_name', _table.name), ('name_space', _table.namespace), (
            'fragment', fragment)

    @cached(cache=TTLCache(maxsize=100, ttl=360))
    def __calc_hash(self, k):
        k_bytes = hashlib.sha1(self._serdes.serialize(k)).digest()
        return int.from_bytes(k_bytes, byteorder='little')

    def __key_to_partition(self, k, partitions):
        i = self.__calc_hash(k)
        return i % partitions

    @staticmethod
    def __get_index_by_proc(proc_id):
        egg_id = EggRoll.proc_egg_map[proc_id]
        return egg_id

    def __get_index(self, k, partitions):
        p, proc_id = self.__get_proc(k, partitions)
        return p, self.__get_index_by_proc(proc_id)

    def __get_proc(self, k, partitions):
        p = self.__key_to_partition(k, partitions)
        return p, p % len(self.proc_list)

    def __get_pair(self, op):
        return (self._serdes.deserialize(op.key), self._serdes.deserialize(op.value)) if len(
            op.value) > 0 else (self._serdes.deserialize(op.key), None)


class _DTable(object):

    def __init__(self, eggroll: EggRoll, _type: int, namespace, name, partition=1):
        self.type = storage_basic_pb2.StorageType.Name(_type)
        self.namespace = namespace
        self.name = name
        self.eggroll = eggroll
        self.partition = partition

    def __str__(self):
        return "type:{} namespace:{} table:{}".format(self.type, self.namespace, self.name)

    def save_as(self, name, namespace, partition=None):
        if partition is None:
            partition = self.partition
        res = EggRoll.get_instance().table(name, namespace, partition)
        res.put_all(self.collect())
        return res

    def map(self, func):
        res = self.eggroll.map(self, func)
        return res.save_as(str(uuid.uuid1()), res.namespace, partition=res.partition)

    def mapValues(self, func):
        res = self.eggroll.mapValues(self, func)
        return res

    def mapPartitions(self, func):
        return self.eggroll.mapPartitions(self, func)

    def put(self, k, v):
        return self.eggroll.put(self, [(k, v)])

    def put_all(self, kv_list):
        return self.eggroll.put(self, kv_list)

    def get(self, k):
        return self.eggroll.get(self, [k])[0]

    def collect(self):
        return self.eggroll.iterate(self)

    def delete(self, k_list):
        return self.eggroll.delete(self, k_list)[1]

    def destroy(self):
        self.eggroll.destroy(self)

    def reduce(self, func):
        return self.eggroll.reduce(self, func)

    def join(self, other, func):
        if other.partition != self.partition:
            if other.count() > self.count():
                return self.save_as(str(uuid.uuid1()), self.eggroll.job_id, partition=other.partition).join(other, func)
            else:
                return self.join(other.save_as(str(uuid.uuid1()), self.eggroll.job_id, partition=self.partition), func)
        return self.eggroll.join(self, other, func)

    def count(self):
        return self.eggroll.count(self)

    def glom(self):
        return self.eggroll.glom(self)

    def put_if_absent(self, k, v):
        return self.eggroll.put_if_absent(self, k, v)

    def sample(self, fraction, seed=None):
        return self.eggroll.sample(self, fraction, seed)


class _PartitionIterator(object):

    def __init__(self, stub, meta, start=None, end=None):
        self.stub = stub
        self.meta = meta
        self._start = start
        self._end = end
        self._cache = None
        self._index = 0
        self._next_item = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return self

    def __refresh_cache(self):
        if self._next_item is None:
            self._cache = list(self.stub.iterate(kv_pb2.Range(start=self._start, end=self._end), metadata=self.meta))
        else:
            self._cache = list(
                self.stub.iterate(kv_pb2.Range(start=self._next_item.key, end=self._end), metadata=self.meta))
        # if self._next_item is not None and len(self._cache) > 0 and self._cache[0].key == self._next_item.key:
        #     self._cache = self._cache[1:]
        if len(self._cache) == 0:
            raise StopIteration
        self._index = 0

    def __next__(self):
        if self._cache is None or self._index >= len(self._cache):
            self.__refresh_cache()
        self._next_item = self._cache[self._index]
        self._index += 1
        return self._next_item
