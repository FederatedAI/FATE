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
from functools import partial
from operator import is_not
from typing import Iterable

import grpc

from arch.api.utils import eggroll_serdes, file_utils
from arch.api.utils.log_utils import getLogger
from arch.api.proto import kv_pb2, kv_pb2_grpc, processor_pb2, processor_pb2_grpc, storage_basic_pb2
from arch.api.utils import cloudpickle
from arch.api.utils.core import string_to_bytes, bytes_to_string
from arch.api.utils.iter_utils import split_every


def init(job_id=None, server_conf_path="arch/conf/server_conf.json"):
    if job_id is None:
        job_id = str(uuid.uuid1())
    global LOGGER
    LOGGER = getLogger()
    server_conf = file_utils.load_json_conf(server_conf_path)
    _roll_host = server_conf.get("servers").get("roll").get("host")
    _roll_port = server_conf.get("servers").get("roll").get("port")
    _EggRoll(job_id, _roll_host, _roll_port)


def _get_meta(_table):
    return ('store_type', _table._type), ('table_name', _table._name), ('name_space', _table._namespace)


empty = kv_pb2.Empty()


class _DTable(object):

    def __init__(self, storage_locator, partitions=1):
        # self.__client = _EggRoll.get_instance()
        self._namespace = storage_locator.namespace
        self._name = storage_locator.name
        self._type = storage_basic_pb2.StorageType.Name(storage_locator.type)
        self._partitions = partitions
        self.schema = {}

    def __str__(self):
        return "type:{} namespace:{} name:{} partitions:{}".format(self._type, self._namespace, self._name,
                                                                   self._partitions)

    '''
    Storage apis
    '''

    def save_as(self, name, namespace, partition=None, use_serialize=True):
        if partition is None:
            partition = self._partitions
        dup = _EggRoll.get_instance().table(name, namespace, partition=partition)
        dup.put_all(self.collect(use_serialize=use_serialize), use_serialize=use_serialize)
        return dup

    def put(self, k, v, use_serialize=True):
        _EggRoll.get_instance().put(self, k, v, use_serialize=use_serialize)

    def put_all(self, kv_list: Iterable, use_serialize=True, chunk_size=100000):
        return _EggRoll.get_instance().put_all(self, kv_list, use_serialize=use_serialize, chunk_size=chunk_size)

    def get(self, k, use_serialize=True):
        return _EggRoll.get_instance().get(self, k, use_serialize=use_serialize)

    def collect(self, use_serialize=True):
        return _EggRollIterator(self, use_serialize=use_serialize)

    def delete(self, k, use_serialize=True):
        return _EggRoll.get_instance().delete(self, k, use_serialize=use_serialize)

    def destroy(self):
        _EggRoll.get_instance().destroy(self)

    def count(self):
        return _EggRoll.get_instance().count(self)

    def put_if_absent(self, k, v, use_serialize=True):
        return _EggRoll.get_instance().put_if_absent(self, k, v, use_serialize=use_serialize)

    '''
    Computing apis
    '''

    def map(self, func):
        _intermediate_result = _EggRoll.get_instance().map(self, func)
        return _intermediate_result.save_as(str(uuid.uuid1()), _intermediate_result._namespace,
                                            partition=_intermediate_result._partitions)

    def mapValues(self, func):
        return _EggRoll.get_instance().map_values(self, func)

    def mapPartitions(self, func):
        return _EggRoll.get_instance().map_partitions(self, func)

    def reduce(self, func):
        return _EggRoll.get_instance().reduce(self, func)

    def join(self, other, func):
        if other._partitions != self._partitions:
            if other.count() > self.count():
                return self.save_as(str(uuid.uuid1()), _EggRoll.get_instance().job_id, partition=other._partitions).join(other,
                                                                                                               func)
            else:
                return self.join(other.save_as(str(uuid.uuid1()), _EggRoll.get_instance().job_id, partition=self._partitions),
                                 func)
        return _EggRoll.get_instance().join(self, other, func)

    def glom(self):
        return _EggRoll.get_instance().glom(self)

    def sample(self, fraction, seed=None):
        return _EggRoll.get_instance().sample(self, fraction, seed)


class _EggRoll(object):
    value_serdes = eggroll_serdes.get_serdes()
    instance = None

    @staticmethod
    def get_instance():
        if _EggRoll.instance is None:
            raise EnvironmentError("eggroll should be initialized before use")
        return _EggRoll.instance

    def __init__(self, job_id, host, port):
        if _EggRoll.instance is not None:
            raise EnvironmentError("eggroll should be initialized only once")
        self.channel = grpc.insecure_channel(target="{}:{}".format(host, port),
                                             options=[('grpc.max_send_message_length', -1),
                                                      ('grpc.max_receive_message_length', -1)])
        self.job_id = job_id
        self.kv_stub = kv_pb2_grpc.KVServiceStub(self.channel)
        self.proc_stub = processor_pb2_grpc.ProcessServiceStub(self.channel)
        _EggRoll.instance = self

    def table(self, name, namespace, partition=1, create_if_missing=True, error_if_exist=False, persistent=True):
        _type = storage_basic_pb2.LMDB if persistent else storage_basic_pb2.IN_MEMORY
        storage_locator = storage_basic_pb2.StorageLocator(type=_type, namespace=namespace, name=name)
        create_table_info = kv_pb2.CreateTableInfo(storageLocator=storage_locator, fragmentCount=partition)
        _table = self._create_table(create_table_info)
        LOGGER.debug("created table: %s", _table)
        return _table

    def parallelize(self, data: Iterable, include_key=False, name=None, partition=1, namespace=None,
                    create_if_missing=True,
                    error_if_exist=False, persistent=False, chunk_size=100000):
        if namespace is None:
            namespace = _EggRoll.get_instance().job_id
        if name is None:
            name = str(uuid.uuid1())
        storage_locator = storage_basic_pb2.StorageLocator(type=storage_basic_pb2.LMDB, namespace=namespace,
                                                           name=name) if persistent else storage_basic_pb2.StorageLocator(
            type=storage_basic_pb2.IN_MEMORY, namespace=namespace, name=name)
        create_table_info = kv_pb2.CreateTableInfo(storageLocator=storage_locator, fragmentCount=partition)
        _table = self._create_table(create_table_info)
        _iter = data if include_key else enumerate(data)
        _table.put_all(_iter, chunk_size=chunk_size)
        LOGGER.debug("created table: %s", _table)
        return _table

    def cleanup(self, name, namespace, persistent):
        if namespace is None or name is None:
            raise ValueError("neither name nor namespace can be None")

        _type = storage_basic_pb2.LMDB if persistent else storage_basic_pb2.IN_MEMORY

        storage_locator = storage_basic_pb2.StorageLocator(type=_type, namespace=namespace, name=name)
        _table = _DTable(storage_locator=storage_locator)

        self.destroy_all(_table)

        LOGGER.debug("cleaned up: %s", _table)


    @staticmethod
    def serialize_and_hash_func(func):
        pickled_function = cloudpickle.dumps(func)
        func_id = str(uuid.uuid1())
        return func_id, pickled_function

    def _create_table(self, create_table_info):
        info = self.kv_stub.createIfAbsent(create_table_info)
        return _DTable(info.storageLocator, info.fragmentCount)

    def _create_table_from_locator(self, storage_locator, partitions):
        create_table_info = kv_pb2.CreateTableInfo(storageLocator=storage_locator, fragmentCount=partitions)
        return self._create_table(create_table_info)

    @staticmethod
    def __generate_operand(kvs: Iterable, use_serialize=True):
        for k, v in kvs:
            yield kv_pb2.Operand(key=_EggRoll.value_serdes.serialize(k) if use_serialize else bytes_to_string(k), value=_EggRoll.value_serdes.serialize(v) if use_serialize else v)

    @staticmethod
    def _deserialize_operand(operand: kv_pb2.Operand, include_key=False, use_serialize=True):
        if operand.value and len(operand.value) > 0:
            if use_serialize:
                return (_EggRoll.value_serdes.deserialize(operand.key), _EggRoll.value_serdes.deserialize(
                    operand.value)) if include_key else _EggRoll.value_serdes.deserialize(operand.value)
            else:
                return (bytes_to_string(operand.key), operand.value) if include_key else operand.value
        return None

    '''
    Storage apis
    '''

    def kv_to_bytes(self, **kwargs):
        use_serialize = kwargs.get("use_serialize", True)
        # can not use is None
        if "k" in kwargs and "v" in kwargs:
            k, v = kwargs["k"], kwargs["v"]
            return (self.value_serdes.serialize(k), self.value_serdes.serialize(v)) if use_serialize \
                else (string_to_bytes(k), string_to_bytes(v))
        elif "k" in kwargs:
            k = kwargs["k"]
            return self.value_serdes.serialize(k) if use_serialize else string_to_bytes(k)
        elif "v" in kwargs:
            v = kwargs["v"]
            return self.value_serdes.serialize(v) if use_serialize else string_to_bytes(v)

    def put(self, _table, k, v, use_serialize=True):
        k, v = self.kv_to_bytes(k=k, v=v, use_serialize=use_serialize)
        self.kv_stub.put(kv_pb2.Operand(key=k, value=v), metadata=_get_meta(_table))

    def put_if_absent(self, _table, k, v, use_serialize=True):
        k, v = self.kv_to_bytes(k=k, v=v, use_serialize=use_serialize)
        operand = self.kv_stub.putIfAbsent(kv_pb2.Operand(key=k, value=v), metadata=_get_meta(_table))
        return self._deserialize_operand(operand, use_serialize=use_serialize)

    def put_all(self, _table, kvs: Iterable, use_serialize=True, chunk_size=100000, skip_chunk=0):
        skipped_chunk = 0
        for chunked_iter in split_every(kvs, chunk_size=chunk_size):
            if skipped_chunk < skip_chunk:
                skipped_chunk += 1
            else:
                self.kv_stub.putAll(self.__generate_operand(chunked_iter, use_serialize=use_serialize), metadata=_get_meta(_table))

    def delete(self, _table, k, use_serialize=True):
        k = self.kv_to_bytes(k=k, use_serialize=use_serialize)
        operand = self.kv_stub.delOne(kv_pb2.Operand(key=k), metadata=_get_meta(_table))
        return self._deserialize_operand(operand, use_serialize=use_serialize)

    def get(self, _table, k, use_serialize=True):
        k = self.kv_to_bytes(k=k, use_serialize=use_serialize)
        operand = self.kv_stub.get(kv_pb2.Operand(key=k), metadata=_get_meta(_table))
        return self._deserialize_operand(operand, use_serialize=use_serialize)

    def iterate(self, _table, _range):
        return self.kv_stub.iterate(_range, metadata=_get_meta(_table))

    def destroy(self, _table):
        self.kv_stub.destroy(empty, metadata=_get_meta(_table))

    def destroy_all(self, _table):
        self.kv_stub.destroyAll(empty, metadata=_get_meta(_table))

    def count(self, _table):
        return self.kv_stub.count(empty, metadata=_get_meta(_table)).value

    '''
    Computing apis
    '''

    def map(self, _table: _DTable, func):
        func_id, func_bytes = self.serialize_and_hash_func(func)
        operand = storage_basic_pb2.StorageLocator(namespace=_table._namespace, type=_table._type, name=_table._name)
        unary_p = processor_pb2.UnaryProcess(operand=operand,
                                             info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                         function_id=func_id,
                                                                         function_bytes=func_bytes))
        resp = self.proc_stub.map(unary_p)

        return self._create_table_from_locator(resp, _table._partitions)

    def map_values(self, _table: _DTable, func):
        func_id, func_bytes = self.serialize_and_hash_func(func)
        operand = storage_basic_pb2.StorageLocator(namespace=_table._namespace, type=_table._type, name=_table._name)
        unary_p = processor_pb2.UnaryProcess(operand=operand,
                                             info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                         function_id=func_id,
                                                                         function_bytes=func_bytes))
        resp = self.proc_stub.mapValues(unary_p)
        return self._create_table_from_locator(resp, _table._partitions)

    def map_partitions(self, _table: _DTable, func):
        func_id, func_bytes = self.serialize_and_hash_func(func)
        operand = storage_basic_pb2.StorageLocator(namespace=_table._namespace, type=_table._type, name=_table._name)
        unary_p = processor_pb2.UnaryProcess(operand=operand,
                                             info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                         function_id=func_id,
                                                                         function_bytes=func_bytes))
        resp = self.proc_stub.mapPartitions(unary_p)
        return self._create_table_from_locator(resp, _table._partitions)

    def reduce(self, _table: _DTable, func):
        func_id, func_bytes = self.serialize_and_hash_func(func)
        operand = storage_basic_pb2.StorageLocator(namespace=_table._namespace, type=_table._type, name=_table._name)
        unary_p = processor_pb2.UnaryProcess(operand=operand,
                                             info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                         function_id=func_id,
                                                                         function_bytes=func_bytes))
        values = [_EggRoll._deserialize_operand(operand) for operand in self.proc_stub.reduce(unary_p)]
        values = [v for v in filter(partial(is_not, None), values)]
        if len(values) <= 0:
            return None
        if len(values) == 1:
            return values[0]
        else:
            val, *remain = values
            for _nv in remain:
                val = func(val, _nv)
        return val

    def join(self, _left: _DTable, _right: _DTable, func):
        func_id, func_bytes = self.serialize_and_hash_func(func)
        l_op = storage_basic_pb2.StorageLocator(namespace=_left._namespace, type=_left._type, name=_left._name)
        r_op = storage_basic_pb2.StorageLocator(namespace=_right._namespace, type=_right._type, name=_right._name)
        binary_p = processor_pb2.BinaryProcess(left=l_op, right=r_op, info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                                                  function_id=func_id,
                                                                                                  function_bytes=func_bytes))
        resp = self.proc_stub.join(binary_p)
        return self._create_table_from_locator(resp, _left._partitions)

    def glom(self, _table: _DTable):
        func_id = str(uuid.uuid1())
        operand = storage_basic_pb2.StorageLocator(namespace=_table._namespace, type=_table._type, name=_table._name)

        unary_p = processor_pb2.UnaryProcess(operand=operand, info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                                          function_id=func_id))
        resp = self.proc_stub.glom(unary_p)
        return self._create_table_from_locator(resp, _table._partitions)

    def sample(self, _table: _DTable, fraction, seed):
        if fraction < 0 or fraction > 1:
            raise ValueError("fraction must be in [0, 1]")
        func_bytes = self.value_serdes.serialize((fraction, seed))
        func_id = str(uuid.uuid1())
        operand = storage_basic_pb2.StorageLocator(namespace=_table._namespace, type=_table._type, name=_table._name)

        unary_p = processor_pb2.UnaryProcess(operand=operand, info=processor_pb2.TaskInfo(task_id=self.job_id,
                                                                                          function_id=func_id,
                                                                                          function_bytes=func_bytes))
        resp = self.proc_stub.sample(unary_p)
        return self._create_table_from_locator(resp, _table._partitions)


class _EggRollIterator(object):

    def __init__(self, _table, start=None, end=None, use_serialize=True):
        self._table = _table
        self._start = start
        self._end = end
        self._cache = None
        self._index = 0
        self._next_item = None
        self._use_serialize = use_serialize

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __iter__(self):
        return self

    def __refresh_cache(self):
        if self._next_item is None:
            self._cache = list(
                _EggRoll.get_instance().iterate(self._table, kv_pb2.Range(start=self._start, end=self._end)))
        else:
            self._cache = list(
                _EggRoll.get_instance().iterate(self._table, kv_pb2.Range(start=self._next_item.key, end=self._end)))
        if len(self._cache) == 0:
            raise StopIteration
        self._index = 0

    def __next__(self):
        if self._cache is None or self._index >= len(self._cache):
            self.__refresh_cache()
        self._next_item = self._cache[self._index]
        self._index += 1
        return _EggRoll._deserialize_operand(self._next_item, include_key=True, use_serialize=self._use_serialize)
