#
#  Copyright 2019 The Eggroll Authors. All Rights Reserved.
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
import asyncio
import hashlib
import itertools
import pickle as c_pickle
import shutil
import typing
import uuid
from collections import Iterable
from concurrent.futures import ProcessPoolExecutor as Executor
from functools import partial
from heapq import heapify, heappop, heapreplace
from operator import is_not
from pathlib import Path

import lmdb
import numpy as np
from cachetools import LRUCache
from cachetools import cached

from fate_arch._interface import GC
from fate_arch.common import file_utils
from fate_arch.common.log import getLogger
from fate_arch.session._session_types import _FederationParties, Party
from fate_arch.session.impl.standalone import _cloudpickle as f_pickle
from fate_arch.session._interface import TableABC, SessionABC, FederationEngineABC

LOGGER = getLogger()


class StandaloneSession(SessionABC):

    def __init__(self, session_id):
        self._session_id = session_id
        self._pool = Executor()
        _set_session(self)

        self._federation_session: typing.Optional[StandaloneFederation] = None
        self._federation_parties: typing.Optional[_FederationParties] = None

    def _init_federation(self, federation_session_id: str,
                         party: Party,
                         parties: typing.MutableMapping[str, typing.List[Party]]):
        if self._federation_session is not None:
            raise RuntimeError("federation session already initialized")
        self._federation_session = StandaloneFederation(federation_session_id, party)
        self._federation_parties = _FederationParties(party, parties)

    def init_federation(self, federation_session_id: str, runtime_conf: dict, **kwargs):
        party, parties = self._parse_runtime_conf(runtime_conf)
        self._init_federation(federation_session_id, party, parties)

    def load(self, name, namespace, **kwargs) -> TableABC:
        return _load_table(name, namespace)

    def parallelize(self, data: Iterable, partition: int, include_key: bool = False, **kwargs):
        if not include_key:
            data = enumerate(data)
        table = _create_table(name=str(uuid.uuid1()), namespace=self._session_id, partitions=partition)
        # noinspection PyProtectedMember
        table._put_all(data)
        return table

    def cleanup(self, name, namespace):
        data_path = _get_data_dir()
        if not data_path.is_dir():
            raise EnvironmentError(f"illegal data dir: {data_path}")

        namespace_dir = data_path.joinpath(namespace)
        if not namespace_dir.is_dir():
            raise EnvironmentError(f"namespace dir {namespace_dir} does not exist")

        for table in namespace_dir.glob(name):
            shutil.rmtree(table)

    def stop(self):
        _set_session(None)

    def kill(self):
        _set_session(None)

    def _get_federation(self):
        return self._federation_session

    def _get_session_id(self):
        return self._session_id

    def _get_federation_parties(self):
        raise self._federation_parties

    def _submit_unary(self, func, _do_func, partitions, name, namespace):
        task_info = _TaskInfo(self._session_id,
                              function_id=str(uuid.uuid1()),
                              function_bytes=f_pickle.dumps(func))
        futures = []
        for p in range(partitions):
            futures.append(self._pool.submit(_do_func, _UnaryProcess(task_info, _Operand(namespace, name, p))))
        results = [r.result() for r in futures]
        return results

    def _submit_binary(self, func, do_func, partitions, name, namespace, other_name, other_namespace):
        task_info = _TaskInfo(self._session_id,
                              function_id=str(uuid.uuid1()),
                              function_bytes=f_pickle.dumps(func))
        futures = []
        for p in range(partitions):
            left = _Operand(namespace, name, p)
            right = _Operand(other_namespace, other_name, p)
            futures.append(self._pool.submit(do_func, _BinaryProcess(task_info, left, right)))
        results = [r.result() for r in futures]
        return results


class Table(TableABC):

    def __init__(self, namespace, name, partitions=1, need_cleanup=True):
        self._need_cleanup = need_cleanup
        self._namespace = namespace
        self._name = name
        self._partitions = partitions

    def __del__(self):
        if self._need_cleanup:
            self._destroy()

    def __str__(self):
        return f"need_cleanup: {self._need_cleanup}, " \
               f"namespace: {self._namespace}," \
               f"name: {self._name}," \
               f"partitions: {self._partitions}"

    def _destroy(self):
        for p in range(self._partitions):
            env = self._get_env_for_partition(p, write=True)
            db = env.open_db()
            with env.begin(write=True) as txn:
                txn.drop(db)

        table_key = f"{self._namespace}.{self._name}"
        _get_meta_table()._delete(table_key)
        path = _get_storage_dir(self._namespace, self._name)
        shutil.rmtree(path, ignore_errors=True)

    def save(self, name, namespace, **kwargs):
        return self._save_as(name, namespace, need_cleanup=False)

    def count(self):
        cnt = 0
        for p in range(self._partitions):
            env = self._get_env_for_partition(p)
            cnt += env.stat()['entries']
        return cnt

    def collect(self, **kwargs):
        iterators = []
        for p in range(self._partitions):
            env = self._get_env_for_partition(p)
            txn = env.begin()
            iterators.append(txn.cursor())

        # Merge sorted
        entries = []
        for _id, it in enumerate(iterators):
            if it.next():
                key, value = it.item()
                entries.append([key, value, _id, it])
            else:
                it.close()
        heapify(entries)
        while entries:
            key, value, _, it = entry = entries[0]
            yield c_pickle.loads(key), c_pickle.loads(value)
            if it.next():
                entry[0], entry[1] = it.item()
                heapreplace(entries, entry)
            else:
                _, _, _, it = heappop(entries)
                it.close()

    def take(self, n=1, **kwargs):
        if n <= 0:
            raise ValueError(f"{n} <= 0")
        return list(itertools.islice(self.collect(), n))

    def first(self, **kwargs):
        resp = self.take(1, **kwargs)
        if len(resp) < 1:
            raise RuntimeError(f"table is empty")
        return resp[0]

    def reduce(self, func):
        # noinspection PyProtectedMember
        rs = get_session()._submit_unary(func, _do_reduce, self._partitions, self._name, self._namespace)
        rs = [r for r in filter(partial(is_not, None), rs)]
        if len(rs) <= 0:
            return None
        rtn = rs[0]
        for r in rs[1:]:
            rtn = func(rtn, r)
        return rtn

    def map(self, func):
        return self._unary(func, _do_map)

    def mapValues(self, func):
        return self._unary(func, _do_map_values)

    def flatMap(self, func):
        return self._unary(func, _do_flat_map)

    def mapPartitions(self, func):
        return self._unary(func, _do_map_partitions)

    def glom(self):
        return self._unary(None, _do_glom)

    def sample(self, fraction, seed=None):
        return self._unary((fraction, seed), _do_sample)

    def filter(self, func):
        return self._unary(func, _do_filter)

    def join(self, other: 'Table', func):
        return self._binary(other, func, _do_join)

    def subtractByKey(self, other: 'Table'):
        func = f"{self._namespace}.{self._name}-{other._namespace}.{other._name}"
        return self._binary(other, func, _do_subtract_by_key)

    def union(self, other: 'Table', func=lambda v1, v2: v1):
        return self._binary(other, func, _do_union)

    def _unary(self, func, do_func):
        session = get_session()
        # noinspection PyProtectedMember
        results = session._submit_unary(func, do_func, self._partitions, self._name, self._namespace)
        result = results[0]
        # noinspection PyProtectedMember
        return _create_table(result.name, result.namespace, self._partitions)

    def _binary(self, other: 'Table', func, do_func):
        session = get_session()
        session_id = session.session_id
        left, right = self, other
        if left._partitions != right._partitions:
            if other.count() > self.count():
                left = left._save_as(str(uuid.uuid1()), session_id, partition=right._partitions)
            else:
                right = other._save_as(str(uuid.uuid1()), session_id, partition=left._partitions)

        # noinspection PyProtectedMember
        results = session._submit_binary(func, do_func,
                                         left._partitions, left._name, left._namespace, right._name, right._namespace)
        result: _Operand = results[0]
        # noinspection PyProtectedMember
        return _create_table(result.name, result.namespace, left._partitions)

    def _save_as(self, name, namespace, partition=None, need_cleanup=True):
        if partition is None:
            partition = self._partitions
        # noinspection PyProtectedMember
        dup = _create_table(name, namespace, partition, need_cleanup)
        dup._put_all(self.collect())
        return dup

    def _get_env_for_partition(self, p: int, write=False):
        return _get_env(self._namespace, self._name, str(p), write=write)

    def _put(self, k, v):
        k_bytes, v_bytes = _kv_to_bytes(k=k, v=v)
        p = _hash_key_to_partition(k_bytes, self._partitions)
        env = self._get_env_for_partition(p, write=True)
        with env.begin(write=True) as txn:
            return txn.put(k_bytes, v_bytes)

    def _put_all(self, kv_list: Iterable):
        txn_map = {}
        is_success = True
        for p in range(self._partitions):
            env = self._get_env_for_partition(p, write=True)
            txn = env.begin(write=True)
            txn_map[p] = env, txn
        for k, v in kv_list:
            try:
                k_bytes, v_bytes = _kv_to_bytes(k=k, v=v)
                p = _hash_key_to_partition(k_bytes, self._partitions)
                is_success = is_success and txn_map[p][1].put(k_bytes, v_bytes)
            except Exception as e:
                is_success = False
                LOGGER.exception(f"put_all for k={k} v={v} fail. exception: {e}")
                break
        for p, (env, txn) in txn_map.items():
            txn.commit() if is_success else txn.abort()

    def _get(self, k):
        k_bytes = _k_to_bytes(k=k)
        p = _hash_key_to_partition(k_bytes, self._partitions)
        env = self._get_env_for_partition(p)
        with env.begin(write=True) as txn:
            old_value_bytes = txn.get(k_bytes)
            return None if old_value_bytes is None else c_pickle.loads(old_value_bytes)

    def _delete(self, k):
        k_bytes = _k_to_bytes(k=k)
        p = _hash_key_to_partition(k_bytes, self._partitions)
        env = self._get_env_for_partition(p, write=True)
        with env.begin(write=True) as txn:
            old_value_bytes = txn.get(k_bytes)
            if txn.delete(k_bytes):
                return None if old_value_bytes is None else c_pickle.loads(old_value_bytes)
            return None


class StandaloneFederation(FederationEngineABC):

    def _federation_object_key(self, name, tag, s_party, d_party):
        return f"{self._session_id}-{name}-{tag}-{s_party.role}-{s_party.party_id}-{d_party.role}-{d_party.party_id}"

    def __init__(self, session_id, party: Party):
        self._session_id = session_id
        self._party: Party = party
        self._loop = asyncio.get_event_loop()

        self._federation_status_table = \
            _create_table("__federation_status__", self._session_id, 10, need_cleanup=False, error_if_exist=False)
        self._federation_object_table = \
            _create_table("__federation_object__", self._session_id, 10, need_cleanup=False, error_if_exist=False)

    # noinspection PyProtectedMember
    def _put_status(self, _tagged_key, value):
        self._federation_status_table._put(_tagged_key, value)

    # noinspection PyProtectedMember
    def _put_object(self, _tagged_key, value):
        self._federation_object_table._put(_tagged_key, value)

    # noinspection PyProtectedMember
    def _get_object(self, _tagged_key):
        return self._federation_object_table._get(_tagged_key)

    # noinspection PyProtectedMember
    def _get_status(self, _tagged_key):
        return self._federation_status_table._get(_tagged_key)

    def remote(self, v, name: str, tag: str, parties: typing.List[Party], gc: GC):
        log_str = f"federation.remote(name={name}, tag={tag}, parties={parties})"

        assert v is not None, \
            f"[{log_str}]remote `None`"
        LOGGER.debug(f"[{log_str}]remote data, type={type(v)}")

        if isinstance(v, Table):
            # noinspection PyProtectedMember
            v = v.save(name=str(uuid.uuid1()), namespace=v._namespace)

        for party in parties:
            _tagged_key = self._federation_object_key(name, tag, self._party, party)
            if isinstance(v, Table):
                # noinspection PyProtectedMember
                self._put_status(_tagged_key, (v._name, v._namespace))
            else:
                self._put_object(_tagged_key, v)
                self._put_status(_tagged_key, _tagged_key)
            LOGGER.debug("[REMOTE] Sent {}".format(_tagged_key))

    # noinspection PyProtectedMember
    def get(self, name: str, tag: str, parties: typing.List[Party], gc: GC) -> typing.List:
        log_str = f"federation.get(name={name}, tag={tag}, party={parties})"
        LOGGER.debug(f"[{log_str}]")
        tasks = []

        for party in parties:
            _tagged_key = self._federation_object_key(name, tag, party, self._party)
            tasks.append(_check_status_and_get_value(self._get_status, _tagged_key))
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        rtn = []
        for r in results:
            LOGGER.debug(f"[GET] {self._party} getting {r} from {parties}")

            if isinstance(r, tuple):
                # noinspection PyTypeChecker
                table: Table = get_session().load(name=r[0], namespace=r[1])
                rtn.append(table)
                gc.add_gc_func(tag, table._destroy)
            else:
                obj = self._get_object(r)
                if obj is None:
                    raise EnvironmentError(f"federation get None from {parties} with name {name}, tag {tag}")
                rtn.append(obj)
                gc.add_gc_func(tag, lambda: self._federation_object_table._delete(r))
        return rtn


__SESSION: typing.Optional['StandaloneSession'] = None


def _set_session(session):
    global __SESSION
    __SESSION = session


def get_session():
    global __SESSION
    return __SESSION


_meta_table: typing.Optional[Table] = None


def _get_meta_table():
    global _meta_table
    if _meta_table is None:
        _meta_table = Table(namespace='__META__', name='fragments', partitions=10, need_cleanup=False)
    return _meta_table


# noinspection PyProtectedMember
def _get_from_meta_table(key):
    return _get_meta_table()._get(key)


# noinspection PyProtectedMember
def _put_to_meta_table(key, value):
    _get_meta_table()._put(key, value)


_data_dir = Path(file_utils.get_project_base_directory()).joinpath('data').absolute()


def _get_data_dir():
    return _data_dir


def _get_storage_dir(*args):
    return _data_dir.joinpath(*args)


async def _check_status_and_get_value(get_func, key):
    value = get_func(key)
    while value is None:
        await asyncio.sleep(0.1)
        value = get_func(key)
    LOGGER.debug("[GET] Got {} type {}".format(key, 'Table' if isinstance(value, tuple) else 'Object'))
    return value


def _create_table(name, namespace, partitions, need_cleanup=True, error_if_exist=False):
    _table_key = ".".join([namespace, name])
    if _get_from_meta_table(_table_key) is not None:
        if error_if_exist:
            raise RuntimeError(f"table already exist: name={name}, namespace={namespace}")
        else:
            partitions = _get_from_meta_table(_table_key)
    else:
        _put_to_meta_table(_table_key, partitions)

    return Table(namespace, name, partitions, need_cleanup=need_cleanup)


def _load_table(name, namespace):
    _table_key = ".".join([namespace, name])
    partitions = _get_from_meta_table(_table_key)
    if partitions is None:
        raise RuntimeError(f"table not exist: name={name}, namespace={namespace}")
    return Table(namespace, name, partitions, need_cleanup=False)


class _TaskInfo:
    def __init__(self, task_id, function_id, function_bytes):
        self.task_id = task_id
        self.function_id = function_id
        self.function_bytes = function_bytes

    def get_func(self):
        return f_pickle.loads(self.function_bytes)

    def output_operand(self, partitions):
        return _Operand(self.task_id, self.function_id, partitions)


class _Operand:
    def __init__(self, namespace, name, partition):
        self.namespace = namespace
        self.name = name
        self.partition = partition

    def as_env(self, write=False):
        return _get_env(self.namespace, self.name, str(self.partition), write=write)


class _UnaryProcess:
    def __init__(self, task_info: _TaskInfo, operand: _Operand):
        self.info = task_info
        self.operand = operand

    def output_operand(self):
        return _Operand(self.info.task_id, self.info.function_id, self.operand.partition)

    def get_func(self):
        return self.info.get_func()


class _BinaryProcess:
    def __init__(self, task_info: _TaskInfo, left: _Operand, right: _Operand):
        self.info = task_info
        self.left = left
        self.right = right

    def output_operand(self):
        return _Operand(self.info.task_id, self.info.function_id, self.left.partition)

    def get_func(self):
        return self.info.get_func()


class _EvictLRUCache(LRUCache):

    def __init__(self, maxsize):
        LRUCache.__init__(self, maxsize, None)

    def popitem(self):
        key, val = LRUCache.popitem(self)
        val.close()
        return key, val


def _get_env(*args, write=False):
    _path = _get_storage_dir(*args)
    return _open_env(_path, write=write)


@cached(cache=_EvictLRUCache(maxsize=64))
def _open_env(path, write=False):
    path.mkdir(parents=True, exist_ok=True)
    return lmdb.open(path.as_posix(), create=True, max_dbs=1, max_readers=1024, lock=write, sync=True,
                     map_size=10_737_418_240)


def _hash_key_to_partition(key, partitions):
    _key = hashlib.sha1(key).digest()
    if isinstance(_key, bytes):
        _key = int.from_bytes(_key, byteorder='little', signed=False)
    if partitions < 1:
        raise ValueError('partitions must be a positive number')
    b, j = -1, 0
    while j < partitions:
        b = int(j)
        _key = ((_key * 2862933555777941757) + 1) & 0xffffffffffffffff
        j = float(b + 1) * (float(1 << 31) / float((_key >> 33) + 1))
    return int(b)


def _do_map(p: _UnaryProcess):
    _mapper = p.get_func()
    op = p.operand
    rtn = p.output_operand()
    source_env = p.operand.as_env()
    serialize = c_pickle.dumps
    deserialize = c_pickle.loads
    _table_key = ".".join([op.namespace, op.name])
    txn_map = {}
    partitions = _get_from_meta_table(_table_key)
    for p in range(partitions):
        env = _get_env(rtn.namespace, rtn.name, str(p), write=True)
        txn = env.begin(write=True)
        txn_map[p] = txn
    with source_env.begin() as source_txn:
        cursor = source_txn.cursor()
        for k_bytes, v_bytes in cursor:
            k, v = deserialize(k_bytes), deserialize(v_bytes)
            k1, v1 = _mapper(k, v)
            k1_bytes, v1_bytes = serialize(k1), serialize(v1)
            p = _hash_key_to_partition(k1_bytes, partitions)
            dest_txn = txn_map[p]
            dest_txn.put(k1_bytes, v1_bytes)
        cursor.close()
    for p, txn in txn_map.items():
        txn.commit()
    return rtn


def _generator_from_cursor(cursor):
    deserialize = c_pickle.loads
    for k, v in cursor:
        yield deserialize(k), deserialize(v)


def _do_map_partitions(p: _UnaryProcess):
    _mapper = p.get_func()
    rtn = p.output_operand()
    source_env = p.operand.as_env()
    dst_env = rtn.as_env(write=True)
    serialize = c_pickle.dumps
    with source_env.begin() as source_txn:
        with dst_env.begin(write=True) as dst_txn:
            cursor = source_txn.cursor()
            v = _mapper(_generator_from_cursor(cursor))
            if cursor.last():
                k_bytes = cursor.key()
                dst_txn.put(k_bytes, serialize(v))
            cursor.close()
    return rtn


def _do_map_partitions2(p: _UnaryProcess):
    _mapper = p.get_func()
    rtn = p.output_operand()
    source_env = p.operand.as_env()
    dst_env = rtn.as_env(write=True)
    serialize = c_pickle.dumps
    with source_env.begin() as source_txn:
        with dst_env.begin(write=True) as dst_txn:
            cursor = source_txn.cursor()
            v = _mapper(_generator_from_cursor(cursor))
            if cursor.last():
                if isinstance(v, Iterable):
                    for k1, v1 in v:
                        dst_txn.put(serialize(k1), serialize(v1))
                else:
                    k_bytes = cursor.key()
                    dst_txn.put(k_bytes, serialize(v))
            cursor.close()
    return rtn


def _do_map_values(p: _UnaryProcess):
    _mapper = p.get_func()
    rtn = p.output_operand()
    source_env = p.operand.as_env()
    dst_env = rtn.as_env(write=True)
    serialize = c_pickle.dumps
    deserialize = c_pickle.loads
    with source_env.begin() as source_txn:
        with dst_env.begin(write=True) as dst_txn:
            cursor = source_txn.cursor()
            for k_bytes, v_bytes in cursor:
                v = deserialize(v_bytes)
                v1 = _mapper(v)
                dst_txn.put(k_bytes, serialize(v1))
            cursor.close()
    return rtn


def _do_flat_map(p: _UnaryProcess):
    _func = p.get_func()
    rtn = p.output_operand()
    source_env = p.operand.as_env()
    dst_env = rtn.as_env(write=True)
    serialize = c_pickle.dumps
    deserialize = c_pickle.loads
    with source_env.begin() as source_txn:
        with dst_env.begin(write=True) as dst_txn:
            cursor = source_txn.cursor()
            for k_bytes, v_bytes in cursor:
                k = deserialize(k_bytes)
                v = deserialize(v_bytes)
                map_result = _func(k, v)
                for result_k, result_v in map_result:
                    dst_txn.put(serialize(result_k), serialize(result_v))
            cursor.close()
    return rtn


def _do_reduce(p: _UnaryProcess):
    _reducer = p.get_func()
    source_env = p.operand.as_env()
    deserialize = c_pickle.loads
    value = None
    with source_env.begin() as source_txn:
        cursor = source_txn.cursor()
        for k_bytes, v_bytes in cursor:
            v = deserialize(v_bytes)
            if value is None:
                value = v
            else:
                value = _reducer(value, v)
    return value


def _do_glom(p: _UnaryProcess):
    rtn = p.output_operand()
    source_env = p.operand.as_env()
    dst_env = rtn.as_env(write=True)
    serialize = c_pickle.dumps
    deserialize = c_pickle.loads
    with source_env.begin() as source_txn:
        with dst_env.begin(write=True) as dest_txn:
            cursor = source_txn.cursor()
            v_list = []
            k_bytes = None
            for k, v in cursor:
                v_list.append((deserialize(k), deserialize(v)))
                k_bytes = k
            if k_bytes is not None:
                dest_txn.put(k_bytes, serialize(v_list))
    return rtn


def _do_sample(p: _UnaryProcess):
    rtn = p.output_operand()
    source_env = p.operand.as_env()
    dst_env = rtn.as_env(write=True)
    deserialize = c_pickle.loads
    fraction, seed = deserialize(p.info.function_bytes)
    with source_env.begin() as source_txn:
        with dst_env.begin(write=True) as dst_txn:
            cursor = source_txn.cursor()
            cursor.first()
            random_state = np.random.RandomState(seed)
            for k, v in cursor:
                # noinspection PyArgumentList
                if random_state.rand() < fraction:
                    dst_txn.put(k, v)
    return rtn


def _do_filter(p: _UnaryProcess):
    _func = p.get_func()
    rtn = p.output_operand()
    source_env = p.operand.as_env()
    dst_env = rtn.as_env(write=True)
    with source_env.begin() as source_txn:
        with dst_env.begin(write=True) as dst_txn:
            cursor = source_txn.cursor()
            for k_bytes, v_bytes in cursor:
                k = c_pickle.loads(k_bytes)
                v = c_pickle.loads(v_bytes)
                if _func(k, v):
                    dst_txn.put(k_bytes, v_bytes)
            cursor.close()
    return rtn


def _do_subtract_by_key(p: _BinaryProcess):
    left_op = p.left
    right_op = p.right
    rtn = p.output_operand()
    right_env = right_op.as_env()
    left_env = left_op.as_env()
    dst_env = rtn.as_env(write=True)
    with left_env.begin() as left_txn:
        with right_env.begin() as right_txn:
            with dst_env.begin(write=True) as dst_txn:
                cursor = left_txn.cursor()
                for k_bytes, left_v_bytes in cursor:
                    right_v_bytes = right_txn.get(k_bytes)
                    if right_v_bytes is None:
                        dst_txn.put(k_bytes, left_v_bytes)
                cursor.close()
    return rtn


def _do_join(p: _BinaryProcess):
    _joiner = p.get_func()
    left_op = p.left
    right_op = p.right
    rtn = p.output_operand()
    right_env = right_op.as_env()
    left_env = left_op.as_env()
    dst_env = rtn.as_env(write=True)
    serialize = c_pickle.dumps
    deserialize = c_pickle.loads
    with left_env.begin() as left_txn:
        with right_env.begin() as right_txn:
            with dst_env.begin(write=True) as dst_txn:
                cursor = left_txn.cursor()
                for k_bytes, v1_bytes in cursor:
                    v2_bytes = right_txn.get(k_bytes)
                    if v2_bytes is None:
                        continue
                    v1 = deserialize(v1_bytes)
                    v2 = deserialize(v2_bytes)
                    v3 = _joiner(v1, v2)
                    dst_txn.put(k_bytes, serialize(v3))
    return rtn


def _do_union(p: _BinaryProcess):
    _func = p.get_func()
    left_op = p.left
    right_op = p.right
    rtn = p.output_operand()
    right_env = right_op.as_env()
    left_env = left_op.as_env()
    dst_env = rtn.as_env(write=True)
    serialize = c_pickle.dumps
    deserialize = c_pickle.loads
    with left_env.begin() as left_txn:
        with right_env.begin() as right_txn:
            with dst_env.begin(write=True) as dst_txn:
                # process left op
                left_cursor = left_txn.cursor()
                for k_bytes, left_v_bytes in left_cursor:
                    right_v_bytes = right_txn.get(k_bytes)
                    if right_v_bytes is None:
                        dst_txn.put(k_bytes, left_v_bytes)
                    else:
                        left_v = deserialize(left_v_bytes)
                        right_v = deserialize(right_v_bytes)
                        final_v = _func(left_v, right_v)
                        dst_txn.put(k_bytes, serialize(final_v))
                left_cursor.close()

                # process right op
                right_cursor = right_txn.cursor()
                for k_bytes, right_v_bytes in right_cursor:
                    final_v_bytes = dst_txn.get(k_bytes)
                    if final_v_bytes is None:
                        dst_txn.put(k_bytes, right_v_bytes)
                right_cursor.close()
    return rtn


def _kv_to_bytes(k, v):
    return c_pickle.dumps(k), c_pickle.dumps(v)


def _k_to_bytes(k):
    return c_pickle.dumps(k)
