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
import itertools
import logging
import logging.config
import os
import pickle as c_pickle
import shutil
import signal
import threading
import time
import uuid
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor as Executor
from contextlib import ExitStack
from functools import partial
from heapq import heapify, heappop, heapreplace
from operator import is_not
from pathlib import Path
from typing import List, Tuple

import cloudpickle as f_pickle
import lmdb
import numpy as np
from fate.arch.abc import PartyMeta

from .federation import FederationDataType

LOGGER = logging.getLogger(__name__)

serialize = c_pickle.dumps
deserialize = c_pickle.loads

# default message max size in bytes = 1MB
DEFAULT_MESSAGE_MAX_SIZE = 1048576

if (STANDALONE_DATA_PATH := os.getenv("STANDALONE_DATA_PATH")) is not None:
    _data_dir = Path(STANDALONE_DATA_PATH)
    LOGGER.debug(f"env STANDALONE_DATA_PATH is set to {STANDALONE_DATA_PATH}, using {_data_dir} as data dir")
else:
    _data_dir = Path(
        os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir, "__standalone_data__"
            )
        )
    )
    LOGGER.debug(f"env STANDALONE_DATA_PATH is not set, using {_data_dir} as data dir")


def _watch_thread_react_to_parent_die(ppid, logger_config):
    """
    this function is call when a process is created, and it will watch parent process and initialize loggers
    Args:
        ppid: parent process id
    """

    # watch parent process, if parent process is dead, then kill self
    # the trick is to use os.kill(ppid, 0) to check if parent process is alive periodically
    # and if parent process is dead, then kill self
    #
    # Note: this trick is modified from the answer by aaron: https://stackoverflow.com/a/71369760/14697733
    pid = os.getpid()

    def f():
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()

    # initialize loggers
    if logger_config is not None:
        logging.config.dictConfig(logger_config)
    # else:
    #     level = os.getenv("DEBUG_MODE_LOG_LEVEL", "DEBUG")
    #     try:
    #         import rich.logging
    #
    #         logging_class = "rich.logging.RichHandler"
    #         logging_formatters = {}
    #         handlers = {
    #             "console": {
    #                 "class": logging_class,
    #                 "level": level,
    #                 "filters": [],
    #             }
    #         }
    #     except ImportError:
    #         logging_class = "logging.StreamHandler"
    #         logging_formatters = {
    #             "console": {
    #                 "format": "[%(levelname)s][%(asctime)-8s][%(process)s][%(module)s.%(funcName)s][line:%(lineno)d]: %(message)s"
    #             }
    #         }
    #         handlers = {
    #             "console": {
    #                 "class": logging_class,
    #                 "level": level,
    #                 "formatter": "console",
    #             }
    #         }
    #     logging.config.dictConfig(dict(
    #         version=1,
    #         formatters=logging_formatters,
    #         handlers=handlers,
    #         filters={},
    #         loggers={},
    #         root=dict(handlers=["console"], level="DEBUG"),
    #         disable_existing_loggers=False,
    #     ))


# noinspection PyPep8Naming
class Table(object):
    def __init__(
        self,
        session: "Session",
        namespace: str,
        name: str,
        partitions,
        need_cleanup=True,
    ):
        self._need_cleanup = need_cleanup
        self._namespace = namespace
        self._name = name
        self._partitions = partitions
        self._session = session

    @property
    def partitions(self):
        return self._partitions

    @property
    def name(self):
        return self._name

    @property
    def namespace(self):
        return self._namespace

    def __del__(self):
        if self._need_cleanup:
            try:
                self.destroy()
            except:
                pass

    def __str__(self):
        return f"<Table {self._namespace}|{self._name}|{self._partitions}|{self._need_cleanup}>"

    def __repr__(self):
        return self.__str__()

    def destroy(self):
        for p in range(self._partitions):
            with self._get_env_for_partition(p, write=True) as env:
                db = env.open_db()
                with env.begin(write=True) as txn:
                    txn.drop(db)
        _TableMetaManager.destroy_table(self._namespace, self._name)

    def take(self, n, **kwargs):
        if n <= 0:
            raise ValueError(f"{n} <= 0")
        return list(itertools.islice(self.collect(**kwargs), n))

    def count(self):
        cnt = 0
        for p in range(self._partitions):
            with self._get_env_for_partition(p) as env:
                cnt += env.stat()["entries"]
        return cnt

    # noinspection PyUnusedLocal
    def collect(self, **kwargs):
        iterators = []
        with ExitStack() as s:
            for p in range(self._partitions):
                env = s.enter_context(self._get_env_for_partition(p))
                txn = s.enter_context(env.begin())
                iterators.append(s.enter_context(txn.cursor()))

            # Merge sorted
            entries = []
            for _id, it in enumerate(iterators):
                if it.next():
                    key, value = it.item()
                    entries.append([key, value, _id, it])
            heapify(entries)
            while entries:
                key, value, _, it = entry = entries[0]
                yield deserialize(key), deserialize(value)
                if it.next():
                    entry[0], entry[1] = it.item()
                    heapreplace(entries, entry)
                else:
                    _, _, _, it = heappop(entries)

    def reduce(self, func):
        # noinspection PyProtectedMember
        rs = self._session._submit_unary(func, _do_reduce, self._partitions, self._name, self._namespace)
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
        _flat_mapped = self._unary(func, _do_flat_map)
        return _flat_mapped.save_as(
            name=str(uuid.uuid1()),
            namespace=_flat_mapped.namespace,
            partition=self._partitions,
            need_cleanup=True,
        )

    def applyPartitions(self, func):
        return self._unary(func, _do_apply_partitions)

    def mapPartitions(self, func, preserves_partitioning=False):
        un_shuffled = self._unary(func, _do_map_partitions)
        if preserves_partitioning:
            return un_shuffled
        return un_shuffled.save_as(
            name=str(uuid.uuid1()),
            namespace=un_shuffled.namespace,
            partition=self._partitions,
            need_cleanup=True,
        )

    def mapPartitionsWithIndex(self, func, preserves_partitioning=False):
        un_shuffled = self._unary(func, _do_map_partitions_with_index)
        if preserves_partitioning:
            return un_shuffled
        return un_shuffled.save_as(
            name=str(uuid.uuid1()),
            namespace=un_shuffled.namespace,
            partition=self._partitions,
            need_cleanup=True,
        )

    def mapReducePartitions(self, mapper, reducer):
        dup = _create_table(
            self._session,
            str(uuid.uuid1()),
            self.namespace,
            self._partitions,
            need_cleanup=True,
        )

        def _dict_reduce(a: dict, b: dict):
            for k, v in b.items():
                if k not in a:
                    a[k] = v
                else:
                    a[k] = reducer(a[k], v)
            return a

        def _local_map_reduce(it):
            ret = {}
            for _k, _v in mapper(it):
                if _k not in ret:
                    ret[_k] = _v
                else:
                    ret[_k] = reducer(ret[_k], _v)
            return ret

        dup.put_all(self.applyPartitions(_local_map_reduce).reduce(_dict_reduce).items())
        return dup

    def glom(self):
        return self._unary(None, _do_glom)

    def sample(self, fraction, seed=None):
        return self._unary((fraction, seed), _do_sample)

    def filter(self, func):
        return self._unary(func, _do_filter)

    def join(self, other: "Table", func):
        return self._binary(other, func, _do_join)

    def subtractByKey(self, other: "Table"):
        func = f"{self._namespace}.{self._name}-{other._namespace}.{other._name}"
        return self._binary(other, func, _do_subtract_by_key)

    def union(self, other: "Table", func=lambda v1, v2: v1):
        return self._binary(other, func, _do_union)

    # noinspection PyProtectedMember
    def _map_reduce(self, mapper, reducer):
        results = self._session._submit_map_reduce_in_partition(
            mapper, reducer, self._partitions, self._name, self._namespace
        )
        result = results[0]
        # noinspection PyProtectedMember
        return _create_table(
            session=self._session,
            name=result.name,
            namespace=result.namespace,
            partitions=self._partitions,
        )

    def _unary(self, func, do_func):
        # noinspection PyProtectedMember
        results = self._session._submit_unary(func, do_func, self._partitions, self._name, self._namespace)
        result = results[0]
        # noinspection PyProtectedMember
        return _create_table(
            session=self._session,
            name=result.name,
            namespace=result.namespace,
            partitions=self._partitions,
        )

    def _binary(self, other: "Table", func, do_func):
        session_id = self._session.session_id
        left, right = self, other
        if left._partitions != right._partitions:
            if other.count() > self.count():
                left = left.save_as(str(uuid.uuid1()), session_id, partition=right._partitions)
            else:
                right = other.save_as(str(uuid.uuid1()), session_id, partition=left._partitions)

        # noinspection PyProtectedMember
        results = self._session._submit_binary(
            func,
            do_func,
            left._partitions,
            left._name,
            left._namespace,
            right._name,
            right._namespace,
        )
        result: _Operand = results[0]
        # noinspection PyProtectedMember
        return _create_table(
            session=self._session,
            name=result.name,
            namespace=result.namespace,
            partitions=left._partitions,
        )

    def save_as(self, name, namespace, partition=None, need_cleanup=True):
        if partition is None:
            partition = self._partitions
        # noinspection PyProtectedMember
        dup = _create_table(self._session, name, namespace, partition, need_cleanup)
        dup.put_all(self.collect())
        return dup

    def _get_env_for_partition(self, p: int, write=False):
        return _get_env(self._namespace, self._name, str(p), write=write)

    def put(self, k, v):
        k_bytes, v_bytes = _kv_to_bytes(k=k, v=v)
        p = _hash_key_to_partition(k_bytes, self._partitions)
        with self._get_env_for_partition(p, write=True) as env:
            with env.begin(write=True) as txn:
                return txn.put(k_bytes, v_bytes)

    def put_all(self, kv_list: Iterable):
        txn_map = {}
        is_success = True
        with ExitStack() as s:
            for p in range(self._partitions):
                env = s.enter_context(self._get_env_for_partition(p, write=True))
                txn_map[p] = env, env.begin(write=True)
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

    def get(self, k):
        k_bytes = _k_to_bytes(k=k)
        p = _hash_key_to_partition(k_bytes, self._partitions)
        with self._get_env_for_partition(p) as env:
            with env.begin(write=True) as txn:
                old_value_bytes = txn.get(k_bytes)
                return None if old_value_bytes is None else deserialize(old_value_bytes)

    def delete(self, k):
        k_bytes = _k_to_bytes(k=k)
        p = _hash_key_to_partition(k_bytes, self._partitions)
        with self._get_env_for_partition(p, write=True) as env:
            with env.begin(write=True) as txn:
                old_value_bytes = txn.get(k_bytes)
                if txn.delete(k_bytes):
                    return None if old_value_bytes is None else deserialize(old_value_bytes)
                return None


# noinspection PyMethodMayBeStatic
class Session(object):
    def __init__(self, session_id, max_workers=None, logger_config=None):
        self.session_id = session_id
        self._pool = Executor(
            max_workers=max_workers,
            initializer=_watch_thread_react_to_parent_die,
            initargs=(
                os.getpid(),
                logger_config,
            ),
        )

    def __getstate__(self):
        # session won't be pickled
        pass

    def load(self, name, namespace):
        return _load_table(session=self, name=name, namespace=namespace)

    def create_table(self, name, namespace, partitions, need_cleanup, error_if_exist):
        return _create_table(
            session=self,
            name=name,
            namespace=namespace,
            partitions=partitions,
            need_cleanup=need_cleanup,
            error_if_exist=error_if_exist,
        )

    # noinspection PyUnusedLocal
    def parallelize(self, data: Iterable, partition: int, include_key: bool = False, **kwargs):
        if not include_key:
            data = enumerate(data)
        table = _create_table(
            session=self,
            name=str(uuid.uuid1()),
            namespace=self.session_id,
            partitions=partition,
        )
        table.put_all(data)
        return table

    def cleanup(self, name, namespace):
        if not _data_dir.is_dir():
            LOGGER.error(f"illegal data dir: {_data_dir}")
            return

        namespace_dir = _data_dir.joinpath(namespace)

        if not namespace_dir.is_dir():
            return

        if name == "*":
            shutil.rmtree(namespace_dir, True)
            return

        for table in namespace_dir.glob(name):
            shutil.rmtree(table, True)

    def stop(self):
        self.cleanup(name="*", namespace=self.session_id)
        self._pool.shutdown()

    def kill(self):
        self.cleanup(name="*", namespace=self.session_id)
        self._pool.shutdown()

    def _submit_unary(self, func, _do_func, partitions, name, namespace):
        task_info = _TaskInfo(
            self.session_id,
            function_id=str(uuid.uuid1()),
            function_bytes=f_pickle.dumps(func),
        )
        futures = []
        for p in range(partitions):
            futures.append(
                self._pool.submit(_do_func, _UnaryProcess(task_info, _Operand(namespace, name, p, partitions)))
            )
        results = [r.result() for r in futures]
        return results

    def _submit_map_reduce_in_partition(self, mapper, reducer, partitions, name, namespace):
        task_info = _MapReduceTaskInfo(
            self.session_id,
            function_id=str(uuid.uuid1()),
            map_function_bytes=f_pickle.dumps(mapper),
            reduce_function_bytes=f_pickle.dumps(reducer),
        )
        futures = []
        for p in range(partitions):
            futures.append(
                self._pool.submit(
                    _do_map_reduce_in_partitions,
                    _MapReduceProcess(task_info, _Operand(namespace, name, p, partitions)),
                )
            )
        results = [r.result() for r in futures]
        return results

    def _submit_binary(self, func, do_func, partitions, name, namespace, other_name, other_namespace):
        task_info = _TaskInfo(
            self.session_id,
            function_id=str(uuid.uuid1()),
            function_bytes=f_pickle.dumps(func),
        )
        futures = []
        for p in range(partitions):
            left = _Operand(namespace, name, p, partitions)
            right = _Operand(other_namespace, other_name, p, partitions)
            futures.append(self._pool.submit(do_func, _BinaryProcess(task_info, left, right)))
        results = []
        for f in futures:
            r = f.result()
            results.append(r)
        return results


def _get_splits(obj, max_message_size):
    obj_bytes = serialize(obj, protocol=4)
    byte_size = len(obj_bytes)
    num_slice = (byte_size - 1) // max_message_size + 1
    if num_slice <= 1:
        return obj, num_slice
    else:
        _max_size = max_message_size
        kv = [(i, obj_bytes[slice(i * _max_size, (i + 1) * _max_size)]) for i in range(num_slice)]
        return kv, num_slice


class Federation(object):
    def _federation_object_key(self, name: str, tag: str, s_party: Tuple[str, str], d_party: Tuple[str, str]):
        return f"{self._session_id}-{name}-{tag}-{s_party[0]}-{s_party[1]}-{d_party[0]}-{d_party[1]}"

    def __init__(self, session: Session, session_id: str, party: Tuple[str, str]):
        self._session_id = session_id
        self._party = party
        self._session = session
        self._max_message_size = DEFAULT_MESSAGE_MAX_SIZE
        self._other_status_tables = {}
        self._other_object_tables = {}
        self._federation_status_table_cache = None
        self._federation_object_table_cache = None

        self._meta = _FederationMetaManager(session_id, party)

    def destroy(self):
        self._session.cleanup(namespace=self._session_id, name="*")

    # noinspection PyUnusedLocal
    def remote(self, v, name: str, tag: str, parties: List[PartyMeta]):
        log_str = f"federation.standalone.remote.{name}.{tag}"

        if v is None:
            raise ValueError(f"[{log_str}]remote `None` to {parties}")

        LOGGER.debug(f"[{log_str}]remote data, type={type(v)}")

        if isinstance(v, Table):
            dtype = FederationDataType.TABLE
            LOGGER.debug(
                f"[{log_str}]remote "
                f"Table(namespace={v.namespace}, name={v.name}, partitions={v.partitions}), dtype={dtype}"
            )
        else:
            v_splits, num_slice = _get_splits(v, self._max_message_size)
            if num_slice > 1:
                v = _create_table(
                    session=self._session,
                    name=str(uuid.uuid1()),
                    namespace=self._session_id,
                    partitions=1,
                    need_cleanup=True,
                    error_if_exist=False,
                )
                v.put_all(kv_list=v_splits)
                dtype = FederationDataType.SPLIT_OBJECT
                LOGGER.debug(
                    f"[{log_str}]remote "
                    f"Table(namespace={v.namespace}, name={v.name}, partitions={v.partitions}), dtype={dtype}"
                )
            else:
                LOGGER.debug(f"[{log_str}]remote object with type: {type(v)}")
                dtype = FederationDataType.OBJECT

        for party in parties:
            _tagged_key = self._federation_object_key(name, tag, self._party, party)
            if isinstance(v, Table):
                saved_name = str(uuid.uuid1())
                LOGGER.debug(
                    f"[{log_str}]save Table(namespace={v.namespace}, name={v.name}, partitions={v.partitions}) as "
                    f"Table(namespace={v.namespace}, name={saved_name}, partitions={v.partitions})"
                )
                _v = v.save_as(name=saved_name, namespace=v.namespace, need_cleanup=False)
                self._meta.set_status(party, _tagged_key, (_v.name, _v.namespace, dtype))
            else:
                self._meta.set_object(party, _tagged_key, v)
                self._meta.set_status(party, _tagged_key, _tagged_key)

    # noinspection PyProtectedMember
    def get(self, name: str, tag: str, parties: List[PartyMeta]) -> List:
        log_str = f"federation.standalone.get.{name}.{tag}"
        LOGGER.debug(f"[{log_str}]")
        results = []

        for party in parties:
            _tagged_key = self._federation_object_key(name, tag, party, self._party)
            results.append(self._meta.wait_status_set(_tagged_key))

        rtn = []
        for r in results:
            if isinstance(r, tuple):
                # noinspection PyTypeChecker
                table: Table = _load_table(session=self._session, name=r[0], namespace=r[1], need_cleanup=True)

                dtype = r[2]
                LOGGER.debug(
                    f"[{log_str}] got "
                    f"Table(namespace={table.namespace}, name={table.name}, partitions={table.partitions}), dtype={dtype}"
                )

                if dtype == FederationDataType.SPLIT_OBJECT:
                    obj_bytes = b"".join(map(lambda t: t[1], sorted(table.collect(), key=lambda x: x[0])))
                    obj = deserialize(obj_bytes)
                    rtn.append(obj)
                else:
                    rtn.append(table)
            else:
                obj = self._meta.get_object(r)
                if obj is None:
                    raise EnvironmentError(f"federation get None from {parties} with name {name}, tag {tag}")
                rtn.append(obj)
                self._meta.ack_object(r)
                LOGGER.debug(f"[{log_str}] got object with type: {type(obj)}")
            self._meta.ack_status(r)
        return rtn


def _create_table(
    session: "Session",
    name: str,
    namespace: str,
    partitions: int,
    need_cleanup=True,
    error_if_exist=False,
):
    assert isinstance(name, str)
    assert isinstance(namespace, str)
    assert isinstance(partitions, int)
    if (exist_partitions := _TableMetaManager.get_table_meta(namespace, name)) is None:
        _TableMetaManager.add_table_meta(namespace, name, partitions)
    else:
        if error_if_exist:
            raise RuntimeError(f"table already exist: name={name}, namespace={namespace}")
        partitions = exist_partitions

    return Table(
        session=session,
        namespace=namespace,
        name=name,
        partitions=partitions,
        need_cleanup=need_cleanup,
    )


def _load_table(session, name: str, namespace: str, need_cleanup=False):
    partitions = _TableMetaManager.get_table_meta(namespace, name)
    if partitions is None:
        raise RuntimeError(f"table not exist: name={name}, namespace={namespace}")
    return Table(
        session=session,
        namespace=namespace,
        name=name,
        partitions=partitions,
        need_cleanup=need_cleanup,
    )


class _TaskInfo:
    def __init__(self, task_id, function_id, function_bytes):
        self.task_id = task_id
        self.function_id = function_id
        self.function_bytes = function_bytes
        self._function_deserialized = None

    def get_func(self):
        if self._function_deserialized is None:
            self._function_deserialized = f_pickle.loads(self.function_bytes)
        return self._function_deserialized


class _MapReduceTaskInfo:
    def __init__(self, task_id, function_id, map_function_bytes, reduce_function_bytes):
        self.task_id = task_id
        self.function_id = function_id
        self.map_function_bytes = map_function_bytes
        self.reduce_function_bytes = reduce_function_bytes
        self._reduce_function_deserialized = None
        self._mapper_function_deserialized = None

    def get_mapper(self):
        if self._mapper_function_deserialized is None:
            self._mapper_function_deserialized = f_pickle.loads(self.map_function_bytes)
        return self._mapper_function_deserialized

    def get_reducer(self):
        if self._reduce_function_deserialized is None:
            self._reduce_function_deserialized = f_pickle.loads(self.reduce_function_bytes)
        return self._reduce_function_deserialized


class _Operand:
    def __init__(self, namespace, name, partition, num_partitions: int):
        self.namespace = namespace
        self.name = name
        self.partition = partition
        self.num_partitions = num_partitions

    def as_env(self, write=False):
        return _get_env(self.namespace, self.name, str(self.partition), write=write)


class _UnaryProcess:
    def __init__(self, task_info: _TaskInfo, operand: _Operand):
        self.info = task_info
        self.operand = operand

    def output_operand(self):
        return _Operand(self.info.task_id, self.info.function_id, self.operand.partition, self.operand.num_partitions)

    def get_func(self):
        return self.info.get_func()


class _MapReduceProcess:
    def __init__(self, task_info: _MapReduceTaskInfo, operand: _Operand):
        self.info = task_info
        self.operand = operand

    def output_operand(self):
        return _Operand(self.info.task_id, self.info.function_id, self.operand.partition, self.operand.num_partitions)

    def get_mapper(self):
        return self.info.get_mapper()

    def get_reducer(self):
        return self.info.get_reducer()


class _BinaryProcess:
    def __init__(self, task_info: _TaskInfo, left: _Operand, right: _Operand):
        self.info = task_info
        self.left = left
        self.right = right

    def output_operand(self):
        return _Operand(self.info.task_id, self.info.function_id, self.left.partition, self.left.num_partitions)

    def get_func(self):
        return self.info.get_func()


def _get_env(*args, write=False):
    _path = _data_dir.joinpath(*args)
    return _open_env(_path, write=write)


def _open_env(path, write=False):
    path.mkdir(parents=True, exist_ok=True)

    t = 0
    while t < 100:
        try:
            env = lmdb.open(
                path.as_posix(),
                create=True,
                max_dbs=1,
                max_readers=1024,
                lock=write,
                sync=True,
                map_size=10_737_418_240,
            )
            return env
        except lmdb.Error as e:
            if "No such file or directory" in e.args[0]:
                time.sleep(0.01)
                t += 1
            else:
                raise e
    raise lmdb.Error(f"No such file or directory: {path}, with {t} times retry")


def _hash_key_to_partition(key, partitions):
    _key = hashlib.sha1(key).digest()
    if isinstance(_key, bytes):
        _key = int.from_bytes(_key, byteorder="little", signed=False)
    if partitions < 1:
        raise ValueError("partitions must be a positive number")
    b, j = -1, 0
    while j < partitions:
        b = int(j)
        _key = ((_key * 2862933555777941757) + 1) & 0xFFFFFFFFFFFFFFFF
        j = float(b + 1) * (float(1 << 31) / float((_key >> 33) + 1))
    return int(b)


def _do_map(p: _UnaryProcess):
    rtn = p.output_operand()
    with ExitStack() as s:
        source_env = s.enter_context(p.operand.as_env())
        txn_map = {}
        for partition in range(p.operand.num_partitions):
            env = s.enter_context(_get_env(rtn.namespace, rtn.name, str(partition), write=True))
            txn_map[partition] = s.enter_context(env.begin(write=True))
        source_txn = s.enter_context(source_env.begin())
        cursor = s.enter_context(source_txn.cursor())
        for k_bytes, v_bytes in cursor:
            k, v = deserialize(k_bytes), deserialize(v_bytes)
            k1, v1 = p.get_func()(k, v)
            k1_bytes, v1_bytes = serialize(k1), serialize(v1)
            partition = _hash_key_to_partition(k1_bytes, p.operand.num_partitions)
            txn_map[partition].put(k1_bytes, v1_bytes)
    return rtn


def _generator_from_cursor(cursor):
    for k, v in cursor:
        yield deserialize(k), deserialize(v)


def _do_apply_partitions(p: _UnaryProcess):
    with ExitStack() as s:
        rtn = p.output_operand()
        source_env = s.enter_context(p.operand.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        source_txn = s.enter_context(source_env.begin())
        dst_txn = s.enter_context(dst_env.begin(write=True))

        cursor = s.enter_context(source_txn.cursor())
        v = p.get_func()(_generator_from_cursor(cursor))
        if cursor.last():
            k_bytes = cursor.key()
            dst_txn.put(k_bytes, serialize(v))
        return rtn


def _do_map_partitions(p: _UnaryProcess):
    with ExitStack() as s:
        rtn = p.output_operand()
        source_env = s.enter_context(p.operand.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        source_txn = s.enter_context(source_env.begin())
        dst_txn = s.enter_context(dst_env.begin(write=True))

        cursor = s.enter_context(source_txn.cursor())
        v = p.get_func()(_generator_from_cursor(cursor))

        if isinstance(v, Iterable):
            for k1, v1 in v:
                dst_txn.put(serialize(k1), serialize(v1))
        else:
            k_bytes = cursor.key()
            dst_txn.put(k_bytes, serialize(v))
        return rtn


def _do_map_partitions_with_index(p: _UnaryProcess):
    with ExitStack() as s:
        rtn = p.output_operand()
        source_env = s.enter_context(p.operand.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        source_txn = s.enter_context(source_env.begin())
        dst_txn = s.enter_context(dst_env.begin(write=True))

        cursor = s.enter_context(source_txn.cursor())
        v = p.get_func()(p.operand.partition, _generator_from_cursor(cursor))

        if isinstance(v, Iterable):
            for k1, v1 in v:
                dst_txn.put(serialize(k1), serialize(v1))
        else:
            k_bytes = cursor.key()
            dst_txn.put(k_bytes, serialize(v))
        return rtn


def _do_map_reduce_in_partitions(p: _MapReduceProcess):
    rtn = p.output_operand()
    with ExitStack() as s:
        source_env = s.enter_context(p.operand.as_env())
        txn_map = {}
        for partition in range(p.operand.num_partitions):
            env = s.enter_context(_get_env(rtn.namespace, rtn.name, str(partition), write=True))
            txn_map[partition] = s.enter_context(env.begin(write=True))
        source_txn = s.enter_context(source_env.begin())
        cursor = s.enter_context(source_txn.cursor())
        mapped = p.get_mapper()(_generator_from_cursor(cursor))
        if not isinstance(mapped, Iterable):
            raise ValueError("mapper function should return a iterable of pair")
        reducer = p.get_reducer()

        for k, v in mapped:
            k_bytes = serialize(k)
            partition = _hash_key_to_partition(k_bytes, p.operand.num_partitions)
            # todo: not atomic, fix me
            pre_v = txn_map[partition].get(k_bytes, None)
            if pre_v is None:
                txn_map[partition].put(k_bytes, serialize(v))
            else:
                txn_map[partition].put(k_bytes, serialize(reducer(deserialize(pre_v), v)))
    return rtn


def _do_map_values(p: _UnaryProcess):
    rtn = p.output_operand()
    with ExitStack() as s:
        source_env = s.enter_context(p.operand.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        source_txn = s.enter_context(source_env.begin())
        dst_txn = s.enter_context(dst_env.begin(write=True))

        cursor = s.enter_context(source_txn.cursor())
        for k_bytes, v_bytes in cursor:
            v = deserialize(v_bytes)
            v1 = p.get_func()(v)
            dst_txn.put(k_bytes, serialize(v1))
    return rtn


def _do_flat_map(p: _UnaryProcess):
    rtn = p.output_operand()
    with ExitStack() as s:
        source_env = s.enter_context(p.operand.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        source_txn = s.enter_context(source_env.begin())
        dst_txn = s.enter_context(dst_env.begin(write=True))

        cursor = s.enter_context(source_txn.cursor())
        for k_bytes, v_bytes in cursor:
            k = deserialize(k_bytes)
            v = deserialize(v_bytes)
            map_result = p.get_func()(k, v)
            for result_k, result_v in map_result:
                dst_txn.put(serialize(result_k), serialize(result_v))
    return rtn


def _do_reduce(p: _UnaryProcess):
    value = None
    with ExitStack() as s:
        source_env = s.enter_context(p.operand.as_env())
        source_txn = s.enter_context(source_env.begin())
        cursor = s.enter_context(source_txn.cursor())
        for _, v_bytes in cursor:
            v = deserialize(v_bytes)
            if value is None:
                value = v
            else:
                value = p.get_func()(value, v)
    return value


def _do_glom(p: _UnaryProcess):
    rtn = p.output_operand()
    with ExitStack() as s:
        source_env = s.enter_context(p.operand.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        source_txn = s.enter_context(source_env.begin())
        dest_txn = s.enter_context(dst_env.begin(write=True))

        cursor = s.enter_context(source_txn.cursor())
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
    fraction, seed = deserialize(p.info.function_bytes)
    with ExitStack() as s:
        source_env = s.enter_context(p.operand.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        source_txn = s.enter_context(source_env.begin())
        dst_txn = s.enter_context(dst_env.begin(write=True))

        cursor = s.enter_context(source_txn.cursor())
        cursor.first()
        random_state = np.random.RandomState(seed)
        for k, v in cursor:
            # noinspection PyArgumentList
            if random_state.rand() < fraction:
                dst_txn.put(k, v)
    return rtn


def _do_filter(p: _UnaryProcess):
    rtn = p.output_operand()
    with ExitStack() as s:
        source_env = s.enter_context(p.operand.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        source_txn = s.enter_context(source_env.begin())
        dst_txn = s.enter_context(dst_env.begin(write=True))

        cursor = s.enter_context(source_txn.cursor())
        for k_bytes, v_bytes in cursor:
            k = deserialize(k_bytes)
            v = deserialize(v_bytes)
            if p.get_func()(k, v):
                dst_txn.put(k_bytes, v_bytes)
    return rtn


def _do_subtract_by_key(p: _BinaryProcess):
    rtn = p.output_operand()
    with ExitStack() as s:
        left_op = p.left
        right_op = p.right
        right_env = s.enter_context(right_op.as_env())
        left_env = s.enter_context(left_op.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        left_txn = s.enter_context(left_env.begin())
        right_txn = s.enter_context(right_env.begin())
        dst_txn = s.enter_context(dst_env.begin(write=True))

        cursor = s.enter_context(left_txn.cursor())
        for k_bytes, left_v_bytes in cursor:
            right_v_bytes = right_txn.get(k_bytes)
            if right_v_bytes is None:
                dst_txn.put(k_bytes, left_v_bytes)
    return rtn


def _do_join(p: _BinaryProcess):
    rtn = p.output_operand()
    with ExitStack() as s:
        right_env = s.enter_context(p.right.as_env())
        left_env = s.enter_context(p.left.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        left_txn = s.enter_context(left_env.begin())
        right_txn = s.enter_context(right_env.begin())
        dst_txn = s.enter_context(dst_env.begin(write=True))

        cursor = s.enter_context(left_txn.cursor())
        for k_bytes, v1_bytes in cursor:
            v2_bytes = right_txn.get(k_bytes)
            if v2_bytes is None:
                continue
            v1 = deserialize(v1_bytes)
            v2 = deserialize(v2_bytes)
            try:
                v3 = p.get_func()(v1, v2)
            except Exception as e:
                raise RuntimeError(
                    f"Error when joining:\n" f"left:\n" f"{v1}\n" f"right:\n" f"{v2}\n" f"error: {e}"
                ) from e
            dst_txn.put(k_bytes, serialize(v3))
    return rtn


def _do_union(p: _BinaryProcess):
    rtn = p.output_operand()
    with ExitStack() as s:
        left_env = s.enter_context(p.left.as_env())
        right_env = s.enter_context(p.right.as_env())
        dst_env = s.enter_context(rtn.as_env(write=True))

        left_txn = s.enter_context(left_env.begin())
        right_txn = s.enter_context(right_env.begin())
        dst_txn = s.enter_context(dst_env.begin(write=True))

        # process left op
        with left_txn.cursor() as left_cursor:
            for k_bytes, left_v_bytes in left_cursor:
                right_v_bytes = right_txn.get(k_bytes)
                if right_v_bytes is None:
                    dst_txn.put(k_bytes, left_v_bytes)
                else:
                    left_v = deserialize(left_v_bytes)
                    right_v = deserialize(right_v_bytes)
                    final_v = p.get_func()(left_v, right_v)
                    dst_txn.put(k_bytes, serialize(final_v))

        # process right op
        with right_txn.cursor() as right_cursor:
            for k_bytes, right_v_bytes in right_cursor:
                final_v_bytes = dst_txn.get(k_bytes)
                if final_v_bytes is None:
                    dst_txn.put(k_bytes, right_v_bytes)
    return rtn


def _kv_to_bytes(k, v):
    return serialize(k), serialize(v)


def _k_to_bytes(k):
    return serialize(k)


class _FederationMetaManager:
    STATUS_TABLE_NAME_PREFIX = "__federation_status__"
    OBJECT_TABLE_NAME_PREFIX = "__federation_object__"

    def __init__(self, session_id, party: Tuple[str, str]) -> None:
        self.session_id = session_id
        self.party = party
        self._env = {}

    def wait_status_set(self, key):
        value = self.get_status(key)
        while value is None:
            time.sleep(0.1)
            value = self.get_status(key)
        LOGGER.debug("[GET] Got {} type {}".format(key, "Table" if isinstance(value, tuple) else "Object"))
        return value

    def get_status(self, key):
        return self._get(self._get_status_table_name(self.party), key)

    def set_status(self, party: Tuple[str, str], key: str, value):
        return self._set(self._get_status_table_name(party), key, value)

    def ack_status(self, key):
        return self._ack(self._get_status_table_name(self.party), key)

    def get_object(self, key):
        return self._get(self._get_object_table_name(self.party), key)

    def set_object(self, party: Tuple[str, str], key, value):
        return self._set(self._get_object_table_name(party), key, value)

    def ack_object(self, key):
        return self._ack(self._get_object_table_name(self.party), key)

    def _get_status_table_name(self, party: Tuple[str, str]):
        return f"{self.STATUS_TABLE_NAME_PREFIX}.{party[0]}_{party[1]}"

    def _get_object_table_name(self, party: Tuple[str, str]):
        return f"{self.OBJECT_TABLE_NAME_PREFIX}.{party[0]}_{party[1]}"

    def _get_env(self, name):
        if name not in self._env:
            self._env[name] = _get_env(self.session_id, name, str(0), write=True)
        return self._env[name]

    def _get(self, name, key):
        env = self._get_env(name)
        with env.begin(write=False) as txn:
            old_value_bytes = txn.get(serialize(key))
            if old_value_bytes is not None:
                old_value_bytes = deserialize(old_value_bytes)
            return old_value_bytes

    def _set(self, name, key, value):
        env = self._get_env(name)
        with env.begin(write=True) as txn:
            return txn.put(serialize(key), serialize(value))

    def _ack(self, name, key):
        env = self._get_env(name)
        with env.begin(write=True) as txn:
            txn.delete(serialize(key))


class _TableMetaManager:
    namespace = "__META__"
    name = "fragments"
    num_partitions = 10
    _env = {}

    @classmethod
    def _get_meta_env(cls, namespace: str, name: str):
        k_bytes = _k_to_bytes(f"{namespace}.{name}")
        p = _hash_key_to_partition(k_bytes, cls.num_partitions)
        if p not in cls._env:
            cls._env[p] = _get_env(cls.namespace, cls.name, str(p), write=True)
        return k_bytes, cls._env[p]

    @classmethod
    def add_table_meta(cls, namespace: str, name: str, num_partitions: int):
        k_bytes, env = cls._get_meta_env(namespace, name)
        with env.begin(write=True) as txn:
            return txn.put(k_bytes, serialize(num_partitions))

    @classmethod
    def get_table_meta(cls, namespace: str, name: str):
        k_bytes, env = cls._get_meta_env(namespace, name)
        with env.begin(write=False) as txn:
            old_value_bytes = txn.get(k_bytes)
            if old_value_bytes is not None:
                old_value_bytes = deserialize(old_value_bytes)
            return old_value_bytes

    @classmethod
    def destroy_table(cls, namespace: str, name: str):
        k_bytes, env = cls._get_meta_env(namespace, name)
        with env.begin(write=True) as txn:
            txn.delete(k_bytes)
        path = _data_dir.joinpath(namespace, name)
        shutil.rmtree(path, ignore_errors=True)
