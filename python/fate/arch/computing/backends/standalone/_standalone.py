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
import shutil
import signal
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor as Executor
from contextlib import ExitStack
from functools import partial
from heapq import heapify, heappop, heapreplace
from operator import is_not
from pathlib import Path
from typing import Callable, Any, Iterable, Optional
from typing import List, Tuple, Literal

import cloudpickle as f_pickle
import lmdb

PartyMeta = Tuple[Literal["guest", "host", "arbiter", "local"], str]

logger = logging.getLogger(__name__)


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


class BasicProcessPool:
    def __init__(self, pool, log_level):
        self._pool = pool
        self._exception_tb = {}
        self.log_level = log_level

    def submit(self, func, process_infos):
        features = []
        outputs = {}
        num_partitions = len(process_infos)

        for p, process_info in enumerate(process_infos):
            features.append(
                self._pool.submit(
                    BasicProcessPool._process_wrapper,
                    func,
                    process_info,
                    self.log_level,
                )
            )

        from concurrent.futures import wait, FIRST_COMPLETED

        not_done = features
        while not_done:
            done, not_done = wait(not_done, return_when=FIRST_COMPLETED)
            for f in done:
                partition_id, output, e = f.result()
                if e is not None:
                    logger.error(f"partition {partition_id} exec failed: {e}")
                    raise RuntimeError(f"Partition {partition_id} exec failed: {e}")
                else:
                    outputs[partition_id] = output

        outputs = [outputs[p] for p in range(num_partitions)]
        return outputs

    @classmethod
    def _process_wrapper(cls, do_func, process_info, log_level):
        try:
            if log_level is not None:
                pass
            output = do_func(process_info)
            return process_info.partition_id, output, None
        except Exception as e:
            logger.error(f"exception in rank {process_info.partition_id}: {e}")
            return process_info.partition_id, None, e

    def shutdown(self):
        self._pool.shutdown()


# noinspection PyPep8Naming
class Table(object):
    def __init__(
        self,
        session: "Session",
        data_dir: str,
        namespace: str,
        name: str,
        partitions,
        key_serdes_type: int,
        value_serdes_type: int,
        partitioner_type: int,
        need_cleanup=True,
    ):
        self._need_cleanup = need_cleanup
        self._data_dir = data_dir
        self._namespace = namespace
        self._name = name
        self._partitions = partitions
        self._session = session
        self._key_serdes_type = key_serdes_type
        self._value_serdes_type = value_serdes_type
        self._partitioner_type = partitioner_type

    @property
    def num_partitions(self):
        return self._partitions

    @property
    def key_serdes_type(self):
        return self._key_serdes_type

    @property
    def value_serdes_type(self):
        return self._value_serdes_type

    @property
    def partitioner_type(self):
        return self._partitioner_type

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
        for p in range(self.num_partitions):
            with self._get_env_for_partition(p, write=True) as env:
                db = env.open_db()
                with env.begin(write=True) as txn:
                    txn.drop(db)
        _TableMetaManager.destroy_table(data_dir=self._data_dir, namespace=self._namespace, name=self._name)

    def take(self, num, **kwargs):
        if num <= 0:
            raise ValueError(f"{num} <= 0")
        return list(itertools.islice(self.collect(**kwargs), num))

    def count(self):
        cnt = 0
        for p in range(self.num_partitions):
            with self._get_env_for_partition(p) as env:
                cnt += env.stat()["entries"]
        return cnt

    # noinspection PyUnusedLocal
    def collect(self, **kwargs):
        iterators = []
        with ExitStack() as s:
            for p in range(self.num_partitions):
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
                yield key, value
                if it.next():
                    entry[0], entry[1] = it.item()
                    heapreplace(entries, entry)
                else:
                    _, _, _, it = heappop(entries)

    def reduce(self, func):
        return self._session.submit_reduce(
            func,
            data_dir=self._data_dir,
            num_partitions=self.num_partitions,
            name=self._name,
            namespace=self._namespace,
        )

    def binary_sorted_map_partitions_with_index(
        self,
        other: "Table",
        binary_map_partitions_with_index_op: Callable[[int, Iterable, Iterable], Iterable],
        key_serdes_type,
        partitioner_type,
        output_value_serdes_type,
        need_cleanup=True,
        output_name=None,
        output_namespace=None,
        output_data_dir=None,
    ):
        if output_data_dir is None:
            output_data_dir = self._data_dir
        if output_name is None:
            output_name = str(uuid.uuid1())
        if output_namespace is None:
            output_namespace = self._namespace

        self._session._submit_sorted_binary_map_partitions_with_index(
            func=binary_map_partitions_with_index_op,
            do_func=_do_binary_sorted_map_with_index,
            num_partitions=self.num_partitions,
            first_input_data_dir=self._data_dir,
            first_input_name=self._name,
            first_input_namespace=self._namespace,
            second_input_data_dir=other._data_dir,
            second_input_name=other._name,
            second_input_namespace=other._namespace,
            output_data_dir=output_data_dir,
            output_name=output_name,
            output_namespace=output_namespace,
        )
        return _create_table(
            session=self._session,
            data_dir=self._data_dir,
            name=output_name,
            namespace=output_namespace,
            partitions=self.num_partitions,
            need_cleanup=need_cleanup,
            key_serdes_type=key_serdes_type,
            value_serdes_type=output_value_serdes_type,
            partitioner_type=partitioner_type,
        )

    def map_reduce_partitions_with_index(
        self,
        map_partition_op: Callable[[int, Iterable], Iterable],
        reduce_partition_op: Optional[Callable[[Any, Any], Any]],
        output_partitioner: Optional[Callable[[bytes, int], int]],
        shuffle,
        output_key_serdes_type,
        output_value_serdes_type,
        output_partitioner_type,
        output_num_partitions,
        need_cleanup=True,
        output_name=None,
        output_namespace=None,
        output_data_dir=None,
    ):
        if output_data_dir is None:
            output_data_dir = self._data_dir
        if output_name is None:
            output_name = str(uuid.uuid1())
        if output_namespace is None:
            output_namespace = self._namespace
        if not shuffle:
            assert output_num_partitions == self.num_partitions and output_partitioner_type == self.partitioner_type
            # noinspection PyProtectedMember
            self._session._submit_map_reduce_partitions_with_index(
                _do_mrwi_no_shuffle,
                mapper=map_partition_op,
                reducer=reduce_partition_op,
                input_num_partitions=self.num_partitions,
                input_data_dir=self._data_dir,
                input_name=self._name,
                input_namespace=self._namespace,
                output_num_partitions=output_num_partitions,
                output_data_dir=output_data_dir,
                output_name=output_name,
                output_namespace=output_namespace,
                output_partitioner=output_partitioner,
            )
            return _create_table(
                session=self._session,
                data_dir=output_data_dir,
                name=output_name,
                namespace=output_namespace,
                partitions=output_num_partitions,
                need_cleanup=need_cleanup,
                key_serdes_type=output_key_serdes_type,
                value_serdes_type=output_value_serdes_type,
                partitioner_type=output_partitioner_type,
            )

        if reduce_partition_op is None:
            _do_shuffle_write_func = _do_mrwi_map_and_shuffle_write_unique
            _do_shuffle_read_func = _do_mrwi_shuffle_read_no_reduce
        else:
            _do_shuffle_write_func = _do_mrwi_map_and_shuffle_write
            _do_shuffle_read_func = _do_mrwi_shuffle_read_and_reduce
        # Step 1: do map and write intermediate results to cache table
        intermediate_name = str(uuid.uuid1())
        intermediate_namespace = self._namespace
        intermediate_data_dir = self._data_dir
        # noinspection PyProtectedMember
        self._session._submit_map_reduce_partitions_with_index(
            _do_shuffle_write_func,
            mapper=map_partition_op,
            reducer=None,
            input_data_dir=self._data_dir,
            input_num_partitions=self.num_partitions,
            input_name=self._name,
            input_namespace=self._namespace,
            output_data_dir=intermediate_data_dir,
            output_num_partitions=output_num_partitions,
            output_name=intermediate_name,
            output_namespace=intermediate_namespace,
            output_partitioner=output_partitioner,
        )
        # Step 2: do shuffle read and reduce
        # noinspection PyProtectedMember
        self._session._submit_map_reduce_partitions_with_index(
            _do_shuffle_read_func,
            mapper=None,
            reducer=reduce_partition_op,
            input_data_dir=intermediate_data_dir,
            input_num_partitions=self.num_partitions,
            input_name=intermediate_name,
            input_namespace=intermediate_namespace,
            output_data_dir=output_data_dir,
            output_num_partitions=output_num_partitions,
            output_name=output_name,
            output_namespace=output_namespace,
        )
        output = _create_table(
            session=self._session,
            data_dir=output_data_dir,
            name=output_name,
            namespace=output_namespace,
            partitions=output_num_partitions,
            need_cleanup=need_cleanup,
            key_serdes_type=output_key_serdes_type,
            value_serdes_type=output_value_serdes_type,
            partitioner_type=output_partitioner_type,
        )

        # drop cache table
        for p in range(self._partitions):
            with _get_env_with_data_dir(
                intermediate_data_dir, intermediate_namespace, intermediate_name, str(p), write=True
            ) as env:
                db = env.open_db()
                with env.begin(write=True) as txn:
                    txn.drop(db)

        path = Path(self._data_dir).joinpath(intermediate_namespace, intermediate_name)
        shutil.rmtree(path, ignore_errors=True)
        return output

    def copy_as(self, name, namespace, need_cleanup=True):
        return self.map_reduce_partitions_with_index(
            map_partition_op=lambda i, x: x,
            reduce_partition_op=None,
            output_partitioner=None,
            shuffle=False,
            need_cleanup=need_cleanup,
            output_name=name,
            output_namespace=namespace,
            output_key_serdes_type=self._key_serdes_type,
            output_value_serdes_type=self._value_serdes_type,
            output_partitioner_type=self._partitioner_type,
            output_num_partitions=self.num_partitions,
        )

    def _get_env_for_partition(self, p: int, write=False):
        return _get_env_with_data_dir(self._data_dir, self._namespace, self._name, str(p), write=write)

    def put(self, k_bytes: bytes, v_bytes: bytes, partitioner: Callable[[bytes, int], int] = None):
        p = partitioner(k_bytes, self._partitions)
        with self._get_env_for_partition(p, write=True) as env:
            with env.begin(write=True) as txn:
                return txn.put(k_bytes, v_bytes)

    def put_all(self, kv_list: Iterable[Tuple[bytes, bytes]], partitioner: Callable[[bytes, int], int]):
        txn_map = {}
        with ExitStack() as s:
            for p in range(self._partitions):
                env = s.enter_context(self._get_env_for_partition(p, write=True))
                txn_map[p] = env, env.begin(write=True)
            try:
                for k_bytes, v_bytes in kv_list:
                    p = partitioner(k_bytes, self._partitions)
                    if not txn_map[p][1].put(k_bytes, v_bytes):
                        break
            except Exception as e:
                for p, (env, txn) in txn_map.items():
                    txn.abort()
                raise e
            else:
                for p, (env, txn) in txn_map.items():
                    txn.commit()

    def get(self, k_bytes: bytes, partitioner: Callable[[bytes, int], int]) -> bytes:
        p = partitioner(k_bytes, self._partitions)
        with self._get_env_for_partition(p) as env:
            with env.begin(write=True) as txn:
                return txn.get(k_bytes)

    def delete(self, k_bytes: bytes, partitioner: Callable[[bytes, int], int]):
        p = partitioner(k_bytes, self._partitions)
        with self._get_env_for_partition(p, write=True) as env:
            with env.begin(write=True) as txn:
                old_value_bytes = txn.get(k_bytes)
                if txn.delete(k_bytes):
                    return old_value_bytes
                return None


# noinspection PyMethodMayBeStatic
class Session(object):
    def __init__(
        self,
        session_id,
        data_dir: str,
        max_workers=None,
        logger_config=None,
        executor_pool_cls=BasicProcessPool,
    ):
        self.session_id = session_id
        self._data_dir = data_dir
        self._max_workers = max_workers
        if self._max_workers is None:
            self._max_workers = os.cpu_count()

        self._enable_process_logger = True
        if self._enable_process_logger:
            log_level = logging.getLevelName(logger.getEffectiveLevel())
        else:
            log_level = None
        self._pool = executor_pool_cls(
            pool=Executor(
                max_workers=max_workers,
                initializer=_watch_thread_react_to_parent_die,
                initargs=(
                    os.getpid(),
                    logger_config,
                ),
            ),
            log_level=log_level,
        )

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def max_workers(self):
        return self._max_workers

    def __getstate__(self):
        # session won't be pickled
        pass

    def load(self, name, namespace):
        return _load_table(session=self, data_dir=self._data_dir, name=name, namespace=namespace)

    def create_table(
        self,
        name,
        namespace,
        partitions,
        need_cleanup,
        error_if_exist,
        key_serdes_type,
        value_serdes_type,
        partitioner_type,
    ):
        return _create_table(
            session=self,
            data_dir=self._data_dir,
            name=name,
            namespace=namespace,
            partitions=partitions,
            need_cleanup=need_cleanup,
            error_if_exist=error_if_exist,
            key_serdes_type=key_serdes_type,
            value_serdes_type=value_serdes_type,
            partitioner_type=partitioner_type,
        )

    # noinspection PyUnusedLocal
    def parallelize(
        self,
        data: Iterable,
        partition: int,
        partitioner: Callable[[bytes, int], int],
        key_serdes_type,
        value_serdes_type,
        partitioner_type,
    ):
        table = _create_table(
            session=self,
            data_dir=self._data_dir,
            name=str(uuid.uuid1()),
            namespace=self.session_id,
            partitions=partition,
            need_cleanup=True,
            key_serdes_type=key_serdes_type,
            value_serdes_type=value_serdes_type,
            partitioner_type=partitioner_type,
        )
        table.put_all(data, partitioner=partitioner)
        return table

    def cleanup(self, name, namespace):
        path = Path(self._data_dir)
        if not path.is_dir():
            return
        namespace_dir = path.joinpath(namespace)
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

    def submit_reduce(self, func, data_dir: str, num_partitions: int, name: str, namespace: str):
        rs = self._pool.submit(
            _do_reduce,
            [
                _ReduceProcess(p, _TaskInputInfo(data_dir, namespace, name, num_partitions), _ReduceFunctorInfo(func))
                for p in range(num_partitions)
            ],
        )
        rs = [r for r in filter(partial(is_not, None), rs)]
        if len(rs) <= 0:
            return None
        rtn = rs[0]
        for r in rs[1:]:
            rtn = func(rtn, r)
        return rtn

    def _submit_map_reduce_partitions_with_index(
        self,
        _do_func,
        mapper,
        reducer,
        input_data_dir: str,
        input_num_partitions,
        input_name,
        input_namespace,
        output_data_dir: str,
        output_num_partitions,
        output_name,
        output_namespace,
        output_partitioner=None,
    ):
        input_info = _TaskInputInfo(input_data_dir, input_namespace, input_name, input_num_partitions)
        output_info = _TaskOutputInfo(
            output_data_dir, output_namespace, output_name, output_num_partitions, partitioner=output_partitioner
        )
        return self._submit_process(
            _do_func,
            [
                _MapReduceProcess(
                    partition_id=p,
                    input_info=input_info,
                    output_info=output_info,
                    operator_info=_MapReduceFunctorInfo(mapper=mapper, reducer=reducer),
                )
                for p in range(max(input_num_partitions, output_num_partitions))
            ],
        )

    def _submit_sorted_binary_map_partitions_with_index(
        self,
        func,
        do_func,
        num_partitions: int,
        first_input_data_dir: str,
        first_input_name: str,
        first_input_namespace: str,
        second_input_data_dir: str,
        second_input_name: str,
        second_input_namespace: str,
        output_data_dir: str,
        output_name: str,
        output_namespace: str,
    ):
        first_input_info = _TaskInputInfo(
            first_input_data_dir, first_input_namespace, first_input_name, num_partitions
        )
        second_input_info = _TaskInputInfo(
            second_input_data_dir, second_input_namespace, second_input_name, num_partitions
        )
        output_info = _TaskOutputInfo(output_data_dir, output_namespace, output_name, num_partitions, partitioner=None)
        return self._submit_process(
            do_func,
            [
                _BinarySortedMapProcess(
                    partition_id=p,
                    first_input_info=first_input_info,
                    second_input_info=second_input_info,
                    output_info=output_info,
                    operator_info=_BinarySortedMapFunctorInfo(func),
                )
                for p in range(num_partitions)
            ],
        )

    def _submit_process(self, do_func, process_infos):
        return self._pool.submit(do_func, process_infos)


class Federation(object):
    def _federation_object_key(self, name: str, tag: str, s_party: Tuple[str, str], d_party: Tuple[str, str]) -> bytes:
        return f"{self._session_id}-{name}-{tag}-{s_party[0]}-{s_party[1]}-{d_party[0]}-{d_party[1]}".encode("utf-8")

    def __init__(self, session: Session, data_dir: str, session_id: str, party: Tuple[str, str]):
        self._session = session
        self._data_dir = data_dir
        self._session_id = session_id
        self._party = party
        self._other_status_tables = {}
        self._other_object_tables = {}
        self._federation_status_table_cache = None
        self._federation_object_table_cache = None

        self._meta = _FederationMetaManager(session_id=session_id, data_dir=data_dir, party=party)

    @classmethod
    def create(cls, session: Session, session_id: str, party: Tuple[str, str]):
        federation = cls(session, session._data_dir, session_id, party)
        return federation

    def destroy(self):
        self._session.cleanup(namespace=self._session_id, name="*")

    def push_table(self, table, name: str, tag: str, parties: List[PartyMeta]):
        for party in parties:
            _tagged_key = self._federation_object_key(name, tag, self._party, party)
            saved_name = str(uuid.uuid1())
            _table = table.copy_as(name=saved_name, namespace=table.namespace, need_cleanup=False)
            self._meta.set_status(party, _tagged_key, _serialize_tuple_of_str(_table.name, _table.namespace))

    def push_bytes(self, v: bytes, name: str, tag: str, parties: List[PartyMeta]):
        for party in parties:
            _tagged_key = self._federation_object_key(name, tag, self._party, party)
            self._meta.set_object(party, _tagged_key, v)
            self._meta.set_status(party, _tagged_key, _tagged_key)

    def pull_table(self, name: str, tag: str, parties: List[PartyMeta]) -> List[Table]:
        results: List[bytes] = []
        for party in parties:
            _tagged_key = self._federation_object_key(name, tag, party, self._party)

            results.append(self._meta.wait_status_set(_tagged_key))

        rtn = []
        for r in results:
            name, namespace = _deserialize_tuple_of_str(self._meta.get_status(r))
            table: Table = _load_table(
                session=self._session, data_dir=self._data_dir, name=name, namespace=namespace, need_cleanup=True
            )
            rtn.append(table)
            self._meta.ack_status(r)
        return rtn

    def pull_bytes(self, name: str, tag: str, parties: List[PartyMeta]) -> List[bytes]:
        results = []
        for party in parties:
            _tagged_key = self._federation_object_key(name, tag, party, self._party)
            results.append(self._meta.wait_status_set(_tagged_key))

        rtn = []
        for r in results:
            obj = self._meta.get_object(r)
            if obj is None:
                raise EnvironmentError(f"object not found: {r}")
            rtn.append(obj)
            self._meta.ack_object(r)
            self._meta.ack_status(r)
        return rtn


def _create_table(
    session: "Session",
    data_dir: str,
    name: str,
    namespace: str,
    partitions: int,
    key_serdes_type: int,
    value_serdes_type: int,
    partitioner_type: int,
    need_cleanup=True,
    error_if_exist=False,
):
    assert isinstance(name, str)
    assert isinstance(namespace, str)
    assert isinstance(partitions, int)
    if (exist_partitions := _TableMetaManager.get_table_meta(data_dir, namespace, name)) is None:
        _TableMetaManager.add_table_meta(
            data_dir, namespace, name, partitions, key_serdes_type, value_serdes_type, partitioner_type
        )
    else:
        if error_if_exist:
            raise RuntimeError(f"table already exist: name={name}, namespace={namespace}")
        partitions = exist_partitions

    return Table(
        session=session,
        data_dir=data_dir,
        namespace=namespace,
        name=name,
        partitions=partitions,
        key_serdes_type=key_serdes_type,
        value_serdes_type=value_serdes_type,
        partitioner_type=partitioner_type,
        need_cleanup=need_cleanup,
    )


def _load_table(session, data_dir: str, name: str, namespace: str, need_cleanup=False):
    table_meta = _TableMetaManager.get_table_meta(data_dir, namespace, name)
    if table_meta is None:
        raise RuntimeError(f"table not exist: name={name}, namespace={namespace}")
    return Table(
        session=session,
        data_dir=data_dir,
        namespace=namespace,
        name=name,
        need_cleanup=need_cleanup,
        partitions=table_meta.num_partitions,
        key_serdes_type=table_meta.key_serdes_type,
        value_serdes_type=table_meta.value_serdes_type,
        partitioner_type=table_meta.partitioner_type,
    )


class _TaskInputInfo:
    def __init__(self, data_dir: str, namespace: str, name: str, num_partitions: int):
        self.data_dir = data_dir
        self.namespace = namespace
        self.name = name
        self.num_partitions = num_partitions

    def get_env(self, pid, write=False):
        return _get_env_with_data_dir(self.data_dir, self.namespace, self.name, str(pid), write=write)


class _TaskOutputInfo:
    def __init__(self, data_dir: str, namespace: str, name: str, num_partitions: int, partitioner):
        self.data_dir = data_dir
        self.namespace = namespace
        self.name = name
        self.num_partitions = num_partitions
        self.partitioner = partitioner

    def get_env(self, pid, write=True):
        return _get_env_with_data_dir(self.data_dir, self.namespace, self.name, str(pid), write=write)

    def get_partition_id(self, key):
        if self.partitioner is None:
            raise RuntimeError("partitioner is None")
        return self.partitioner(key, self.num_partitions)


class _MapReduceFunctorInfo:
    def __init__(self, mapper, reducer):
        if mapper is not None:
            self.mapper_bytes = f_pickle.dumps(mapper)
        else:
            self.mapper_bytes = None
        if reducer is not None:
            self.reducer_bytes = f_pickle.dumps(reducer)
        else:
            self.reducer_bytes = None

    def get_mapper(self):
        if self.mapper_bytes is None:
            raise RuntimeError("mapper is None")
        return f_pickle.loads(self.mapper_bytes)

    def get_reducer(self):
        if self.reducer_bytes is None:
            raise RuntimeError("reducer is None")
        return f_pickle.loads(self.reducer_bytes)


class _BinarySortedMapFunctorInfo:
    def __init__(self, mapper):
        if mapper is not None:
            self.mapper_bytes = f_pickle.dumps(mapper)
        else:
            self.mapper_bytes = None

    def get_mapper(self):
        if self.mapper_bytes is None:
            raise RuntimeError("mapper is None")
        return f_pickle.loads(self.mapper_bytes)


class _ReduceFunctorInfo:
    def __init__(self, reducer):
        if reducer is not None:
            self.reducer_bytes = f_pickle.dumps(reducer)
        else:
            self.reducer_bytes = None

    def get_reducer(self):
        if self.reducer_bytes is None:
            raise RuntimeError("reducer is None")
        return f_pickle.loads(self.reducer_bytes)


class _ReduceProcess:
    def __init__(
        self,
        partition_id: int,
        input_info: _TaskInputInfo,
        operator_info: _ReduceFunctorInfo,
    ):
        self.partition_id = partition_id
        self.input_info = input_info
        self.operator_info = operator_info

    def as_input_env(self, pid, write=False):
        return self.input_info.get_env(pid, write=write)

    def input_cursor(self, stack: ExitStack):
        return stack.enter_context(stack.enter_context(self.as_input_env(self.partition_id).begin()).cursor())

    def get_reducer(self):
        return self.operator_info.get_reducer()


class _MapReduceProcess:
    def __init__(
        self,
        partition_id: int,
        input_info: _TaskInputInfo,
        output_info: _TaskOutputInfo,
        operator_info: _MapReduceFunctorInfo,
    ):
        self.partition_id = partition_id
        self.input_info = input_info
        self.output_info = output_info
        self.operator_info = operator_info

    def get_input_partition_num(self):
        return self.input_info.num_partitions

    def get_output_partition_num(self):
        return self.output_info.num_partitions

    def get_input_env(self, pid, write=False):
        return self.input_info.get_env(pid, write=write)

    def get_output_env(self, pid, write=True):
        return self.output_info.get_env(pid, write=write)

    def get_input_cursor(self, stack: ExitStack, pid=None):
        if pid is None:
            pid = self.partition_id
        if isinstance(pid, int) and pid >= self.input_info.num_partitions:
            raise RuntimeError(f"pid {pid} >= input_info.num_partitions {self.input_info.num_partitions}")
        return stack.enter_context(
            stack.enter_context(stack.enter_context(self.get_input_env(pid, write=False)).begin(write=False)).cursor()
        )

    def has_partition(self, pid):
        return pid < self.input_info.num_partitions

    def get_output_transaction(self, pid, stack: ExitStack):
        return stack.enter_context(stack.enter_context(self.get_output_env(pid, write=True)).begin(write=True))

    def get_output_partition_id(self, key: bytes):
        return self.output_info.get_partition_id(key)

    def get_mapper(self):
        return self.operator_info.get_mapper()

    def get_reducer(self):
        return self.operator_info.get_reducer()


class _BinarySortedMapProcess:
    def __init__(
        self,
        partition_id,
        first_input_info: _TaskInputInfo,
        second_input_info: _TaskInputInfo,
        output_info: _TaskOutputInfo,
        operator_info: _BinarySortedMapFunctorInfo,
    ):
        self.partition_id = partition_id
        self.first_input = first_input_info
        self.second_input = second_input_info
        self.output_info = output_info
        self.operator_info = operator_info

    def get_input_partition_num(self):
        return self.first_input.num_partitions

    def get_output_partition_num(self):
        return self.output_info.num_partitions

    def get_first_input_env(self, pid, write=False):
        return self.first_input.get_env(pid, write=write)

    def get_second_input_env(self, pid, write=False):
        return self.second_input.get_env(pid, write=write)

    def get_output_env(self, pid, write=True):
        return self.output_info.get_env(pid, write=write)

    def get_first_input_cursor(self, stack: ExitStack, pid=None):
        if pid is None:
            pid = self.partition_id
        return stack.enter_context(
            stack.enter_context(
                stack.enter_context(self.get_first_input_env(pid, write=False)).begin(write=False)
            ).cursor()
        )

    def get_second_input_cursor(self, stack: ExitStack, pid=None):
        if pid is None:
            pid = self.partition_id
        return stack.enter_context(
            stack.enter_context(
                stack.enter_context(self.get_second_input_env(pid, write=False)).begin(write=False)
            ).cursor()
        )

    def get_output_transaction(self, pid, stack: ExitStack):
        return stack.enter_context(stack.enter_context(self.get_output_env(pid, write=True)).begin(write=True))

    def get_output_partition_id(self, key: bytes):
        return self.output_info.get_partition_id(key)

    def get_func(self):
        return self.operator_info.get_mapper()


def _get_env_with_data_dir(data_dir: str, *args, write=False):
    _path = Path(data_dir).joinpath(*args)
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
                time.sleep(0.001)
                t += 1
            else:
                raise e
    raise lmdb.Error(f"No such file or directory: {path}, with {t} times retry")


def _generator_from_cursor(cursor):
    for k, v in cursor:
        yield k, v


def _do_mrwi_no_shuffle(p: _MapReduceProcess):
    rtn = p.output_info
    with ExitStack() as s:
        dst_txn = p.get_output_transaction(p.partition_id, s)
        cursor = p.get_input_cursor(s)
        v = p.get_mapper()(p.partition_id, _generator_from_cursor(cursor))
        for k1, v1 in v:
            dst_txn.put(k1, v1)
        return rtn


def _do_binary_sorted_map_with_index(p: _BinarySortedMapProcess):
    rtn = p.output_info
    with ExitStack() as s:
        first_cursor = p.get_first_input_cursor(s)
        second_cursor = p.get_second_input_cursor(s)
        dst_txn = p.get_output_transaction(p.partition_id, s)
        output_kv_iter = p.get_func()(
            p.partition_id, _generator_from_cursor(first_cursor), _generator_from_cursor(second_cursor)
        )
        for k_bytes, v_bytes in output_kv_iter:
            dst_txn.put(k_bytes, v_bytes)
        return rtn


def _serialize_shuffle_write_key(iteration_index: int, k_bytes: bytes) -> bytes:
    iteration_bytes = iteration_index.to_bytes(4, "big")  # 4 bytes for the iteration index
    serialized_key = iteration_bytes + k_bytes

    return serialized_key


def _deserialize_shuffle_write_key(serialized_key: bytes) -> (int, int, bytes):
    iteration_bytes = serialized_key[:4]
    k_bytes = serialized_key[4:]
    iteration_index = int.from_bytes(iteration_bytes, "big")
    return iteration_index, k_bytes


def _get_shuffle_partition_id(shuffle_source_partition_id: int, shuffle_destination_partition_id: int) -> str:
    return f"{shuffle_source_partition_id}_{shuffle_destination_partition_id}"


def _do_mrwi_map_and_shuffle_write(p: _MapReduceProcess):
    rtn = p.output_info
    if p.has_partition(p.partition_id):
        with ExitStack() as s:
            cursor = p.get_input_cursor(s)
            shuffle_write_txn_map = {}
            for output_partition_id in range(p.get_output_partition_num()):
                shuffle_partition_id = _get_shuffle_partition_id(p.partition_id, output_partition_id)
                shuffle_write_txn_map[output_partition_id] = p.get_output_transaction(shuffle_partition_id, s)

            output_kv_iter = p.get_mapper()(p.partition_id, _generator_from_cursor(cursor))
            for index, (k_bytes, v_bytes) in enumerate(output_kv_iter):
                shuffle_write_txn_map[p.get_output_partition_id(k_bytes)].put(
                    _serialize_shuffle_write_key(index, k_bytes), v_bytes, overwrite=False
                )
    return rtn


def _do_mrwi_map_and_shuffle_write_unique(p: _MapReduceProcess):
    rtn = p.output_info
    if p.has_partition(p.partition_id):
        with ExitStack() as s:
            cursor = p.get_input_cursor(s)
            shuffle_write_txn_map = {}
            for output_partition_id in range(p.get_output_partition_num()):
                shuffle_partition_id = _get_shuffle_partition_id(p.partition_id, output_partition_id)
                shuffle_write_txn_map[output_partition_id] = p.get_output_transaction(shuffle_partition_id, s)

            output_kv_iter = p.get_mapper()(p.partition_id, _generator_from_cursor(cursor))
            for k_bytes, v_bytes in output_kv_iter:
                shuffle_write_txn_map[p.get_output_partition_id(k_bytes)].put(k_bytes, v_bytes, overwrite=False)
    return rtn


def _do_mrwi_shuffle_read_and_reduce(p: _MapReduceProcess):
    rtn = p.output_info
    reducer = p.get_reducer()
    with ExitStack() as s:
        dst_txn = p.get_output_transaction(p.partition_id, s)
        for input_partition_id in range(p.get_input_partition_num()):
            for k_bytes, v_bytes in p.get_input_cursor(
                s, pid=_get_shuffle_partition_id(input_partition_id, p.partition_id)
            ):
                _, key = _deserialize_shuffle_write_key(k_bytes)
                if (old := dst_txn.get(key)) is None:
                    dst_txn.put(key, v_bytes)
                else:
                    dst_txn.put(key, reducer(old, v_bytes))
    return rtn


def _do_mrwi_shuffle_read_no_reduce(p: _MapReduceProcess):
    rtn = p.output_info
    with ExitStack() as s:
        dst_txn = p.get_output_transaction(p.partition_id, s)
        for input_partition_id in range(p.get_input_partition_num()):
            for k_bytes, v_bytes in p.get_input_cursor(
                s, pid=_get_shuffle_partition_id(input_partition_id, p.partition_id)
            ):
                dst_txn.put(k_bytes, v_bytes)
    return rtn


def _do_reduce(p: _ReduceProcess):
    value = None
    with ExitStack() as s:
        cursor = p.input_cursor(s)
        for _, v_bytes in cursor:
            if value is None:
                value = v_bytes
            else:
                value = p.get_reducer()(value, v_bytes)
    return value


class _FederationMetaManager:
    STATUS_TABLE_NAME_PREFIX = "__federation_status__"
    OBJECT_TABLE_NAME_PREFIX = "__federation_object__"

    def __init__(self, data_dir: str, session_id, party: Tuple[str, str]) -> None:
        self.session_id = session_id
        self.party = party
        self._data_dir = data_dir
        self._env = {}

    def wait_status_set(self, key: bytes) -> bytes:
        value = self.get_status(key)
        while value is None:
            time.sleep(0.001)
            value = self.get_status(key)
        return key

    def get_status(self, key: bytes):
        return self._get(self._get_status_table_name(self.party), key)

    def set_status(self, party: Tuple[str, str], key: bytes, value: bytes):
        return self._set(self._get_status_table_name(party), key, value)

    def ack_status(self, key: bytes):
        return self._ack(self._get_status_table_name(self.party), key)

    def get_object(self, key: bytes):
        return self._get(self._get_object_table_name(self.party), key)

    def set_object(self, party: Tuple[str, str], key: bytes, value: bytes):
        return self._set(self._get_object_table_name(party), key, value)

    def ack_object(self, key: bytes):
        return self._ack(self._get_object_table_name(self.party), key)

    def _get_status_table_name(self, party: Tuple[str, str]):
        return f"{self.STATUS_TABLE_NAME_PREFIX}.{party[0]}_{party[1]}"

    def _get_object_table_name(self, party: Tuple[str, str]):
        return f"{self.OBJECT_TABLE_NAME_PREFIX}.{party[0]}_{party[1]}"

    def _get_env(self, name):
        if name not in self._env:
            self._env[name] = _get_env_with_data_dir(self._data_dir, self.session_id, name, str(0), write=True)
        return self._env[name]

    def _get(self, name: str, key: bytes) -> bytes:
        env = self._get_env(name)
        with env.begin(write=False) as txn:
            return txn.get(key)

    def _set(self, name, key: bytes, value: bytes):
        env = self._get_env(name)
        with env.begin(write=True) as txn:
            return txn.put(key, value)

    def _ack(self, name, key: bytes):
        env = self._get_env(name)
        with env.begin(write=True) as txn:
            txn.delete(key)


def _hash_namespace_name_to_partition(namespace: str, name: str, partitions: int) -> Tuple[bytes, int]:
    k_bytes = f"{name}.{namespace}".encode("utf-8")
    partition_id = int.from_bytes(hashlib.sha256(k_bytes).digest(), "big") % partitions
    return k_bytes, partition_id


class _TableMetaManager:
    namespace = "__META__"
    name = "fragments"
    num_partitions = 11
    _env = {}

    @classmethod
    def _get_or_create_meta_env(cls, data_dir: str, p):
        if p not in cls._env:
            cls._env[p] = _get_env_with_data_dir(data_dir, cls.namespace, cls.name, str(p), write=True)
        return cls._env[p]

    @classmethod
    def _get_meta_env(cls, data_dir: str, namespace: str, name: str):
        k_bytes, p = _hash_namespace_name_to_partition(namespace, name, cls.num_partitions)
        env = cls._get_or_create_meta_env(data_dir, p)
        return k_bytes, env

    @classmethod
    def add_table_meta(
        cls,
        data_dir: str,
        namespace: str,
        name: str,
        num_partitions: int,
        key_serdes_type: int,
        value_serdes_type: int,
        partitioner_type: int,
    ):
        k_bytes, env = cls._get_meta_env(data_dir, namespace, name)
        meta = _TableMeta(num_partitions, key_serdes_type, value_serdes_type, partitioner_type)
        with env.begin(write=True) as txn:
            return txn.put(k_bytes, meta.serialize())

    @classmethod
    def get_table_meta(cls, data_dir: str, namespace: str, name: str) -> "_TableMeta":
        k_bytes, env = cls._get_meta_env(data_dir, namespace, name)
        with env.begin(write=False) as txn:
            old_value_bytes = txn.get(k_bytes)
            if old_value_bytes is not None:
                old_value_bytes = _TableMeta.deserialize(old_value_bytes)
            return old_value_bytes

    @classmethod
    def destroy_table(cls, data_dir: str, namespace: str, name: str):
        k_bytes, env = cls._get_meta_env(data_dir, namespace, name)
        with env.begin(write=True) as txn:
            txn.delete(k_bytes)
        path = Path(data_dir).joinpath(namespace, name)
        shutil.rmtree(path, ignore_errors=True)


class _TableMeta:
    def __init__(self, num_partitions: int, key_serdes_type: int, value_serdes_type: int, partitioner_type: int):
        self.num_partitions = num_partitions
        self.key_serdes_type = key_serdes_type
        self.value_serdes_type = value_serdes_type
        self.partitioner_type = partitioner_type

    def serialize(self) -> bytes:
        num_partitions_bytes = self.num_partitions.to_bytes(4, "big")
        key_serdes_type_bytes = self.key_serdes_type.to_bytes(4, "big")
        value_serdes_type_bytes = self.value_serdes_type.to_bytes(4, "big")
        partitioner_type_bytes = self.partitioner_type.to_bytes(4, "big")
        return num_partitions_bytes + key_serdes_type_bytes + value_serdes_type_bytes + partitioner_type_bytes

    @classmethod
    def deserialize(cls, serialized_bytes: bytes) -> "_TableMeta":
        num_partitions = int.from_bytes(serialized_bytes[:4], "big")
        key_serdes_type = int.from_bytes(serialized_bytes[4:8], "big")
        value_serdes_type = int.from_bytes(serialized_bytes[8:12], "big")
        partitioner_type = int.from_bytes(serialized_bytes[12:16], "big")
        return cls(num_partitions, key_serdes_type, value_serdes_type, partitioner_type)


def _serialize_tuple_of_str(name: str, namespace: str):
    name_bytes = name.encode("utf-8")
    namespace_bytes = namespace.encode("utf-8")
    split_index_bytes = len(name_bytes).to_bytes(4, "big")
    return split_index_bytes + name_bytes + namespace_bytes


def _deserialize_tuple_of_str(serialized_bytes: bytes):
    split_index = int.from_bytes(serialized_bytes[:4], "big")
    name = serialized_bytes[4 : 4 + split_index].decode("utf-8")
    namespace = serialized_bytes[4 + split_index :].decode("utf-8")
    return name, namespace
