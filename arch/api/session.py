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

import os
import typing
import uuid
from typing import Iterable

from arch.api import RuntimeInstance, _EGGROLL_VERSION
from arch.api import WorkMode, Backend
from arch.api.base.table import Table
from arch.api.base.utils.store_type import StoreTypes
from arch.api.utils import file_utils
from arch.api.utils.log_utils import LoggerFactory
from arch.api.utils.profile_util import log_elapsed


# noinspection PyProtectedMember
def init(job_id=None,
         mode: typing.Union[int, WorkMode] = WorkMode.STANDALONE,
         backend: typing.Union[int, Backend] = Backend.EGGROLL,
         persistent_engine: str = StoreTypes.ROLLPAIR_LMDB,
         eggroll_version=None,
         set_log_dir=True):
    if RuntimeInstance.SESSION:
        return

    if isinstance(mode, int):
        mode = WorkMode(mode)
    if isinstance(backend, int):
        backend = Backend(backend)
    if job_id is None:
        job_id = str(uuid.uuid1())
        if True:
            LoggerFactory.set_directory()
    else:
        if set_log_dir:
            LoggerFactory.set_directory(os.path.join(file_utils.get_project_base_directory(), 'logs', job_id))
    if eggroll_version is None:
        eggroll_version = _EGGROLL_VERSION

    if backend.is_eggroll():
        if eggroll_version < 2:
            from arch.api.impl.based_1x import build
            builder = build.Builder(session_id=job_id, work_mode=mode, persistent_engine=persistent_engine)

        else:
            from arch.api.impl.based_2x import build
            builder = build.Builder(session_id=job_id, work_mode=mode, persistent_engine=persistent_engine)

    elif backend.is_spark():
        if eggroll_version < 2:
            from arch.api.impl.based_spark.based_1x import build
            builder = build.Builder(session_id=job_id, work_mode=mode, persistent_engine=persistent_engine)
        else:
            from arch.api.impl.based_spark.based_2x import build
            builder = build.Builder(session_id=job_id, work_mode=mode, persistent_engine=persistent_engine)

    else:
        raise ValueError(f"backend: ${backend} unknown")

    RuntimeInstance.MODE = mode
    RuntimeInstance.BACKEND = backend
    RuntimeInstance.BUILDER = builder
    RuntimeInstance.SESSION = builder.build_session()


@log_elapsed
def table(name, namespace=None, partition=1, persistent=True, create_if_missing=True, error_if_exist=False,
          in_place_computing=False, **kwargs) -> Table:
    namespace = namespace or get_session_id()
    return RuntimeInstance.SESSION.table(name=name,
                                         namespace=namespace,
                                         partition=partition,
                                         persistent=persistent,
                                         in_place_computing=in_place_computing,
                                         create_if_missing=create_if_missing,
                                         error_if_exist=error_if_exist,
                                         **kwargs)


@log_elapsed
def parallelize(data: Iterable, include_key=False, name=None, partition=1, namespace=None, persistent=False,
                create_if_missing=True, error_if_exist=False, chunk_size=100000, in_place_computing=False) -> Table:
    return RuntimeInstance.SESSION.parallelize(data=data, include_key=include_key, name=name, partition=partition,
                                               namespace=namespace,
                                               persistent=persistent,
                                               chunk_size=chunk_size,
                                               in_place_computing=in_place_computing,
                                               create_if_missing=create_if_missing,
                                               error_if_exist=error_if_exist)


def cleanup(name, namespace, persistent=False):
    return RuntimeInstance.SESSION.cleanup(name=name, namespace=namespace, persistent=persistent)


# noinspection PyPep8Naming
def generateUniqueId():
    return RuntimeInstance.SESSION.generateUniqueId()


def get_session_id():
    return RuntimeInstance.SESSION.get_session_id()


def get_data_table(name, namespace):
    """
    return data table instance by table name and table name space
    :param name: table name of data table
    :param namespace: table name space of data table
    :return:
        data table instance
    """
    return RuntimeInstance.SESSION.get_data_table(name=name, namespace=namespace)


def save_data_table_meta(kv, data_table_name, data_table_namespace):
    """
    save data table meta information
    :param kv: v should be serialized by JSON
    :param data_table_name: table name of this data table
    :param data_table_namespace: table name of this data table
    :return:
    """
    return RuntimeInstance.SESSION.save_data_table_meta(kv=kv,
                                                        data_table_name=data_table_name,
                                                        data_table_namespace=data_table_namespace)


def get_data_table_meta(key, data_table_name, data_table_namespace):
    """
    get data table meta information
    :param key:
    :param data_table_name: table name of this data table
    :param data_table_namespace: table name of this data table
    :return:
    """
    return RuntimeInstance.SESSION.get_data_table_meta(key=key,
                                                       data_table_name=data_table_name,
                                                       data_table_namespace=data_table_namespace)


def get_data_table_metas(data_table_name, data_table_namespace):
    """
    get data table meta information
    :param data_table_name: table name of this data table
    :param data_table_namespace: table name of this data table
    :return:
    """
    return RuntimeInstance.SESSION.get_data_table_metas(data_table_name=data_table_name,
                                                        data_table_namespace=data_table_namespace)


def clean_tables(namespace, regex_string='*'):
    RuntimeInstance.SESSION.clean_table(namespace=namespace, regex_string=regex_string)


def save_data(kv_data: Iterable,
              name,
              namespace,
              partition=1,
              persistent: bool = True,
              create_if_missing=True,
              error_if_exist=False,
              in_version: bool = False,
              version_log=None):
    """
    save data into data table
    :param kv_data:
    :param name: table name of data table
    :param namespace: table namespace of data table
    :param partition: number of partition
    :param persistent
    :param create_if_missing:
    :param error_if_exist:
    :param in_version:
    :param version_log
    :return:
        data table instance
    """
    return RuntimeInstance.SESSION.save_data(kv_data=kv_data,
                                             name=name,
                                             namespace=namespace,
                                             partition=partition,
                                             persistent=persistent,
                                             create_if_missing=create_if_missing,
                                             error_if_exist=error_if_exist,
                                             in_version=in_version,
                                             version_log=version_log)


def stop():
    RuntimeInstance.SESSION.stop()


def kill():
    RuntimeInstance.SESSION.kill()
