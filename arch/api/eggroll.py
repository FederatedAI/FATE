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
import uuid
from typing import Iterable

from arch.api import RuntimeInstance
from arch.api import WorkMode, NamingPolicy, Backend
from eggroll.api.core import EggrollSession
from arch.api.table.table import Table
from arch.api.utils import file_utils
from arch.api.utils.log_utils import LoggerFactory
from arch.api.utils.profile_util import log_elapsed
import typing
import warnings

warnings.warn("eggroll is deprecated, use table_manager instead", DeprecationWarning, stacklevel=2)


# noinspection PyProtectedMember
def init(job_id=None,
         mode: typing.Union[int, WorkMode] = WorkMode.STANDALONE,
         naming_policy: NamingPolicy = NamingPolicy.DEFAULT,
         backend: typing.Union[int, Backend] = Backend.EGGROLL):

    if isinstance(mode, int):
        mode = WorkMode(mode)
    if isinstance(backend, int):
        backend = Backend(backend)
    if RuntimeInstance.TABLE_MANAGER:
        return
    if job_id is None:
        job_id = str(uuid.uuid1())
        LoggerFactory.set_directory()
    else:
        LoggerFactory.set_directory(os.path.join(file_utils.get_project_base_directory(), 'logs', job_id))
        
    RuntimeInstance.MODE = mode
    RuntimeInstance.Backend = backend
    eggroll_session = EggrollSession(session_id=job_id, naming_policy=naming_policy)

    if backend.is_eggroll():
        if mode.is_standalone():
            from arch.api.table.eggroll.standalone.table_manager import DTableManager

            RuntimeInstance.TABLE_MANAGER = DTableManager(eggroll_session)
        elif mode.is_cluster():
            from arch.api.table.eggroll.cluster.table_manager import DTableManager
            RuntimeInstance.TABLE_MANAGER = DTableManager(eggroll_session)
        else:
            raise NotImplemented(f"Backend {backend} with WorkMode {mode} is not supported")

    elif backend.is_spark():
        if mode.is_standalone():
            from arch.api.table.pyspark.standalone.table_manager import RDDTableManager
            rdd_manager = RDDTableManager(eggroll_session)
            RuntimeInstance.TABLE_MANAGER = rdd_manager
        elif mode.is_cluster():
            from arch.api.table.pyspark.cluster.table_manager import RDDTableManager
            rdd_manager = RDDTableManager(eggroll_session)
            RuntimeInstance.TABLE_MANAGER = rdd_manager
        else:
            raise NotImplemented(f"Backend {backend} with WorkMode {mode} is not supported")

    table("__federation__", job_id, partition=10)


@log_elapsed
def table(name, namespace, partition=1, persistent=True, create_if_missing=True, error_if_exist=False,
          in_place_computing=False) -> Table:
    return RuntimeInstance.TABLE_MANAGER.table(name=name,
                                               namespace=namespace,
                                               partition=partition,
                                               persistent=persistent,
                                               in_place_computing=in_place_computing,
                                               create_if_missing=create_if_missing,
                                               error_if_exist=error_if_exist)


@log_elapsed
def parallelize(data: Iterable, include_key=False, name=None, partition=1, namespace=None, persistent=False,
                create_if_missing=True, error_if_exist=False, chunk_size=100000, in_place_computing=False) -> Table:
    return RuntimeInstance.TABLE_MANAGER.parallelize(data=data, include_key=include_key, name=name, partition=partition,
                                                     namespace=namespace,
                                                     persistent=persistent,
                                                     chunk_size=chunk_size,
                                                     in_place_computing=in_place_computing,
                                                     create_if_missing=create_if_missing,
                                                     error_if_exist=error_if_exist)


def cleanup(name, namespace, persistent=False):
    return RuntimeInstance.TABLE_MANAGER.cleanup(name=name, namespace=namespace, persistent=persistent)


# noinspection PyPep8Naming
def generateUniqueId():
    return RuntimeInstance.TABLE_MANAGER.generateUniqueId()


def get_job_id():
    return RuntimeInstance.TABLE_MANAGER.job_id


def get_data_table(name, namespace):
    """
    return data table instance by table name and table name space
    :param name: table name of data table
    :param namespace: table name space of data table
    :return:
        data table instance
    """
    return RuntimeInstance.TABLE_MANAGER.get_data_table(name=name, namespace=namespace)


def save_data_table_meta(kv, data_table_name, data_table_namespace):
    """
    save data table meta information
    :param kv: v should be serialized by JSON
    :param data_table_name: table name of this data table
    :param data_table_namespace: table name of this data table
    :return:
    """
    return RuntimeInstance.TABLE_MANAGER.save_data_table_meta(kv=kv,
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
    return RuntimeInstance.TABLE_MANAGER.get_data_table_meta(key=key,
                                                             data_table_name=data_table_name,
                                                             data_table_namespace=data_table_namespace)


def get_data_table_metas(data_table_name, data_table_namespace):
    """
    get data table meta information
    :param data_table_name: table name of this data table
    :param data_table_namespace: table name of this data table
    :return:
    """
    return RuntimeInstance.TABLE_MANAGER.get_data_table_metas(data_table_name=data_table_name,
                                                              data_table_namespace=data_table_namespace)


def clean_tables(namespace, regex_string='*'):
    RuntimeInstance.TABLE_MANAGER.clean_table(namespace=namespace, regex_string=regex_string)


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
    return RuntimeInstance.TABLE_MANAGER.save_data(kv_data=kv_data,
                                                   name=name,
                                                   namespace=namespace,
                                                   partition=partition,
                                                   persistent=persistent,
                                                   create_if_missing=create_if_missing,
                                                   error_if_exist=error_if_exist,
                                                   in_version=in_version,
                                                   version_log=version_log)
