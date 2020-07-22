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

import typing
from typing import Iterable

from fate_arch import session
from fate_arch.common.log import getLogger
from fate_arch.session import WorkMode, Backend, TableABC

LOGGER = getLogger()


def init(job_id=None,
         mode: typing.Union[int, WorkMode] = WorkMode.STANDALONE,
         backend: typing.Union[int, Backend] = Backend.EGGROLL,
         options: dict = None,
         **kwargs):
    """
    Initializes session, should be called before all.

    Parameters
    ---------
    job_id : string
      job id and default table namespace of this runtime.
    mode : WorkMode
      set work mode,

        - standalone: `WorkMode.STANDALONE` or 0
        - cluster: `WorkMode.CLUSTER` or 1
    backend : Backend
      set computing backend,
        
        - eggroll: `Backend.EGGROLL` or 0
        - spark: `Backend.SAPRK` or 1
    options : None or dict
      additional options

    Returns
    -------
    None
      nothing returns

    Examples
    --------
    >>> from arch.api import session
    >>> session.init("a_job_id", WorkMode.Standalone, Backend.EGGROLL)
    """
    if kwargs:
        LOGGER.warning(f"{kwargs} not used, check!")

    if session.has_default():
        return session.default()
    return session.init(job_id, mode, backend, options)


def table(name, namespace, **kwargs) -> TableABC:
    """
    Loads an existing Table.

    Parameters
    ---------
    name : string
      Table name of result Table.
    namespace : string
      Table namespace of result Table.

    Returns
    -------
    Table
      A Table consisting data loaded.

    Examples
    --------
    >>> from arch.api import session
    >>> a = session.table('foo', 'bar')
    """
    return session.default().load(name=name, namespace=namespace, **kwargs)


def parallelize(data: Iterable, partition, include_key=False, **kwargs) -> TableABC:
    """
    Transforms an existing iterable data into a Table.

    Parameters
    ---------
    data : Iterable
      Data to be put.
    include_key : boolean
      Whether to include key when parallelizing data into table.
    partition : int
      Number of partitions when parallelizing data.

    Returns
    -------
    Table
      A Table consisting of parallelized data.

    Examples
    --------
    >>> from arch.api import session
    >>> table = session.parallelize(range(10), 2)
    """
    return session.default().parallelize(data=data, partition=partition, include_key=include_key, **kwargs)


def cleanup(name, namespace, *args, **kwargs):
    """
    Destroys Table(s). Wildcard can be used in `name` parameter.

    Parameters
    ---------
    name : string
      Table name to be cleanup. Wildcard can be used here.
    namespace : string
      Table namespace to be cleanup. This needs to be a exact match.

    Returns
    -------
    None

    Examples
    --------
    >>> from arch.api import session
    >>> session.cleanup('foo*', 'bar', persistent=True)
    """
    if len(args) > 0 or len(kwargs) > 0:
        LOGGER.warning(f"some args removed, please check! {args}, {kwargs}")
    return session.default().cleanup(name=name, namespace=namespace)


def get_session_id():
    """
    Returns session id.

    Returns
    -------
    string
      session id

    Examples
    --------
    >>> from arch.api import session
    >>> session.get_session_id()
    """
    return session.default().session_id


def get_data_table(name, namespace):
    """
    return data table instance by table name and table name space

    Parameters
    ---------
    name : string
      table name of data table
    namespace : string
      table name space of data table
    
    returns
    -------
    DTable
      data table instance

    Examples
    --------
    >>> from arch.api import session
    >>> session.get_data_table(name, namespace)
    """
    LOGGER.warning(f"don't use this, use table directly")
    return session.default().load(name=name, namespace=namespace)


def clean_tables(namespace, regex_string='*'):
    session.default().cleanup(namespace=namespace, name=regex_string)


def stop():
    """
    Stops session, clean all tables associated with this session.

    Examples
    --------
    >>> from arch.api import session
    >>> session.stop()
    """
    session.default().stop()


def kill():
    session.default().kill()


def exit():
    session.exit_session()
