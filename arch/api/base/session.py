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
import abc
import datetime
import threading
from typing import Iterable

import six

from arch.api.base.table import Table


@six.add_metaclass(abc.ABCMeta)
class FateSession(object):
    _instance: 'FateSession' = None
    __lock = threading.Lock()

    @staticmethod
    def set_instance(instance):
        if not FateSession._instance:
            with FateSession.__lock:
                if not FateSession._instance:
                    FateSession._instance = instance

    @staticmethod
    def get_instance():
        return FateSession._instance

    @abc.abstractmethod
    def get_persistent_engine(self):
        pass

    @abc.abstractmethod
    def table(self,
              name,
              namespace,
              partition,
              persistent,
              in_place_computing,
              create_if_missing,
              error_if_exist,
              **kwargs) -> Table:
        pass

    @abc.abstractmethod
    def parallelize(self,
                    data: Iterable,
                    include_key,
                    name,
                    partition,
                    namespace,
                    persistent,
                    chunk_size,
                    in_place_computing,
                    create_if_missing,
                    error_if_exist) -> Table:
        pass

    @abc.abstractmethod
    def cleanup(self, name, namespace, persistent):
        pass

    # noinspection PyPep8Naming
    @abc.abstractmethod
    def generateUniqueId(self):
        pass

    @abc.abstractmethod
    def get_session_id(self):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def kill(self):
        pass

    @staticmethod
    def get_data_table(name, namespace):
        """
        return data table instance by table name and table name space
        :param name: table name of data table
        :param namespace: table name space of data table
        :return:
            data table instance
        """
        return FateSession.get_instance().table(name=name,
                                                namespace=namespace,
                                                create_if_missing=False,
                                                persistent=True,
                                                error_if_exist=False,
                                                in_place_computing=False,
                                                partition=1)

    @staticmethod
    def save_data_table_meta(kv, data_table_name, data_table_namespace):
        """
        save data table meta information
        :param kv: v should be serialized by JSON
        :param data_table_name: table name of this data table
        :param data_table_namespace: table name of this data table
        :return:
        """
        from arch.api.utils.core import json_dumps
        data_meta_table = FateSession.get_instance().table(name="%s.meta" % data_table_name,
                                                           namespace=data_table_namespace,
                                                           partition=1,
                                                           create_if_missing=True,
                                                           error_if_exist=False,
                                                           persistent=True,
                                                           in_place_computing=False)
        for k, v in kv.items():
            data_meta_table.put(k, json_dumps(v), use_serialize=False)

    @staticmethod
    def get_data_table_meta(key, data_table_name, data_table_namespace):
        """
        get data table meta information
        :param key:
        :param data_table_name: table name of this data table
        :param data_table_namespace: table name of this data table
        :return:
        """
        from arch.api.utils.core import json_loads
        data_meta_table = FateSession.get_instance().table(name="%s.meta" % data_table_name,
                                                           namespace=data_table_namespace,
                                                           create_if_missing=True,
                                                           error_if_exist=False,
                                                           in_place_computing=False,
                                                           persistent=True,
                                                           partition=1)
        if data_meta_table:
            value_bytes = data_meta_table.get(key, use_serialize=False)
            if value_bytes:
                return json_loads(value_bytes)
            else:
                return None
        else:
            return None

    @staticmethod
    def get_data_table_metas(data_table_name, data_table_namespace):
        """
        get data table meta information
        :param data_table_name: table name of this data table
        :param data_table_namespace: table name of this data table
        :return:
        """
        from arch.api.utils.core import json_loads
        data_meta_table = FateSession.get_instance().table(name="%s.meta" % data_table_name,
                                                           namespace=data_table_namespace,
                                                           partition=1,
                                                           persistent=True,
                                                           in_place_computing=False,
                                                           create_if_missing=True,
                                                           error_if_exist=False)
        if data_meta_table:
            metas = dict()
            for k, v in data_meta_table.collect(use_serialize=False):
                metas[k] = json_loads(v)
            return metas
        else:
            return None

    @staticmethod
    def clean_table(namespace, regex_string='*'):
        try:
            FateSession.get_instance().cleanup(name=regex_string, namespace=namespace, persistent=False)
        except Exception as e:
            print(e)

    @staticmethod
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
        :param version_log:
        :param in_version:
        :param kv_data:
        :param name: table name of data table
        :param namespace: table namespace of data table
        :param partition: number of partition
        :param persistent: bool = True,
        :param create_if_missing:
        :param error_if_exist:
        :return:
            data table instance
        """
        from arch.api.utils import version_control
        data_table = FateSession.get_instance().table(name=name,
                                                      namespace=namespace,
                                                      partition=partition,
                                                      persistent=persistent,
                                                      in_place_computing=False,
                                                      create_if_missing=create_if_missing,
                                                      error_if_exist=error_if_exist)
        data_table.put_all(kv_data)
        if in_version:
            version_log = "[AUTO] save data at %s." % datetime.datetime.now() if not version_log else version_log
            version_control.save_version(name=name, namespace=namespace, version_log=version_log)
        return data_table
