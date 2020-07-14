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
import pickle
from typing import Iterable

import six


# noinspection PyPep8Naming
from arch.api.utils.core_utils import current_timestamp
from fate_flow.db.db_models import DB, MachineLearningDataSchema


@six.add_metaclass(abc.ABCMeta)
class Table(object):
    """
    table for distributed storage
    """

    @abc.abstractmethod
    def get_partitions(self):
        pass

    @abc.abstractmethod
    def get_storage_engine(self):
        pass

    @abc.abstractmethod
    def get_address(self):
        pass

    @abc.abstractmethod
    def get_namespace(self):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def put_all(self, kv_list: Iterable, use_serialize=True, chunk_size=100000):
        """
        Puts (key, value) 2-tuple stream from the iterable items.

        Elements must be exact 2-tuples, they may not be of any other type, or tuple subclass.
        Parameters
        ----------
        kv_list : Iterable
          Key-Value 2-tuple iterable. Will be serialized.
        Notes
        -----
        Each key must be less than 512 bytes, value must be less than 32 MB(implementation depends).
        """
        pass

    @abc.abstractmethod
    def collect(self, min_chunk_size=0, use_serialize=True) -> list:
        """
        Returns an iterator of (key, value) 2-tuple from the Table.

        Parameters
        ---------
        min_chunk_size : int
          Minimum chunk size (key bytes + value bytes) returned if end of table is not hit.
          0 indicates a default chunk size (partition_num * 1.75 MB)
          negative number indicates no chunk limit, i.e. returning all records.
          Default chunk size is recommended if there is no special needs from user.

        Returns
        -------
        Iterator
        """
        pass

    @abc.abstractmethod
    def delete(self, k, use_serialize=True):
        """
        Returns the deleted value corresponding to the key.

        Parameters
        ----------
          k : object
            Key object. Will be serialized. Must be less than 512 bytes.
        Returns
        -------
        object
          Corresponding value of the deleted key. Returns None if key does not exist.
        """
        pass

    @abc.abstractmethod
    def destroy(self):
        """
        Destroys this Table, freeing its associated storage resources.

        Returns
        -------
        None

        """
        pass

    @abc.abstractmethod
    def count(self):
        """
        Returns the number of elements in the Table.

        Returns
        -------
        int
          Number of elements in this Table.
        """
        pass

    @abc.abstractmethod
    def save_as(self, name, namespace, partition=None, use_serialize=True, **kwargs):
        """
        Transforms a temporary table to a persistent table.

        Parameters
        ----------
        name : string
          Table name of result Table.
        namespace: string
          Table namespace of result Table.
        partition : int
          Number of partition for the new persistent table.
        use_serialize

        Returns
        -------
        Table
           Result persistent Table.
        """
        pass

    @abc.abstractmethod
    def close(self):
        pass

    """
    meta utils
    """

    def get_schema(self):
        with DB.connection_context():
            schema = MachineLearningDataSchema.select().where(MachineLearningDataSchema.f_table_name == self.name,
                                                              MachineLearningDataSchema.f_namespace == self.namespace)
            schema_data = {}
            if schema:
                schema = schema[0]
                schema_data = pickle.loads(schema.f_content)
        return schema_data

    def save_schema(self, kv):
        # save metas to mysql
        with DB.connection_context():
            schema = MachineLearningDataSchema.select().where(MachineLearningDataSchema.f_table_name == self.name,
                                                            MachineLearningDataSchema.f_namespace == self.namespace)
            is_insert = True
            if schema:
                schema = schema[0]
                is_insert = False
                schema_data = pickle.loads(schema.f_content)
                schema_data.updata(kv)
                schema.f_content = pickle.dumps(schema_data)

            else:
                schema_data = kv
                schema = MachineLearningDataSchema()
                schema.f_create_time = current_timestamp()
                schema.f_table_name = self.name
                schema.f_namespace = self.namespace
                schema.f_content = pickle.dumps(schema_data)
            schema.f_update_time = current_timestamp()
            if is_insert:
                schema.save(force_insert=True)
            else:
                schema.save()
        return schema_data

