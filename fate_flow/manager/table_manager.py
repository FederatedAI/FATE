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
import json

from arch.api.utils.core_utils import current_timestamp, serialize_b64, deserialize_b64
from fate_arch.data_table.base import EggRollAddress, HDFSAddress
from fate_arch.data_table.store_type import Relationship
from fate_arch.session import Backend
from fate_flow.db.db_models import DB, MachineLearningDataSchema
from fate_flow.utils import data_utils


def create(name, namespace, store_engine, address=None, partitions=1, count=0):
    with DB.connection_context():
        schema = MachineLearningDataSchema.select().where(MachineLearningDataSchema.f_table_name == name,
                                                          MachineLearningDataSchema.f_namespace == namespace)
        is_insert = True
        if schema:
            if store_engine != schema.f_data_store_engine:
                raise Exception('table {} {} has been created by store engine {} '.format(name, namespace, schema.f_data_store_engine))
            else:
                return
        else:
            schema = MachineLearningDataSchema()
            schema.f_create_time = current_timestamp()
            schema.f_table_name = name
            schema.f_namespace = namespace
            schema.f_data_store_engine = store_engine
            if not address:
                if store_engine in Relationship.CompToStore.get(Backend.EGGROLL):
                    address = EggRollAddress(name=name, namespace=namespace, storage_type=store_engine)
                elif store_engine in Relationship.CompToStore.get(Backend.SPARK):
                    address = HDFSAddress(path=data_utils.generate_hdfs_address())
            schema.f_address = serialize_b64(address, to_str=True)
            schema.f_partitions = partitions
            schema.f_count = count
            schema.f_schema = serialize_b64({}, to_str=True)
            schema.f_part_of_data = serialize_b64([], to_str=True)
        schema.f_update_time = current_timestamp()
        if is_insert:
            schema.save(force_insert=True)
        else:
            schema.save()
        return address


def get_store_info(name, namespace):
    with DB.connection_context():
        schema = MachineLearningDataSchema.select().where(MachineLearningDataSchema.f_table_name == name,
                                                          MachineLearningDataSchema.f_namespace == namespace)
        if schema:
            schema = schema[0]
            store_info = schema.f_data_store_engine
            address = deserialize_b64(schema.f_address)
            partitions = schema.f_partitions
        else:
            return None, None, None
    return store_info, address, partitions