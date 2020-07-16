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

import numpy

from arch.api.utils.core_utils import current_timestamp
from fate_flow.db.db_models import DB, MachineLearningDataSchema
from federatedml.feature.sparse_vector import SparseVector


def dataset_to_list(src):
    if isinstance(src, numpy.ndarray):
        return src.tolist()
    elif isinstance(src, list):
        return src
    elif isinstance(src, SparseVector):
        vector = [0] * src.get_shape()
        for idx, v in src.get_all_data():
            vector[idx] = v
        return vector
    else:
        return [src]


def create(name, namespace, store_engine, address: dict, partitions=1):
    with DB.connection_context():
        schema = MachineLearningDataSchema.select().where(MachineLearningDataSchema.f_table_name == name,
                                                          MachineLearningDataSchema.f_namespace == namespace)
        is_insert = True
        if schema:
            schema = schema[0]
            is_insert = False
            if schema.f_data_store_engine:
                # schema.f_data_store_engine = ',{}'.format(store_engine)
                raise Exception('table has been created as {} store engine'.format(schema.f_data_store_engine))
            else:

                schema.f_data_store_engine = store_engine
                schema.f_address = json.dumps(address)
        else:
            schema = MachineLearningDataSchema()
            schema.f_create_time = current_timestamp()
            schema.f_table_name = name
            schema.f_namespace = namespace
            schema.f_data_store_engine = store_engine
            schema.f_address = json.dumps(address)
            schema.f_partitions = partitions
        schema.f_update_time = current_timestamp()
        if is_insert:
            schema.save(force_insert=True)
        else:
            schema.save()


def get_store_info(name, namespace):
    with DB.connection_context():
        schema = MachineLearningDataSchema.select().where(MachineLearningDataSchema.f_table_name == name,
                                                          MachineLearningDataSchema.f_namespace == namespace)
        if schema:
            schema = schema[0]
            store_info = schema.f_data_store_engine
            address = schema.f_address
            partitions = schema.f_partitions
        else:
            return None, None
    return store_info, address, partitions

