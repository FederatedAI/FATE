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
from fate_arch.common import base_utils
import numpy

from fate_arch import storage
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


def get_header_schema(header_line, id_delimiter):
    header_source_item = header_line.split(id_delimiter)
    return {'header': id_delimiter.join(header_source_item[1:]).strip(), 'sid': header_source_item[0].strip()}


def list_to_str(input_list, id_delimiter):
    return id_delimiter.join(list(map(str, input_list)))


def default_output_table_info(task_id, task_version):
    return f"output_data_{task_id}_{task_version}", base_utils.fate_uuid()


def default_output_path(name, namespace):
    return f"/fate/output_data/{namespace}/{name}"


def default_input_path(name, namespace):
    return f"/fate/input_data/{namespace}/{name}"


def get_input_data_min_partitions(input_data, role, party_id):
    min_partition = None
    if role != 'arbiter':
        for data_type, data_location in input_data[role][party_id].items():

            table_info = {'name': data_location.split('.')[1], 'namespace': data_location.split('.')[0]}
            table_meta = storage.StorageTableMeta(name=table_info['name'], namespace=table_info['namespace'])
            if table_meta:
                table_partition = table_meta.get_partitions()
                if not min_partition or min_partition > table_partition:
                    min_partition = table_partition
    return min_partition
