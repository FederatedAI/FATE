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

import uuid
from arch.api.proto.model_meta_pb2 import ModelMeta
from arch.api.proto.model_param_pb2 import ModelParam
from arch.api import eggroll


def save_model(name, name_space, buffer_type, proto_buffer, model_id=None):
    model_id = str(uuid.uuid1().hex) if not model_id else model_id
    data_table = eggroll.table("%s_%s" % (name, model_id), name_space, partition=1, create_if_missing=True, error_if_exist=False)
    data_table.put(buffer_type, proto_buffer.SerializeToString(), use_pickle=False)
    return model_id


def read_model(name, name_space, buffer_type, proto_buffer, model_id):
    data_table = eggroll.table("%s_%s" % (name, model_id), name_space, partition=1, create_if_missing=True, error_if_exist=False)
    proto_buffer.ParseFromString(data_table.get(buffer_type, use_pickle=False))


def save_data_transform(name, proto_buffer, model_id=None):
    model_id = str(uuid.uuid1().hex) if not model_id else model_id
    data_table = eggroll.table("%s_%s" % (name, model_id), "preprocessing", partition=1, create_if_missing=True, error_if_exist=False)
    data_table.put("transform", proto_buffer.SerializeToString(), use_pickle=False)
    return model_id


def read_data_transform(name, proto_buffer, model_id):
    model_id = str(uuid.uuid1().hex) if not model_id else model_id
    data_table = eggroll.table("%s_%s" % (name, model_id), "preprocessing", partition=1, create_if_missing=True, error_if_exist=False)
    proto_buffer.ParseFromString(data_table.get("transform", use_pickle=False))


if __name__ == '__main__':
    job_id = str(uuid.uuid1().hex)
    print(job_id)
    eggroll.init()

    # guest
    model_meta1 = ModelMeta()
    model_meta1.name = "HeteroLRGuest"
    save_model("HeteroLRGuest", "HeteroLR", "meta", model_meta1, job_id)

    model_param1 = ModelParam()
    model_param1.weight['k1'] = 1
    model_param1.weight['k2'] = 2
    model_param1.weight['k3'] = 3
    model_param1.intercept = 0.1
    save_model("HeteroLRGuest", "HeteroLR", "param", model_param1, job_id)

    # host
    model_meta1.name = "HeteroLRHost"
    save_model("HeteroLRHost", "HeteroLR", "meta", model_meta1, job_id)

    model_param11 = ModelParam()
    model_param11.weight['k4'] = 4
    model_param11.weight['k5'] = 5
    model_param11.intercept = 0.2
    save_model("HeteroLRHost", "HeteroLR", "param", model_param11, job_id)

    # read
    model_meta2 = ModelMeta()
    read_model("HeteroLRGuest", "HeteroLR", "meta", model_meta2, job_id)
    print(model_meta2)

    model_meta22 = ModelMeta()
    read_model("HeteroLRHost", "HeteroLR", "meta", model_meta22, job_id)
    print(model_meta22)

    model_param2 = ModelParam()
    read_model("HeteroLRGuest", "HeteroLR", "param", model_param2, job_id)
    print(model_param2)

    model_param22 = ModelParam()
    read_model("HeteroLRHost", "HeteroLR", "param", model_param22, job_id)
    print(model_param22)

    # data transform
    data_transform1 = ModelMeta()
    data_transform1.name = "xxxxx"
    save_data_transform("data_transform_guest", data_transform1, job_id)

    data_transform2 = ModelMeta()
    read_data_transform("data_transform_guest", data_transform2, job_id)
    print(data_transform2)
