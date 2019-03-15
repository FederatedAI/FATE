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
import os
from arch.api.common.mlmodel.ModelBuffer import ModelBuffer
from arch.api.common.mlmodel.DataTransformBuffer import DataTransformBuffer
from arch.api.utils import file_utils

data_dir = os.path.join(file_utils.get_project_base_directory(), 'data')


def save_model(model_buffer, model_id=None):
    model_id = str(uuid.uuid1().hex) if not model_id else model_id
    check_storage()
    meta_stream, param_stream = model_buffer.serialize()
    with open(os.path.join(data_dir, "%s.meta" % (model_id)), 'wb') as fw:
        fw.write(meta_stream)
    with open(os.path.join(data_dir, "%s.param" % (model_id)), 'wb') as fw:
        fw.write(param_stream)
    return model_id


def read_model(modelId):
    with open(os.path.join(data_dir, "%s.meta" % (modelId)), 'rb') as fr:
        meta_stream = fr.read()
    with open(os.path.join(data_dir, "%s.param" % (modelId)), 'rb') as fr:
        param_stream = fr.read()
    return meta_stream, param_stream

def save_data_transform(data_transform_buffer, model_id=None):
    model_id = str(uuid.uuid1().hex) if not model_id else model_id
    check_storage()
    stream = data_transform_buffer.serialize()
    with open(os.path.join(data_dir, "%s.transform" % (model_id)), 'wb') as fw:
        fw.write(stream)
    return modelId

def read_data_transform(model_id):
    model_id = str(uuid.uuid1().hex) if not model_id else model_id
    with open(os.path.join(data_dir, "%s.transform" % (model_id)), 'rb') as fr:
        stream = fr.read()
    return stream


def check_storage():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

if __name__ == '__main__':
    modelBuffer1 = ModelBuffer()
    modelBuffer1.set_meta_field("name", "test")
    modelId = save_model(modelBuffer1)
    print(modelId)

    meta_stream, param_stream = read_model(modelId)
    modelBuffer2 = ModelBuffer()
    print(modelBuffer2.meta)
    modelBuffer2.deserialize(meta_stream, param_stream)
    print(modelBuffer2.meta)

    data_transform_buffer1 = DataTransformBuffer()
    data_transform_buffer1.set_transform_fields({"name": "xxxx"})
    modelId = save_data_transform(data_transform_buffer1, modelId)
    data_transform_stream = read_data_transform(modelId)
    data_transform_buffer2 = DataTransformBuffer()
    data_transform_buffer2.deserialize(data_transform_stream)
    print(data_transform_buffer2.data_transform)
