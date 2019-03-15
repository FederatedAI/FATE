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

from arch.api.proto.model_meta_pb2 import ModelMeta
from arch.api.common.mlmodel.ProtoBuffer import ProtoBuffer


class DataTransformBuffer(ProtoBuffer):

    def __init__(self):
        self.data_transform = ModelMeta()

    def set_transform_field(self, name, value):
        self.set_field(self.data_transform, name, value)

    def set_transform_fields(self, fields):
        self.set_fields(self.data_transform, fields)

    def serialize(self):
        return self.serialize_buffer(self.data_transform)

    def deserialize(self, data_transform_stream):
        self.deserialize_stream(self.data_transform, data_transform_stream)
