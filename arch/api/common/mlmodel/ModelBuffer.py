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
from arch.api.proto.model_param_pb2 import ModelParam
from arch.api.common.mlmodel.ProtoBuffer import ProtoBuffer


class ModelBuffer(ProtoBuffer):

    def __init__(self):
        self.meta = ModelMeta()
        self.param = ModelParam()

    def set_meta_field(self, name, value):
        self.set_field(self.meta, name, value)

    def set_param_field(self, name, value):
        self.set_field(self.param, name, value)

    def set_meta_fields(self, fields):
        self.set_fields(self.meta, fields)

    def set_param_fields(self, fields):
        self.set_fields(self.param, fields)

    def serialize(self):
        return self.serialize_buffer(self.meta), self.serialize_buffer(self.param)

    def deserialize(self, meta_stream, param_stream):
        self.deserialize_stream(self.meta, meta_stream)
        self.deserialize_stream(self.param, param_stream)
