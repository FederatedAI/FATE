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
FRAME_SCHEME = "fate.arch.dataframe"


def build_schema(data):
    meta = data.data_manager.serialize()

    built_schema = dict()
    built_schema["schema_meta"] = meta
    built_schema["partition_order_mappings"] = data.partition_order_mappings
    built_schema["type"] = FRAME_SCHEME

    return built_schema


def parse_schema(schema):
    if schema.get("type") != FRAME_SCHEME:
        raise ValueError(f"deserialize data error, schema type is not {FRAME_SCHEME}")

    schema_meta = schema["schema_meta"]
    partition_order_mappings = schema["partition_order_mappings"]

    return schema_meta, partition_order_mappings
