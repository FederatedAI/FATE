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
from ..manager import SchemaManager, BlockManager

FRAME_SCHEME = "fate.arch.dataframe"


def build_schema(data):
    schema_manager = data.schema_manager
    block_manager = data.block_manager
    fields = schema_manager.serialize()
    for col_id, field in enumerate(fields):
        block_id = block_manager.get_block_id(col_id)[0]
        should_compress = block_manager.get_block(block_id).should_compress
        field["should_compress"] = should_compress

    built_schema = dict()
    built_schema["fields"] = fields
    built_schema["partition_order_mappings"] = data.partition_order_mappings
    built_schema["type"] = FRAME_SCHEME
    return built_schema


def parse_schema(schema):
    if schema.get("type") != FRAME_SCHEME:
        raise ValueError(f"deserialize data error, schema type is not {FRAME_SCHEME}")

    fields = schema["fields"]
    partition_order_mappings = schema["partition_order_mappings"]

    return fields, partition_order_mappings
