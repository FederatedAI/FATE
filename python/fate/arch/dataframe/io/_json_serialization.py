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
from fate.arch.context.io.data import df

from .._dataframe import DataFrame
from ._json_schema import build_schema, parse_schema
from ..ops import transform_block_to_list, transform_list_to_block
from ..manager import SchemaManager, BlockManager


def _serialize(ctx, data):
    """
    index, match_id, label, weight, values
    """
    # TODO: tensor does not provide method to get raw values directly, so we use .storages.blocks first
    schema = build_schema(data)

    serialize_data = transform_block_to_list(data.block_table, data.block_manager)
    serialize_data.schema = schema
    return serialize_data


def serialize(ctx, data):
    if isinstance(data, df.Dataframe):
        data = data.data

    return _serialize(ctx, data)


def deserialize(ctx, data):
    fields, partition_order_mappings = parse_schema(data.schema)

    schema_manager = SchemaManager.deserialize(fields)
    block_manager = BlockManager()
    block_manager.initialize_blocks(schema_manager)

    block_table = transform_list_to_block(data, block_manager)

    return DataFrame(ctx, block_table, partition_order_mappings, schema_manager, block_manager)
